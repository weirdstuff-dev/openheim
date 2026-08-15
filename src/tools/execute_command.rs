//! Built-in tool: `execute_command` — runs a shell command and returns its output.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::core::models::{FunctionDefinition, Tool};
use crate::error::{Error, Result};

use super::ToolHandler;

/// Default hard wall-clock limit on a single command. Anything still running
/// past this is killed (whole process group) and reported as an error, so a
/// `sleep infinity` can't pin the agent turn forever.
pub(crate) const DEFAULT_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// Default per-stream output cap (stdout and stderr each). Past the cap the
/// pipe is closed and the output is returned with a truncation marker, so a
/// chatty command can't balloon memory or the LLM context.
pub(crate) const MAX_COMMAND_OUTPUT_BYTES: usize = 64 * 1024;

/// How long to wait for the child to disappear after a SIGKILL before giving
/// up on reaping it (the `kill_on_drop` backstop remains either way).
const REAP_TIMEOUT: Duration = Duration::from_secs(5);

/// Knobs for [`run_command`]; see the [`DEFAULT_COMMAND_TIMEOUT`] and
/// [`MAX_COMMAND_OUTPUT_BYTES`] consts for the default values' rationale.
pub(crate) struct RunCommandOptions<'a> {
    pub cwd: Option<&'a Path>,
    /// Turn cancellation; when it fires the command is killed as on timeout.
    /// `None` where no token is reachable (the bare `ToolHandler` path — the
    /// trait has no turn context yet); the timeout still bounds execution.
    pub cancel: Option<&'a CancellationToken>,
    pub timeout: Duration,
    pub max_output_bytes: usize,
}

impl Default for RunCommandOptions<'_> {
    fn default() -> Self {
        Self {
            cwd: None,
            cancel: None,
            timeout: DEFAULT_COMMAND_TIMEOUT,
            max_output_bytes: MAX_COMMAND_OUTPUT_BYTES,
        }
    }
}

/// Runs `command` via the platform shell (`sh -c` on Unix, `cmd /C` on
/// Windows), optionally pinned to `cwd`. Returns stdout on success, or an
/// error carrying the combined stdout+stderr diagnostic on a non-zero exit.
///
/// Single source of truth for the `execute_command` behaviour, shared by
/// [`ExecuteCommandTool`] and [`crate::tools::SandboxedExecutor`] (which passes
/// the work directory as `cwd` so relative paths resolve inside the sandbox).
///
/// Hardening applied to every invocation:
///
/// - **Timeout** — killed after `opts.timeout`, so runaway commands can't hang
///   the turn. The error carries whatever output was collected first.
/// - **Output cap** — each stream stops at `opts.max_output_bytes` and is
///   returned with a truncation marker; the closed pipe makes further writes
///   fail with `SIGPIPE`/`EPIPE` instead of buffering without bound.
/// - **Cancellation** — an fired [`CancellationToken`](opts.cancel) kills the
///   command the same way the timeout does.
/// - **Process group + reaping** — the child leads its own process group
///   (Unix), so `sh -c "sleep 999 &"` grandchildren die with it, and the exit
///   is always waited on so no zombie is left behind.
pub(crate) async fn run_command(command: &str, opts: &RunCommandOptions<'_>) -> Result<String> {
    #[cfg(target_family = "unix")]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    };
    #[cfg(target_family = "windows")]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command);
        c
    };

    if let Some(dir) = opts.cwd {
        cmd.current_dir(dir);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Backstop for paths that return without an explicit wait; the normal
        // and kill paths both reap below.
        .kill_on_drop(true);
    #[cfg(target_family = "unix")]
    cmd.process_group(0);

    let mut child = cmd
        .spawn()
        .map_err(|e| Error::ToolExecutionError(format!("Failed to execute command: {}", e)))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let mut out_buf: Vec<u8> = Vec::new();
    let mut err_buf: Vec<u8> = Vec::new();
    let mut out_truncated = false;
    let mut err_truncated = false;

    enum Termination {
        Completed(std::io::Result<std::process::ExitStatus>),
        TimedOut,
        Cancelled,
    }

    // Both streams must be drained (or capped) concurrently with waiting:
    // pipe buffers are finite, and a full one would block the child forever.
    // Scoped so the coroutine (and its borrows) drop when the select resolves;
    // on the kill paths that also closes the pipes, so a still-writing
    // grandchild dies of SIGPIPE instead of outliving the turn.
    let outcome = {
        let collect = async {
            let (t_out, t_err) = tokio::join!(
                read_capped(stdout, &mut out_buf, opts.max_output_bytes),
                read_capped(stderr, &mut err_buf, opts.max_output_bytes),
            );
            out_truncated = t_out;
            err_truncated = t_err;
            child.wait().await
        };
        tokio::pin!(collect);

        tokio::select! {
            status = &mut collect => Termination::Completed(status),
            _ = tokio::time::sleep(opts.timeout) => Termination::TimedOut,
            _ = wait_for_cancel(opts.cancel) => Termination::Cancelled,
        }
    };

    match outcome {
        Termination::Completed(Ok(status)) => {
            let stdout_s = render_stream(&out_buf, out_truncated, opts.max_output_bytes);
            let stderr_s = render_stream(&err_buf, err_truncated, opts.max_output_bytes);
            if status.success() {
                Ok(stdout_s)
            } else {
                Err(Error::ToolExecutionError(format!(
                    "Command failed:\nStdout: {}\nStderr: {}",
                    stdout_s, stderr_s
                )))
            }
        }
        Termination::Completed(Err(e)) => Err(Error::ToolExecutionError(format!(
            "Failed to wait for command: {}",
            e
        ))),
        Termination::TimedOut => {
            kill_and_reap(&mut child).await;
            let stdout_s = render_stream(&out_buf, out_truncated, opts.max_output_bytes);
            let stderr_s = render_stream(&err_buf, err_truncated, opts.max_output_bytes);
            Err(Error::ToolExecutionError(format!(
                "Command timed out after {:?} and was killed.\nStdout: {}\nStderr: {}",
                opts.timeout, stdout_s, stderr_s
            )))
        }
        Termination::Cancelled => {
            kill_and_reap(&mut child).await;
            Err(Error::ToolExecutionError("Command cancelled.".to_string()))
        }
    }
}

/// Reads `pipe` into `buf` until EOF or until `cap` bytes have been collected,
/// returning whether the cap stopped it early. On the truncated path the pipe
/// is dropped, closing the read end, so the command's further writes fail with
/// `SIGPIPE`/`EPIPE` instead of blocking or buffering without bound.
async fn read_capped<R: AsyncRead + Unpin>(pipe: Option<R>, buf: &mut Vec<u8>, cap: usize) -> bool {
    let Some(mut pipe) = pipe else {
        return false;
    };
    let mut chunk = [0u8; 8192];
    loop {
        match pipe.read(&mut chunk).await {
            Ok(0) => return false,
            Ok(n) => {
                let keep = (cap - buf.len()).min(n);
                buf.extend_from_slice(&chunk[..keep]);
                if buf.len() == cap {
                    return true;
                }
            }
            // A mid-stream read error still yields what came before it; the
            // exit status carries the failure.
            Err(_) => return false,
        }
    }
}

/// Decodes a collected stream, appending a truncation marker if the cap
/// clipped it, so the LLM knows the output is partial.
fn render_stream(buf: &[u8], truncated: bool, cap: usize) -> String {
    let mut s = String::from_utf8_lossy(buf).to_string();
    if truncated {
        s.push_str(&format!("\n[output truncated at {} bytes]\n", cap));
    }
    s
}

/// Resolves only if `cancel` is `Some` and its token fires; a `None` cancel
/// never resolves, keeping the `select!` arm inert.
async fn wait_for_cancel(cancel: Option<&CancellationToken>) {
    match cancel {
        Some(token) => token.cancelled().await,
        None => std::future::pending::<()>().await,
    }
}

/// Kills the child's whole process group (Unix) or just the child (Windows),
/// then waits so the exit is reaped instead of leaving a zombie. Bounded: if
/// something somehow survives SIGKILL, give up rather than hang the turn —
/// `kill_on_drop` remains as a backstop.
async fn kill_and_reap(child: &mut tokio::process::Child) {
    #[cfg(target_family = "unix")]
    {
        use nix::sys::signal::{Signal, killpg};
        use nix::unistd::Pid;
        // `process_group(0)` at spawn made the child its own group leader, so
        // its pid doubles as the pgid and the group signal reaches every
        // descendant (`sh -c "sleep 999 &"` included). The `> 0` filter guards
        // against pid 0, which would otherwise signal *our* group.
        if let Some(pgid) = child.id().map(|pid| pid as i32).filter(|&pgid| pgid > 0)
            && let Err(e) = killpg(Pid::from_raw(pgid), Signal::SIGKILL)
        {
            tracing::debug!("killpg({pgid}) failed: {e}");
        }
    }
    // Fallback (and the Windows path): direct SIGKILL to the child itself.
    let _ = child.start_kill();
    if tokio::time::timeout(REAP_TIMEOUT, child.wait())
        .await
        .is_err()
    {
        tracing::error!("command child did not exit after SIGKILL; abandoning reap");
    }
}

/// Executes an arbitrary shell command and returns stdout on success, or a
/// combined stdout+stderr diagnostic string on failure.
///
/// Uses `sh -c` on Unix and `cmd /C` on Windows. Non-zero exit codes produce
/// a descriptive string rather than an error so the LLM can interpret and react
/// to the failure output.
pub struct ExecuteCommandTool;

#[async_trait]
impl ToolHandler for ExecuteCommandTool {
    fn definition(&self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "execute_command".to_string(),
                description: "Execute a shell command (e.g., ls, pwd, echo). Use this for listing directories and running system commands. Commands are killed after 120 seconds and output is truncated at 64 KiB per stream.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute"
                        }
                    },
                    "required": ["command"]
                }),
            },
        }
    }

    async fn execute(&self, args: &str) -> Result<String> {
        let args: serde_json::Value = serde_json::from_str(args)
            .map_err(|e| Error::ParseError(format!("Failed to parse tool arguments: {}", e)))?;

        let command = args["command"]
            .as_str()
            .ok_or_else(|| Error::ParseError("Missing 'command' argument".to_string()))?;

        // No cancellation token here: `ToolHandler::execute` has no turn
        // context to take one from. The timeout, output cap, and process
        // group handling still apply; wiring the token through belongs to
        // threading `TurnContext` into the trait.
        run_command(command, &RunCommandOptions::default()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let tool = ExecuteCommandTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "execute_command");
        assert_eq!(def.tool_type, "function".to_string());
    }

    #[tokio::test]
    async fn execute_runs_simple_command() {
        let tool = ExecuteCommandTool;
        let args = r#"{"command": "echo hello"}"#;
        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[tokio::test]
    async fn execute_errors_for_failing_command() {
        let tool = ExecuteCommandTool;
        let args = r#"{"command": "ls /nonexistent_dir_12345"}"#;
        let result = tool.execute(args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Command failed:"));
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let tool = ExecuteCommandTool;
        let result = tool.execute("bad json").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_missing_command() {
        let tool = ExecuteCommandTool;
        let result = tool.execute(r#"{"other": "value"}"#).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("command"));
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn run_command_kills_command_on_timeout() {
        let opts = RunCommandOptions {
            timeout: Duration::from_millis(200),
            ..Default::default()
        };
        let result = tokio::time::timeout(Duration::from_secs(10), run_command("sleep 30", &opts))
            .await
            .expect("timeout should kill the command instead of hanging");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("timed out"), "unexpected error: {err}");
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn timeout_error_carries_partial_output() {
        let opts = RunCommandOptions {
            timeout: Duration::from_millis(300),
            ..Default::default()
        };
        let err = run_command("echo partial-output-marker; sleep 30", &opts)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("timed out"), "unexpected error: {err}");
        assert!(
            err.contains("partial-output-marker"),
            "unexpected error: {err}"
        );
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn output_beyond_the_cap_is_truncated() {
        let opts = RunCommandOptions {
            max_output_bytes: 100,
            ..Default::default()
        };
        // 2000 bytes of 'x' — past the cap, but small enough to fit any pipe
        // buffer, so the pipeline's exit status is deterministic (success).
        let output = match run_command("head -c 2000 /dev/zero | tr '\\0' x", &opts).await {
            Ok(s) => s,
            Err(e) => e.to_string(),
        };
        assert!(output.contains("truncated"), "unexpected output: {output}");
        assert_eq!(
            output.matches('x').count(),
            100,
            "unexpected output: {output}"
        );
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn cancellation_kills_running_command() {
        let token = CancellationToken::new();
        let delayed = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            delayed.cancel();
        });
        let opts = RunCommandOptions {
            cancel: Some(&token),
            // Long enough that only cancellation can end this.
            timeout: Duration::from_secs(60),
            ..Default::default()
        };
        let result = tokio::time::timeout(Duration::from_secs(10), run_command("sleep 30", &opts))
            .await
            .expect("cancellation should beat the 60s timeout");
        assert!(
            result.unwrap_err().to_string().contains("cancelled"),
            "should report cancellation"
        );
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn timeout_kills_grandchildren_too() {
        let dir = tempfile::tempdir().unwrap();
        let opts = RunCommandOptions {
            cwd: Some(dir.path()),
            timeout: Duration::from_millis(300),
            ..Default::default()
        };
        // The backgrounded `sleep` is a grandchild of `sh`; without the
        // process-group kill it would outlive the timeout.
        let result = run_command("sleep 30 & echo $! > grandchild.pid; wait", &opts).await;
        assert!(result.is_err());

        let pid: i32 = std::fs::read_to_string(dir.path().join("grandchild.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let mut gone = false;
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            let exists = !matches!(
                nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
                Err(nix::errno::Errno::ESRCH)
            );
            if !exists {
                gone = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            gone,
            "grandchild `sleep 30` (pid {pid}) survived the group kill"
        );
    }
}
