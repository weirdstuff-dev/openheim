//! Built-in tool: `execute_command` — runs a shell command and returns its output.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::parse;
use super::capabilities::{ApprovalScope, ToolCapabilities, ToolKindHint};

/// Default wall-clock limit on a single command; past it the command's
/// process group is killed.
pub(crate) const DEFAULT_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// Default output cap per stream (stdout and stderr each).
pub(crate) const MAX_COMMAND_OUTPUT_BYTES: usize = 64 * 1024;

/// How long to wait for the child to disappear after a SIGKILL before giving
/// up on reaping it (the `kill_on_drop` backstop remains either way).
const REAP_TIMEOUT: Duration = Duration::from_secs(5);

/// How long after the shell exits to keep collecting output when a process
/// it started in the background still holds the pipes.
const BACKGROUND_OUTPUT_GRACE: Duration = Duration::from_millis(250);

/// How often a command's background processes are checked for having all
/// exited, so their process group is unlisted.
const BACKGROUND_GROUP_CHECK_INTERVAL: Duration = Duration::from_secs(1);

/// Appended to a command's stdout when processes it started in the
/// background are still running.
const BACKGROUND_NOTE: &str = "\n[background processes are still running; their further output is discarded, and they are killed when openheim exits]\n";

/// Process groups of the commands running in this process, for
/// [`kill_running_commands`].
#[cfg(target_family = "unix")]
static RUNNING_GROUPS: std::sync::Mutex<std::collections::BTreeSet<i32>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// Kills every shell command `execute_command` is running in this process,
/// each with its whole process group. Unix only; a no-op elsewhere.
///
/// For a binary's signal handler: each command runs in a process group of
/// its own, so a terminal's Ctrl-C doesn't reach it, and a process killed
/// by a signal runs no destructors, so without this the commands outlive it.
pub fn kill_running_commands() {
    #[cfg(target_family = "unix")]
    {
        let groups = RUNNING_GROUPS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for &pgid in groups.iter() {
            kill_group(pgid);
        }
    }
}

/// Sends SIGKILL to process group `pgid`.
#[cfg(target_family = "unix")]
fn kill_group(pgid: i32) {
    use nix::sys::signal::{Signal, killpg};
    use nix::unistd::Pid;
    if let Err(e) = killpg(Pid::from_raw(pgid), Signal::SIGKILL) {
        tracing::debug!("killpg({pgid}) failed: {e}");
    }
}

/// A spawned command's process group, listed in [`RUNNING_GROUPS`] while
/// this lives. Dropped before [`Self::reaped`] (the turn was dropped
/// mid-command, or the runtime shut down), it kills the whole group, not
/// just the child `kill_on_drop` reaches.
struct RunningGroup {
    #[cfg(target_family = "unix")]
    pgid: Option<i32>,
}

impl RunningGroup {
    /// Lists the group led by the child with process id `pid`. Pid 0 would
    /// name our own group, so it's never listed.
    fn register(pid: Option<u32>) -> Self {
        #[cfg(target_family = "unix")]
        {
            let pgid = pid.map(|pid| pid as i32).filter(|&pgid| pgid > 0);
            if let Some(pgid) = pgid {
                RUNNING_GROUPS
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(pgid);
            }
            Self { pgid }
        }
        #[cfg(not(target_family = "unix"))]
        {
            let _ = pid;
            Self {}
        }
    }

    /// Whether no process is left in the group (always `false` where groups
    /// aren't tracked).
    fn is_gone(&self) -> bool {
        #[cfg(target_family = "unix")]
        if let Some(pgid) = self.pgid {
            use nix::sys::signal::killpg;
            use nix::unistd::Pid;
            return killpg(Pid::from_raw(pgid), None) == Err(nix::errno::Errno::ESRCH);
        }
        false
    }

    /// The child has been reaped: its pid may be reused from now on, so the
    /// group is unlisted without being signalled again.
    fn reaped(&mut self) {
        #[cfg(target_family = "unix")]
        if let Some(pgid) = self.pgid.take() {
            RUNNING_GROUPS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&pgid);
        }
    }
}

impl Drop for RunningGroup {
    fn drop(&mut self) {
        #[cfg(target_family = "unix")]
        if let Some(pgid) = self.pgid {
            kill_group(pgid);
            self.reaped();
        }
    }
}

/// Knobs for [`run_command`].
pub(crate) struct RunCommandOptions<'a> {
    pub cwd: Option<&'a Path>,
    /// Turn cancellation; when it fires the command is killed as on timeout.
    /// `None` only in unit tests that exercise the timeout alone.
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
/// Windows), optionally in `cwd`. Returns stdout on success (then stderr,
/// labelled, if there was any), or an error carrying stdout and stderr on
/// a non-zero exit.
///
/// - **Timeout** — killed after `opts.timeout`; the error carries the output
///   collected so far.
/// - **Output cap** — each stream stops at `opts.max_output_bytes`, with a
///   truncation marker; the closed pipe makes further writes fail.
/// - **Cancellation** — `opts.cancel` firing kills it like the timeout.
/// - **Process group** — the child leads its own group (Unix), so
///   backgrounded grandchildren die with it on timeout or cancellation, and
///   its exit is always reaped.
/// - **Background processes** — returns shortly after the shell exits even
///   if processes it started in the background still hold its pipes; their
///   later output is discarded and they keep running until they exit or
///   this process does (see `drain_background`).
pub(crate) async fn run_command(command: &str, opts: &RunCommandOptions<'_>) -> Result<String> {
    #[cfg(target_family = "unix")]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    };
    #[cfg(target_family = "windows")]
    let mut cmd = {
        use std::os::windows::process::CommandExt;
        // Without this, spawning `cmd` allocates and briefly flashes a
        // console window even though stdio is fully piped below.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command).creation_flags(CREATE_NO_WINDOW);
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
    let mut group = RunningGroup::register(child.id());
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();

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
    // Scoped so the coroutine (and its borrows) drop when the select resolves.
    // The pipes outlive it, for `drain_background`; on the kill paths they're
    // closed first, so a still-writing grandchild dies of SIGPIPE instead of
    // outliving the turn.
    let outcome = {
        let collect = async {
            let (status, truncated) = {
                let reads = async {
                    tokio::join!(
                        read_capped(&mut stdout, &mut out_buf, opts.max_output_bytes),
                        read_capped(&mut stderr, &mut err_buf, opts.max_output_bytes),
                    )
                };
                tokio::pin!(reads);
                tokio::select! {
                    truncated = &mut reads => (child.wait().await, Some(truncated)),
                    // The shell exited with a pipe still open: a process it
                    // started in the background holds it. What the shell
                    // wrote is already in the pipes, so a short grace
                    // collects it.
                    status = child.wait() => {
                        let truncated =
                            tokio::time::timeout(BACKGROUND_OUTPUT_GRACE, &mut reads)
                                .await
                                .ok();
                        (status, truncated)
                    }
                }
            };
            // Unknown if the grace cut the reads short; a buffer at the cap
            // was clipped either way.
            (out_truncated, err_truncated) = truncated.unwrap_or((
                out_buf.len() >= opts.max_output_bytes,
                err_buf.len() >= opts.max_output_bytes,
            ));
            status
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
            let background = stdout.is_some() || stderr.is_some();
            if background {
                tokio::spawn(drain_background(stdout, stderr, group));
            } else {
                group.reaped();
            }
            let stdout_s = render_stream(&out_buf, out_truncated, opts.max_output_bytes);
            let stderr_s = render_stream(&err_buf, err_truncated, opts.max_output_bytes);
            let mut output = if status.success() {
                with_stderr(stdout_s, &stderr_s)
            } else {
                format!("Command failed:\nStdout: {stdout_s}\nStderr: {stderr_s}")
            };
            if background {
                output.push_str(BACKGROUND_NOTE);
            }
            if status.success() {
                Ok(output)
            } else {
                Err(Error::ToolExecutionError(output))
            }
        }
        Termination::Completed(Err(e)) => Err(Error::ToolExecutionError(format!(
            "Failed to wait for command: {}",
            e
        ))),
        Termination::TimedOut => {
            drop((stdout, stderr));
            kill_and_reap(&mut child, &mut group).await;
            // The `*_truncated` flags are only set once both reads finish,
            // which the timeout may have cut short; a buffer at the cap was
            // clipped either way.
            let out_capped = out_buf.len() >= opts.max_output_bytes;
            let err_capped = err_buf.len() >= opts.max_output_bytes;
            let stdout_s = render_stream(&out_buf, out_capped, opts.max_output_bytes);
            let stderr_s = render_stream(&err_buf, err_capped, opts.max_output_bytes);
            Err(Error::ToolExecutionError(format!(
                "Command timed out after {:?} and was killed.\nStdout: {}\nStderr: {}",
                opts.timeout, stdout_s, stderr_s
            )))
        }
        Termination::Cancelled => {
            drop((stdout, stderr));
            kill_and_reap(&mut child, &mut group).await;
            Err(Error::ToolExecutionError("Command cancelled.".to_string()))
        }
    }
}

/// Reads `pipe` into `buf` until EOF or until `cap` bytes have been collected,
/// returning whether the cap stopped it early. Either way the pipe is then
/// dropped (`*pipe` becomes `None`); on the truncated path that closes the
/// read end, so the command's further writes fail with `SIGPIPE`/`EPIPE`
/// instead of blocking or buffering without bound. Cancel-safe: if the
/// future is dropped mid-read, `*pipe` is still open and `buf` holds
/// everything read so far.
async fn read_capped<R: AsyncRead + Unpin>(
    pipe: &mut Option<R>,
    buf: &mut Vec<u8>,
    cap: usize,
) -> bool {
    let Some(reader) = pipe.as_mut() else {
        return false;
    };
    let mut chunk = [0u8; 8192];
    let truncated = loop {
        match reader.read(&mut chunk).await {
            Ok(0) => break false,
            Ok(n) => {
                let keep = (cap - buf.len()).min(n);
                buf.extend_from_slice(&chunk[..keep]);
                if buf.len() == cap {
                    break true;
                }
            }
            // A mid-stream read error still yields what came before it; the
            // exit status carries the failure.
            Err(_) => break false,
        }
    };
    *pipe = None;
    truncated
}

/// Reads `pipe` to EOF, discarding what it reads.
async fn discard<R: AsyncRead + Unpin>(pipe: Option<R>) {
    let Some(mut pipe) = pipe else {
        return;
    };
    let mut chunk = [0u8; 8192];
    while matches!(pipe.read(&mut chunk).await, Ok(n) if n > 0) {}
}

/// Keeps reading, and discarding, the pipes a command's background
/// processes still hold after its shell exited, so they neither die of
/// `SIGPIPE` nor block on a full pipe. `group` stays listed meanwhile, so
/// they're killed with this process ([`kill_running_commands`], or the
/// group's drop on runtime shutdown). It's unlisted once the pipes close,
/// or as soon as the group no longer exists, since its id may then be
/// reused.
async fn drain_background<O, E>(stdout: Option<O>, stderr: Option<E>, mut group: RunningGroup)
where
    O: AsyncRead + Unpin,
    E: AsyncRead + Unpin,
{
    let drain = async {
        tokio::join!(discard(stdout), discard(stderr));
    };
    tokio::pin!(drain);
    let group_gone = async {
        while !group.is_gone() {
            tokio::time::sleep(BACKGROUND_GROUP_CHECK_INTERVAL).await;
        }
    };
    let pipes_closed = tokio::select! {
        _ = &mut drain => true,
        _ = group_gone => false,
    };
    group.reaped();
    if !pipes_closed {
        drain.await;
    }
}

/// A successful command's result: its stdout, then its stderr under a
/// `Stderr:` label if it wrote any (warnings, progress, or tools that
/// report on stderr), so it isn't lost. Just stdout otherwise.
fn with_stderr(mut stdout: String, stderr: &str) -> String {
    if stderr.trim().is_empty() {
        return stdout;
    }
    if !stdout.is_empty() && !stdout.ends_with('\n') {
        stdout.push('\n');
    }
    stdout.push_str("Stderr: ");
    stdout.push_str(stderr);
    stdout
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

/// Kills the child's whole process group (Unix) or just the child (Windows)
/// and reaps it, giving up after [`REAP_TIMEOUT`] rather than hang the turn.
/// `group` is the child's; it's unlisted once the child is reaped.
async fn kill_and_reap(child: &mut tokio::process::Child, group: &mut RunningGroup) {
    // The child leads its own group, so its pid is the pgid. Pid 0 would
    // signal our own group, hence the `> 0`.
    #[cfg(target_family = "unix")]
    if let Some(pgid) = child.id().map(|pid| pid as i32).filter(|&pgid| pgid > 0) {
        kill_group(pgid);
    }
    // Fallback (and the Windows path): direct SIGKILL to the child itself.
    let _ = child.start_kill();
    match tokio::time::timeout(REAP_TIMEOUT, child.wait()).await {
        Ok(Ok(_)) => group.reaped(),
        _ => tracing::error!("command child did not exit after SIGKILL; abandoning reap"),
    }
}

/// Runs a shell command in the turn's working directory (`TurnContext::cwd`)
/// with [`run_command`]'s limits. Only registered when `allow_shell` is set.
/// Paths inside the command aren't confined to the work directory; that
/// needs OS-level sandboxing.
pub struct ExecuteCommandTool;

#[derive(Deserialize)]
struct ExecuteCommandArgs {
    command: String,
}

#[async_trait]
impl ToolHandler for ExecuteCommandTool {
    fn definition(&self) -> Tool {
        Tool::function(
            "execute_command",
            "Execute a shell command (e.g., ls, pwd, echo). Use this for listing directories and running system commands. Commands are killed after 120 seconds and output is truncated at 64 KiB per stream. A process started in the background with `&` keeps running after the call returns, but its later output isn't captured; redirect it to a file to read it.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }),
        )
    }

    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args: ExecuteCommandArgs = parse(args)?;
        run_command(
            &args.command,
            &RunCommandOptions {
                cwd: Some(turn.cwd),
                cancel: Some(turn.cancel),
                ..Default::default()
            },
        )
        .await
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            kind: ToolKindHint::Execute,
            // Approving one command must not silently cover every other
            // command — see `core::permission::approval_key`'s doc comment.
            approval_scope: ApprovalScope::ExactArguments,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::test_support::TurnHarness;

    #[test]
    fn definition_has_correct_name() {
        let tool = ExecuteCommandTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "execute_command");
        assert_eq!(def.tool_type, "function".to_string());
    }

    #[tokio::test]
    async fn execute_runs_simple_command() {
        let harness = TurnHarness::new();
        let result = ExecuteCommandTool
            .execute(r#"{"command": "echo hello"}"#, &harness.turn())
            .await
            .unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn execute_runs_in_work_dir() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("marker.txt"), "x").unwrap();
        let result = ExecuteCommandTool
            .execute(r#"{"command": "ls"}"#, &harness.turn())
            .await
            .unwrap();
        assert_eq!(result.trim(), "marker.txt");
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn execute_runs_in_the_working_directory() {
        let harness = TurnHarness::new().with_cwd("sub");
        std::fs::write(harness.work_dir().join("sub/inner.txt"), "x").unwrap();
        let result = ExecuteCommandTool
            .execute(r#"{"command": "ls"}"#, &harness.turn())
            .await
            .unwrap();
        assert_eq!(result.trim(), "inner.txt");
    }

    #[tokio::test]
    async fn execute_errors_for_failing_command() {
        let harness = TurnHarness::new();
        let result = ExecuteCommandTool
            .execute(
                r#"{"command": "ls /nonexistent_dir_12345"}"#,
                &harness.turn(),
            )
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Command failed:"));
    }

    // Warnings and progress a successful command writes to stderr reach the
    // model too.
    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn a_successful_command_keeps_its_stderr() {
        let opts = RunCommandOptions::default();
        let out = run_command("echo out; echo warn >&2", &opts).await.unwrap();
        assert_eq!(out, "out\nStderr: warn\n");

        let out = run_command("echo warn >&2", &opts).await.unwrap();
        assert_eq!(out, "Stderr: warn\n");

        let out = run_command("printf out; echo warn >&2", &opts)
            .await
            .unwrap();
        assert_eq!(out, "out\nStderr: warn\n");
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn a_successful_command_without_stderr_returns_just_stdout() {
        let out = run_command("echo out", &RunCommandOptions::default())
            .await
            .unwrap();
        assert_eq!(out, "out\n");
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let harness = TurnHarness::new();
        let result = ExecuteCommandTool
            .execute("bad json", &harness.turn())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_errors_for_missing_command() {
        let harness = TurnHarness::new();
        let result = ExecuteCommandTool
            .execute(r#"{"other": "value"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("command"));
    }

    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn execute_cancel_aborts_running_shell() {
        let harness = TurnHarness::new();
        let cancel = harness.cancel_handle();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel.cancel();
        });

        let result = tokio::time::timeout(
            Duration::from_secs(10),
            ExecuteCommandTool.execute(r#"{"command": "sleep 30"}"#, &harness.turn()),
        )
        .await
        .expect("turn cancel should abort the running command");
        assert!(result.is_err());
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
    async fn timeout_marks_capped_stream_when_reads_were_interrupted() {
        let opts = RunCommandOptions {
            timeout: Duration::from_millis(300),
            max_output_bytes: 100,
            ..Default::default()
        };
        // stdout blows past the cap immediately; the trailing sleep holds
        // both pipe ends open with stderr silent, so the read join is still
        // pending when the timeout fires — the cap state must come from the
        // buffer, not the flags that never got assigned.
        let err = run_command("head -c 200 /dev/zero | tr '\\0' x; sleep 30", &opts)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("timed out"), "unexpected error: {err}");
        assert!(
            err.contains("truncated"),
            "capped stream missing its truncation marker: {err}"
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

        let pid = read_pid(&dir.path().join("grandchild.pid")).await;
        assert!(
            exits_soon(pid).await,
            "grandchild `sleep 30` (pid {pid}) survived the group kill"
        );
    }

    // A turn dropped mid-command (or a runtime shutting down) takes the
    // command's whole group with it, not just the `sh` that `kill_on_drop`
    // reaches.
    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn dropping_a_running_command_kills_its_group() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_path_buf();
        let task = tokio::spawn(async move {
            let opts = RunCommandOptions {
                cwd: Some(&cwd),
                timeout: Duration::from_secs(60),
                ..Default::default()
            };
            run_command(
                "echo $$ > sh.pid; sleep 30 & echo $! > grandchild.pid; wait",
                &opts,
            )
            .await
        });
        let sh = read_pid(&dir.path().join("sh.pid")).await;
        let grandchild = read_pid(&dir.path().join("grandchild.pid")).await;
        assert!(
            running_groups().contains(&sh),
            "the group is listed while it runs"
        );

        task.abort();
        let _ = task.await;

        assert!(
            exits_soon(grandchild).await,
            "grandchild `sleep 30` (pid {grandchild}) outlived the dropped command"
        );
        assert!(
            !running_groups().contains(&sh),
            "the group is unlisted once dropped"
        );
    }

    // A process started with `&` keeps the pipes open; the call still
    // returns once the shell exits, and the process keeps running: it isn't
    // killed, and its later writes neither fail nor block.
    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn a_backgrounded_process_does_not_hold_up_the_call() {
        let dir = tempfile::tempdir().unwrap();
        let opts = RunCommandOptions {
            cwd: Some(dir.path()),
            timeout: Duration::from_secs(60),
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let out = run_command(
            "echo $$ > sh.pid; \
             (sleep 0.3; while true; do echo tick; sleep 0.05; done) & \
             echo $! > writer.pid; echo started",
            &opts,
        )
        .await
        .unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        assert!(out.starts_with("started\n"), "{out}");
        assert!(out.ends_with(BACKGROUND_NOTE), "{out}");

        let sh = read_pid(&dir.path().join("sh.pid")).await;
        let writer = read_pid(&dir.path().join("writer.pid")).await;
        // Past its first writes, after the call returned.
        tokio::time::sleep(Duration::from_millis(800)).await;
        assert!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(writer), None).is_ok(),
            "the background writer (pid {writer}) died"
        );
        assert!(
            running_groups().contains(&sh),
            "its group stays listed, to be killed with this process"
        );

        kill_group(sh);
        assert!(exits_soon(writer).await);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while running_groups().contains(&sh) {
            assert!(
                std::time::Instant::now() < deadline,
                "the group is unlisted once its processes exit"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    // A background process that doesn't hold the pipes changes nothing.
    #[cfg(target_family = "unix")]
    #[tokio::test]
    async fn a_redirected_background_process_gets_no_note() {
        let dir = tempfile::tempdir().unwrap();
        let opts = RunCommandOptions {
            cwd: Some(dir.path()),
            ..Default::default()
        };
        let out = run_command(
            "sleep 30 > /dev/null 2>&1 & echo $! > bg.pid; echo done",
            &opts,
        )
        .await
        .unwrap();
        assert_eq!(out, "done\n");
        let bg = read_pid(&dir.path().join("bg.pid")).await;
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(bg),
            nix::sys::signal::Signal::SIGKILL,
        );
    }

    #[cfg(target_family = "unix")]
    fn running_groups() -> std::collections::BTreeSet<i32> {
        RUNNING_GROUPS.lock().unwrap().clone()
    }

    /// The pid a command wrote to `path`, waiting up to 5s for it.
    #[cfg(target_family = "unix")]
    async fn read_pid(path: &Path) -> i32 {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok(text) = std::fs::read_to_string(path)
                && let Ok(pid) = text.trim().parse()
            {
                return pid;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no pid in {}",
                path.display()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Whether process `pid` is gone within 5s.
    #[cfg(target_family = "unix")]
    async fn exits_soon(pid: i32) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if matches!(
                nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
                Err(nix::errno::Errno::ESRCH)
            ) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }

    #[test]
    fn args_struct_matches_schema() {
        crate::tools::args::assert_args_match_schema::<ExecuteCommandArgs>(
            &ExecuteCommandTool,
            serde_json::json!({"command": "ls"}),
        );
    }
}
