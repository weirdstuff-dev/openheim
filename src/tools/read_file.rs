//! Built-in tool: `read_file` — reads a file, or a range of its lines, and
//! returns the text.

use std::io;
use std::path::Path;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::fs;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, BufReader};

use crate::core::client_io::LineRange;
use crate::core::models::Tool;
use crate::core::turn::TurnContext;
use crate::error::{Error, Result};

use super::ToolHandler;
use super::args::parse;
use super::capabilities::{ToolCapabilities, ToolKindHint};

/// Most bytes of file text one `read_file` call returns. Below the agent
/// loop's cap on tool results, so the continuation marker is never cut off.
const MAX_READ_BYTES: usize = 100 * 1024;

/// How far past the lines shown a local read scans to count the file's
/// lines. Past that, the continuation marker leaves the total out.
const MAX_COUNT_SCAN_BYTES: u64 = 64 * 1024 * 1024;

fn not_utf8() -> Error {
    Error::ToolExecutionError("the file is not valid UTF-8 text".to_string())
}

/// Reads the whole of `path` (already resolved with
/// `TurnContext::resolve_path`) as UTF-8, asking `turn.client_io` first and
/// falling back to local disk. Fails without reading a local file over
/// `max_bytes`. The client is raced against `turn.cancel`, so a hung client
/// can't block cancellation. Used by `edit_file`, which needs the whole text.
pub(crate) async fn read_text(
    path: &Path,
    max_bytes: u64,
    turn: &TurnContext<'_>,
) -> Result<String> {
    let too_big = |size: u64| {
        Error::ToolExecutionError(format!(
            "the file is {size} bytes; only files up to {max_bytes} bytes can be read whole"
        ))
    };
    let from_client = tokio::select! {
        _ = turn.cancel.cancelled() => {
            return Err(Error::ToolExecutionError("file read cancelled".to_string()));
        }
        result = turn.client_io.read_file(path, LineRange::default()) => result,
    };
    if let Some(result) = from_client {
        let text = result?;
        if text.len() as u64 > max_bytes {
            return Err(too_big(text.len() as u64));
        }
        return Ok(text);
    }

    let size = fs::metadata(path).await?.len();
    if size > max_bytes {
        return Err(too_big(size));
    }
    // Bounded again in case the file grew since the size check.
    let mut bytes = Vec::new();
    fs::File::open(path)
        .await?
        .take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() as u64 > max_bytes {
        return Err(too_big(bytes.len() as u64));
    }
    String::from_utf8(bytes).map_err(|_| not_utf8())
}

/// Some lines of a file, as read by [`read_excerpt`].
#[derive(Debug, PartialEq)]
struct Excerpt {
    text: String,
    /// Number of the first line asked for.
    first: u64,
    /// Lines shown in full.
    shown: u64,
    /// Whether the last line in `text` is only the start of a line longer
    /// than the byte cap.
    cut_line: bool,
    /// Whether the file goes on after `text`.
    more: bool,
    /// How many lines the file has, if counted.
    total: Option<u64>,
}

/// Consumes the rest of the current line, newline included, scanning at
/// most `budget` bytes (reduced by what it scans). `Some(true)` if it
/// consumed anything, `Some(false)` at the end of the input, `None` if the
/// budget ran out first.
async fn skip_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    budget: &mut u64,
) -> io::Result<Option<bool>> {
    let mut consumed = false;
    loop {
        let buf = reader.fill_buf().await?;
        if buf.is_empty() {
            return Ok(Some(consumed));
        }
        if *budget == 0 {
            return Ok(None);
        }
        let window = &buf[..buf
            .len()
            .min(usize::try_from(*budget).unwrap_or(usize::MAX))];
        let (used, done) = match window.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (window.len(), false),
        };
        reader.consume(used);
        *budget -= used as u64;
        consumed = true;
        if done {
            return Ok(Some(true));
        }
    }
}

/// `begun` plus the lines left in `reader`, or `None` if that takes more
/// than [`MAX_COUNT_SCAN_BYTES`] of scanning. `mid_line` means the reader
/// is partway through a line `begun` already counts.
async fn count_lines<R: AsyncBufRead + Unpin>(
    mut reader: R,
    begun: u64,
    mid_line: bool,
) -> io::Result<Option<u64>> {
    let mut budget = MAX_COUNT_SCAN_BYTES;
    if mid_line && skip_line(&mut reader, &mut budget).await?.is_none() {
        return Ok(None);
    }
    let mut count = begun;
    loop {
        match skip_line(&mut reader, &mut budget).await? {
            Some(true) => count += 1,
            Some(false) => return Ok(Some(count)),
            None => return Ok(None),
        }
    }
}

/// Up to `limit` lines of `reader` from line `first` (1-based), holding at
/// most `max_bytes` of text. Stops before a line that doesn't fit, unless
/// it's the first one: that one is cut at the byte cap. With `count_rest`,
/// scans on to count the lines, up to [`MAX_COUNT_SCAN_BYTES`].
async fn read_excerpt<R: AsyncBufRead + Unpin>(
    mut reader: R,
    first: u64,
    limit: Option<u64>,
    max_bytes: usize,
    count_rest: bool,
) -> Result<Excerpt> {
    // Lines begun so far.
    let mut begun: u64 = 0;
    let mut unbounded = u64::MAX;
    while begun + 1 < first {
        match skip_line(&mut reader, &mut unbounded).await? {
            Some(true) => begun += 1,
            _ => break,
        }
    }

    let mut out = Vec::new();
    let mut shown = 0;
    let mut cut_line = false;
    let mut mid_line = false;
    while limit.is_none_or(|limit| shown < limit) && out.len() < max_bytes {
        let start = out.len();
        let budget = (max_bytes - start) as u64;
        if (&mut reader)
            .take(budget)
            .read_until(b'\n', &mut out)
            .await?
            == 0
        {
            break;
        }
        begun += 1;
        if out.last() == Some(&b'\n') || reader.fill_buf().await?.is_empty() {
            shown += 1;
            continue;
        }
        // The line doesn't fit in what's left of the cap. The first one is
        // shown cut; a later one is left for the next call.
        if shown == 0 {
            cut_line = true;
        } else {
            out.truncate(start);
        }
        mid_line = true;
        break;
    }

    let text = match String::from_utf8(out) {
        Ok(text) => text,
        // A cut line may end partway through a character.
        Err(e) if cut_line && e.utf8_error().error_len().is_none() => {
            let valid = e.utf8_error().valid_up_to();
            let mut bytes = e.into_bytes();
            bytes.truncate(valid);
            String::from_utf8(bytes).map_err(|_| not_utf8())?
        }
        Err(_) => return Err(not_utf8()),
    };

    let more = mid_line || !reader.fill_buf().await?.is_empty();
    let total = if count_rest {
        count_lines(reader, begun, mid_line).await?
    } else {
        None
    };

    Ok(Excerpt {
        text,
        first,
        shown,
        cut_line,
        more,
        total,
    })
}

/// `excerpt`'s text, followed by a note on how to read on when the file
/// continues past it.
fn render(excerpt: Excerpt) -> String {
    let Excerpt {
        mut text,
        first,
        shown,
        cut_line,
        more,
        total,
    } = excerpt;
    let of_total = total.map(|t| format!(" of {t}")).unwrap_or_default();

    if text.is_empty() && !more && first > 1 {
        return match total {
            Some(t) => format!("[offset {first} is past the end of the file, which has {t} lines]"),
            None => format!("[nothing at offset {first}: the file ends before it]"),
        };
    }
    let note = if cut_line {
        let rest = if total.is_none_or(|t| t > first) {
            format!(
                "; call again with offset={} for the lines after it",
                first + 1
            )
        } else {
            String::new()
        };
        Some(format!(
            "[line {first}{of_total} is longer than {MAX_READ_BYTES} bytes; only its start is shown{rest}]"
        ))
    } else if more {
        let last = first + shown - 1;
        Some(format!(
            "[lines {first}–{last}{of_total} shown; call again with offset={} to read on]",
            last + 1
        ))
    } else {
        None
    };
    if let Some(note) = note {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&note);
    }
    text
}

/// Reads a file, or `limit` lines of it from line `offset`, as UTF-8.
///
/// Returns at most [`MAX_READ_BYTES`] of text, streaming a local file
/// rather than loading it. When the file continues past what's returned,
/// the text ends with a note giving the `offset` to read on from. Returns
/// an error if the path is outside the work directory, the file does not
/// exist, cannot be read, or is not valid UTF-8.
pub struct ReadFileTool;

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
    /// `Option` so an explicit `null` means "unset", not a parse error.
    #[serde(default)]
    offset: Option<u32>,
    #[serde(default)]
    limit: Option<u32>,
}

#[async_trait]
impl ToolHandler for ReadFileTool {
    fn definition(&self) -> Tool {
        Tool::function(
            "read_file",
            "Read a text file. Returns up to 100 KB; a longer file ends with a note giving the offset to call again with. Use offset and limit to read a specific range of lines.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Line number to start reading from (1-based). Defaults to 1."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum number of lines to read. Defaults to as many as fit in 100 KB."
                    }
                },
                "required": ["path"]
            }),
        )
    }

    async fn execute(&self, args: &str, turn: &TurnContext<'_>) -> Result<String> {
        let args: ReadFileArgs = parse(args)?;
        if args.offset == Some(0) || args.limit == Some(0) {
            return Err(Error::InvalidArgument(
                "offset and limit must be at least 1".to_string(),
            ));
        }
        let validated = turn.resolve_path(&args.path)?;
        let first = u64::from(args.offset.unwrap_or(1));
        let limit = args.limit.map(u64::from);

        let lines = LineRange {
            line: args.offset,
            limit: args.limit,
        };
        let from_client = tokio::select! {
            _ = turn.cancel.cancelled() => {
                return Err(Error::ToolExecutionError("file read cancelled".to_string()));
            }
            result = turn.client_io.read_file(&validated, lines) => result,
        };
        let excerpt = match from_client {
            // The client already picked the lines; only the byte cap is
            // left to apply, and the file's length is unknown.
            Some(result) => {
                let text = result?;
                let mut excerpt =
                    read_excerpt(text.as_bytes(), 1, None, MAX_READ_BYTES, false).await?;
                excerpt.first = first;
                excerpt.more |= limit.is_some_and(|limit| excerpt.shown == limit);
                excerpt
            }
            None => {
                let file = fs::File::open(&validated).await?;
                read_excerpt(BufReader::new(file), first, limit, MAX_READ_BYTES, true).await?
            }
        };
        Ok(render(excerpt))
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            kind: ToolKindHint::Read,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tools::test_support::{
        FixedClientIo, HangingClientIo, RecordingClientIo, TurnHarness,
    };

    #[test]
    fn definition_has_correct_name() {
        let tool = ReadFileTool;
        let def = tool.definition();
        assert_eq!(def.function.name, "read_file");
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn execute_reads_existing_file() {
        let harness = TurnHarness::new();
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "hello world").unwrap();

        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let result = ReadFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn execute_resolves_relative_paths_against_work_dir() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("rel.txt"), "relative").unwrap();

        let result = ReadFileTool
            .execute(r#"{"path": "rel.txt"}"#, &harness.turn())
            .await
            .unwrap();
        assert_eq!(result, "relative");
    }

    #[tokio::test]
    async fn execute_resolves_relative_paths_against_the_working_directory() {
        let harness = TurnHarness::new().with_cwd("sub");
        std::fs::write(harness.work_dir().join("sub/rel.txt"), "in sub").unwrap();
        std::fs::write(harness.work_dir().join("top.txt"), "at the top").unwrap();

        let turn = harness.turn();
        let read = |path: &'static str| {
            let args = serde_json::json!({ "path": path }).to_string();
            let turn = &turn;
            async move { ReadFileTool.execute(&args, turn).await.unwrap() }
        };
        assert_eq!(read("rel.txt").await, "in sub");
        assert_eq!(read("../top.txt").await, "at the top");
    }

    #[tokio::test]
    async fn execute_errors_for_nonexistent_file() {
        let harness = TurnHarness::new();
        let result = ReadFileTool
            .execute(r#"{"path": "missing.txt"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn execute_rejects_path_outside_work_dir() {
        let harness = TurnHarness::new();
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "x").unwrap();

        let args = serde_json::json!({"path": secret.to_str().unwrap()}).to_string();
        let err = ReadFileTool
            .execute(&args, &harness.turn())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("outside the work directory"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn execute_errors_for_malformed_json() {
        let harness = TurnHarness::new();
        let result = ReadFileTool.execute("not json", &harness.turn()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    #[tokio::test]
    async fn execute_errors_for_missing_path() {
        let harness = TurnHarness::new();
        let result = ReadFileTool
            .execute(r#"{"other": "value"}"#, &harness.turn())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path"));
    }

    #[tokio::test]
    async fn read_file_prefers_client_io_over_local_disk() {
        let harness = TurnHarness::new().with_client_io(Arc::new(FixedClientIo("client content")));
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let result = ReadFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert_eq!(result, "client content");
    }

    #[tokio::test]
    async fn read_file_falls_back_to_local_disk_when_client_io_defers() {
        let harness = TurnHarness::new();
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let result = ReadFileTool.execute(&args, &harness.turn()).await.unwrap();
        assert_eq!(result, "local content");
    }

    #[tokio::test]
    async fn read_file_cancel_aborts_hanging_client_io() {
        let harness = TurnHarness::new().with_client_io(Arc::new(HangingClientIo));
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "local content").unwrap();

        let args = serde_json::json!({"path": path.to_str().unwrap()}).to_string();
        let cancel = harness.cancel_handle();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel.cancel();
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            ReadFileTool.execute(&args, &harness.turn()),
        )
        .await
        .expect("cancellation should abort the hanging client_io call");
        assert!(result.is_err());
    }

    /// `lines` numbered lines, `"line 1\n"` to `"line N\n"`.
    fn numbered(lines: u64) -> String {
        (1..=lines).map(|i| format!("line {i}\n")).collect()
    }

    async fn read(harness: &TurnHarness, args: serde_json::Value) -> Result<String> {
        ReadFileTool
            .execute(&args.to_string(), &harness.turn())
            .await
    }

    #[tokio::test]
    async fn offset_and_limit_pick_lines() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("a.txt"), numbered(10)).unwrap();

        let result = read(&harness, json!({"path": "a.txt", "offset": 3, "limit": 2}))
            .await
            .unwrap();
        assert_eq!(
            result,
            "line 3\nline 4\n[lines 3–4 of 10 shown; call again with offset=5 to read on]"
        );

        // Reading to the end needs no note.
        let result = read(&harness, json!({"path": "a.txt", "offset": 9}))
            .await
            .unwrap();
        assert_eq!(result, "line 9\nline 10\n");
        let result = read(&harness, json!({"path": "a.txt", "offset": 9, "limit": 2}))
            .await
            .unwrap();
        assert_eq!(result, "line 9\nline 10\n");
    }

    #[tokio::test]
    async fn last_line_without_a_newline_is_read() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("a.txt"), "one\ntwo").unwrap();

        let result = read(&harness, json!({"path": "a.txt", "offset": 2}))
            .await
            .unwrap();
        assert_eq!(result, "two");
    }

    #[tokio::test]
    async fn offset_past_the_end_says_so() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("a.txt"), numbered(3)).unwrap();

        let result = read(&harness, json!({"path": "a.txt", "offset": 7}))
            .await
            .unwrap();
        assert_eq!(
            result,
            "[offset 7 is past the end of the file, which has 3 lines]"
        );
    }

    #[tokio::test]
    async fn zero_offset_or_limit_is_rejected() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("a.txt"), "x").unwrap();
        for args in [
            json!({"path": "a.txt", "offset": 0}),
            json!({"path": "a.txt", "limit": 0}),
        ] {
            let err = read(&harness, args).await.unwrap_err();
            assert!(err.to_string().contains("at least 1"), "{err}");
        }
    }

    #[tokio::test]
    async fn large_file_stops_at_the_byte_cap_on_a_line_boundary() {
        let harness = TurnHarness::new();
        let content = numbered(100_000);
        std::fs::write(harness.work_dir().join("big.txt"), &content).unwrap();

        let result = read(&harness, json!({"path": "big.txt"})).await.unwrap();
        let (text, note) = result.rsplit_once('[').unwrap();
        assert!(text.len() <= MAX_READ_BYTES, "{}", text.len());
        assert!(content.starts_with(text));
        assert!(text.ends_with('\n'));
        let last = text.lines().count();
        assert_eq!(
            note,
            format!(
                "lines 1–{last} of 100000 shown; call again with offset={} to read on]",
                last + 1
            )
        );

        // Reading on picks up at the next line.
        let next = read(&harness, json!({"path": "big.txt", "offset": last + 1}))
            .await
            .unwrap();
        assert!(next.starts_with(&format!("line {}\n", last + 1)), "{next}");
    }

    #[tokio::test]
    async fn a_single_huge_line_is_cut_at_the_byte_cap() {
        let harness = TurnHarness::new();
        // Two-byte characters after one ASCII one, so the cap lands
        // mid-character.
        let huge = format!("a{}", "é".repeat(MAX_READ_BYTES));
        std::fs::write(
            harness.work_dir().join("one.txt"),
            format!("{huge}\nnext\n"),
        )
        .unwrap();

        let result = read(&harness, json!({"path": "one.txt"})).await.unwrap();
        let (text, note) = result.split_once("\n[").unwrap();
        assert_eq!(text, format!("a{}", "é".repeat((MAX_READ_BYTES - 1) / 2)));
        assert_eq!(
            note,
            format!(
                "line 1 of 2 is longer than {MAX_READ_BYTES} bytes; only its start is shown; call again with offset=2 for the lines after it]"
            )
        );
    }

    #[tokio::test]
    async fn non_utf8_is_rejected() {
        let harness = TurnHarness::new();
        std::fs::write(harness.work_dir().join("bin"), [0xff, 0xfe, 0x00]).unwrap();

        let err = read(&harness, json!({"path": "bin"})).await.unwrap_err();
        assert!(err.to_string().contains("not valid UTF-8"), "{err}");
    }

    #[tokio::test]
    async fn client_io_gets_the_line_range() {
        let client = Arc::new(RecordingClientIo::new("line 3\nline 4\n"));
        let harness = TurnHarness::new().with_client_io(client.clone());

        let result = read(&harness, json!({"path": "a.txt", "offset": 3, "limit": 2}))
            .await
            .unwrap();
        assert_eq!(
            result,
            "line 3\nline 4\n[lines 3–4 shown; call again with offset=5 to read on]"
        );
        read(&harness, json!({"path": "a.txt"})).await.unwrap();

        assert_eq!(
            *client.reads.lock().unwrap(),
            [
                LineRange {
                    line: Some(3),
                    limit: Some(2)
                },
                LineRange::default(),
            ]
        );
    }

    #[tokio::test]
    async fn client_io_text_is_capped_too() {
        let client = Arc::new(RecordingClientIo::new(numbered(100_000)));
        let harness = TurnHarness::new().with_client_io(client);

        let result = read(&harness, json!({"path": "a.txt"})).await.unwrap();
        let (text, note) = result.rsplit_once('[').unwrap();
        assert!(text.len() <= MAX_READ_BYTES);
        let last = text.lines().count();
        assert_eq!(
            note,
            format!(
                "lines 1–{last} shown; call again with offset={} to read on]",
                last + 1
            )
        );
    }

    #[tokio::test]
    async fn read_text_refuses_files_over_the_limit() {
        let harness = TurnHarness::new();
        let path = harness.work_dir().join("a.txt");
        std::fs::write(&path, "0123456789").unwrap();

        assert_eq!(
            read_text(&path, 10, &harness.turn()).await.unwrap(),
            "0123456789"
        );
        let err = read_text(&path, 9, &harness.turn()).await.unwrap_err();
        assert!(err.to_string().contains("10 bytes"), "{err}");
    }

    #[test]
    fn args_struct_matches_schema() {
        crate::tools::args::assert_args_match_schema::<ReadFileArgs>(
            &ReadFileTool,
            serde_json::json!({"path": "a.txt", "offset": 2, "limit": 10}),
        );
    }
}
