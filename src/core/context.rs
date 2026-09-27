//! Fitting a request into the model's context window by leaving out the
//! oldest whole turns. Only the request changes; the stored history is sent
//! in full again whenever it fits.

use crate::core::models::{ContentBlock, Message, Role, Tool, Usage};

/// Share of the context window a request may fill before older turns are
/// left out. The rest is headroom for the estimate's error and the reply.
const FILL_RATIO: f64 = 0.9;

/// Bytes per token assumed until a call in the turn reports real usage.
const DEFAULT_BYTES_PER_TOKEN: f64 = 4.0;

/// After an overflow, the next attempt aims this far below the rejected
/// request's size.
const OVERFLOW_SHRINK: f64 = 0.75;

/// Retries of one LLM call after the provider rejected it as too long.
pub(crate) const MAX_OVERFLOW_RETRIES: u32 = 2;

/// Per-message framing (role, block tags) not counted in its content.
const MESSAGE_OVERHEAD_BYTES: usize = 16;

/// What an image counts as. Providers bill images by resolution, not by
/// their base64 size, which would overstate them many times over.
const IMAGE_BYTES: usize = 6_000;

/// The note that opens the first kept turn when earlier ones are left out.
fn omitted_note(dropped: usize) -> String {
    format!("[{dropped} earlier messages omitted to fit the context window]")
}

/// Rough size of `message` on the wire, in bytes.
fn message_bytes(message: &Message) -> usize {
    let content: usize = message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => text.len(),
            ContentBlock::Thinking {
                thinking,
                signature,
            } => thinking.len() + signature.as_ref().map_or(0, String::len),
            ContentBlock::RedactedThinking { data } => data.len(),
            ContentBlock::Image { .. } => IMAGE_BYTES,
            ContentBlock::ToolUse {
                id,
                name,
                arguments,
                signature,
            } => {
                id.len() + name.len() + arguments.len() + signature.as_ref().map_or(0, String::len)
            }
            ContentBlock::ToolResult {
                tool_call_id,
                tool_name,
                content,
                ..
            } => tool_call_id.len() + tool_name.len() + content.len(),
        })
        .sum();
    content + MESSAGE_OVERHEAD_BYTES
}

/// Rough size of the tool definitions sent with every request, in bytes.
fn tools_bytes(tools: &[Tool]) -> usize {
    serde_json::to_string(tools).map_or(0, |json| json.len())
}

/// Index of the latest user message, where the turn in progress starts;
/// nothing from there on is ever left out. 0 if there's none.
fn latest_turn_start(messages: &[Message]) -> usize {
    messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or(0)
}

/// A request's history after fitting.
#[derive(Debug)]
pub(crate) struct Fitted {
    pub(crate) messages: Vec<Message>,
    /// How many history messages were left out.
    pub(crate) dropped: usize,
    /// Whether everything before the latest turn is already left out, so
    /// no smaller request can be built.
    pub(crate) exhausted: bool,
}

/// `history` cut to fit `max_bytes` (with `fixed_bytes` of system prompt and
/// tools on top), or `None` if it fits whole or nothing can be left out.
///
/// Cuts only just before a user message, so a tool call and its results stay
/// together, and never after the latest one. The first kept user message
/// gets a note saying how many messages were left out. If even the latest
/// turn alone is over the limit, everything before it is left out.
fn trim(history: &[Message], fixed_bytes: usize, max_bytes: usize) -> Option<Fitted> {
    let sizes: Vec<usize> = history.iter().map(message_bytes).collect();
    let total: usize = fixed_bytes + sizes.iter().sum::<usize>();
    let latest = latest_turn_start(history);
    if total <= max_bytes || latest == 0 {
        return None;
    }

    let mut remaining = total;
    let mut cut = latest;
    for (i, message) in history.iter().enumerate().take(latest + 1) {
        if i > 0 && message.role == Role::User && remaining + omitted_note(i).len() <= max_bytes {
            cut = i;
            break;
        }
        remaining -= sizes[i];
    }

    let mut messages = history[cut..].to_vec();
    messages[0]
        .content
        .insert(0, ContentBlock::from(omitted_note(cut)));
    Some(Fitted {
        messages,
        dropped: cut,
        exhausted: cut == latest,
    })
}

/// Keeps the requests of one turn inside the model's context window.
///
/// With a known window, a request estimated at over [`FILL_RATIO`] of it
/// leaves out the oldest turns. The estimate counts bytes and converts them
/// with the ratio the turn's last call reported, or [`DEFAULT_BYTES_PER_TOKEN`]
/// before any call has. Without a known window only a provider's overflow
/// error ([`Self::record_overflow`]) makes it trim.
pub(crate) struct ContextFitter {
    window: Option<u64>,
    /// Bytes every request carries besides the history: system prompt and
    /// tool definitions.
    fixed_bytes: usize,
    bytes_per_token: f64,
    /// Ceiling learned from overflow errors this turn.
    overflow_limit: Option<usize>,
}

impl ContextFitter {
    /// A fitter for requests that carry `system` (the system prompt, if
    /// any) and `tools` besides the history.
    pub(crate) fn new(window: Option<u64>, system: &[Message], tools: &[Tool]) -> Self {
        Self {
            window,
            fixed_bytes: system.iter().map(message_bytes).sum::<usize>() + tools_bytes(tools),
            bytes_per_token: DEFAULT_BYTES_PER_TOKEN,
            overflow_limit: None,
        }
    }

    fn max_bytes(&self) -> Option<usize> {
        let from_window = self
            .window
            .map(|tokens| (tokens as f64 * FILL_RATIO * self.bytes_per_token) as usize);
        match (from_window, self.overflow_limit) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// `history` as it should be sent: whole, or with the oldest turns left out.
    pub(crate) fn fit(&self, history: &[Message]) -> Fitted {
        self.max_bytes()
            .and_then(|max| trim(history, self.fixed_bytes, max))
            .unwrap_or_else(|| Fitted {
                messages: history.to_vec(),
                dropped: 0,
                exhausted: latest_turn_start(history) == 0,
            })
    }

    /// Size of `request` (as fitted) with the system prompt and tools, in bytes.
    pub(crate) fn request_bytes(&self, request: &[Message]) -> usize {
        self.fixed_bytes + request.iter().map(message_bytes).sum::<usize>()
    }

    /// Calibrates the byte-to-token ratio from a call that sent
    /// `sent_bytes` and reported `usage`.
    pub(crate) fn record_usage(&mut self, sent_bytes: usize, usage: &Usage) {
        let prompt_tokens =
            usage.input_tokens + usage.cache_creation_tokens + usage.cache_read_tokens;
        if prompt_tokens > 0 {
            self.bytes_per_token = (sent_bytes as f64 / prompt_tokens as f64).clamp(1.0, 8.0);
        }
    }

    /// Notes that the request `sent` (of `sent_bytes`) was too long, and
    /// returns whether a smaller one can still be built: `false` once only
    /// the latest turn was left.
    pub(crate) fn record_overflow(&mut self, sent: &Fitted, sent_bytes: usize) -> bool {
        if sent.exhausted {
            return false;
        }
        let limit = (sent_bytes as f64 * OVERFLOW_SHRINK) as usize;
        self.overflow_limit = Some(self.overflow_limit.map_or(limit, |l| l.min(limit)));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calls(id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::tool_use(id, "read_file", "{}")],
        }
    }

    fn answer(id: &str, content: &str) -> Message {
        Message::tool_result(id, "read_file", content, false)
    }

    /// Three turns: two finished ones with a tool call each, then the
    /// latest, still running.
    fn history() -> Vec<Message> {
        let big = "x".repeat(1000);
        vec![
            Message::user("first"),
            calls("a"),
            answer("a", &big),
            Message::assistant("done a"),
            Message::user("second"),
            calls("b"),
            answer("b", &big),
            Message::assistant("done b"),
            Message::user("third"),
            calls("c"),
            answer("c", "small"),
        ]
    }

    fn total(messages: &[Message]) -> usize {
        messages.iter().map(message_bytes).sum()
    }

    /// Every `ToolUse` in `messages` is answered by the message right after
    /// its own, and every result follows its call.
    fn assert_tool_pairs_intact(messages: &[Message]) {
        for (i, message) in messages.iter().enumerate() {
            if let Some(result) = message.tool_result_block() {
                let call = &messages[i - 1];
                assert_eq!(call.tool_calls()[0].id, result.tool_call_id);
            }
            for call in message.tool_calls() {
                let result = messages[i + 1].tool_result_block().unwrap();
                assert_eq!(result.tool_call_id, call.id);
            }
        }
    }

    fn note_of(message: &Message) -> String {
        match &message.content[0] {
            ContentBlock::Text { text } => text.clone(),
            other => panic!("expected the note first, got {other:?}"),
        }
    }

    #[test]
    fn history_that_fits_is_left_alone() {
        let history = history();
        assert!(trim(&history, 0, total(&history)).is_none());
    }

    #[test]
    fn oldest_turn_is_dropped_first() {
        let history = history();
        let fitted = trim(&history, 0, total(&history) - 1).unwrap();

        assert_eq!(fitted.dropped, 4);
        assert_eq!(fitted.messages.len(), history.len() - 4);
        assert_eq!(fitted.messages[0].role, Role::User);
        assert_eq!(
            note_of(&fitted.messages[0]),
            "[4 earlier messages omitted to fit the context window]"
        );
        // The note opens the first kept turn; its own text follows.
        assert_eq!(fitted.messages[0].content[1], ContentBlock::from("second"));
        assert_eq!(fitted.messages[1..], history[5..]);
        assert_tool_pairs_intact(&fitted.messages);
    }

    #[test]
    fn latest_turn_is_kept_even_when_it_alone_is_too_big() {
        let history = history();
        let fitted = trim(&history, 0, 1).unwrap();

        assert_eq!(fitted.dropped, 8);
        assert_eq!(fitted.messages[1..], history[9..]);
        assert!(note_of(&fitted.messages[0]).starts_with("[8 earlier"));
        assert_eq!(fitted.messages[0].content[1], ContentBlock::from("third"));
        assert_tool_pairs_intact(&fitted.messages);
    }

    #[test]
    fn fixed_bytes_count_against_the_limit() {
        let history = history();
        let max = total(&history);
        assert!(trim(&history, 0, max).is_none());
        assert_eq!(trim(&history, 10, max).unwrap().dropped, 4);
    }

    #[test]
    fn a_single_turn_is_never_cut() {
        let history = vec![Message::user("only"), calls("a"), answer("a", "big")];
        assert!(trim(&history, 0, 1).is_none());
    }

    #[test]
    fn cuts_only_before_user_messages() {
        let history = history();
        for max in 0..total(&history) {
            if let Some(fitted) = trim(&history, 0, max) {
                assert!(matches!(fitted.dropped, 4 | 8), "cut at {}", fitted.dropped);
                assert_tool_pairs_intact(&fitted.messages);
            }
        }
    }

    #[test]
    fn fitter_without_a_window_sends_everything() {
        let history = history();
        let fitted = ContextFitter::new(None, &[], &[]).fit(&history);
        assert_eq!(fitted.dropped, 0);
        assert!(!fitted.exhausted);
        assert_eq!(fitted.messages, history);
    }

    #[test]
    fn fitter_trims_to_the_window() {
        let history = history();
        // At 4 bytes per token, room for the last two turns only.
        let window = (total(&history) - 1000) as u64 / 4 * 10 / 9;
        let fitted = ContextFitter::new(Some(window), &[], &[]).fit(&history);
        assert_eq!(fitted.dropped, 4);
        assert!(!fitted.exhausted);
    }

    #[test]
    fn system_prompt_and_tools_count_against_the_window() {
        let history = history();
        let window = (total(&history) as u64).div_ceil(4) * 10 / 9 + 1;
        assert_eq!(
            ContextFitter::new(Some(window), &[], &[])
                .fit(&history)
                .dropped,
            0
        );

        let system = [Message {
            role: Role::System,
            content: vec![ContentBlock::from("s".repeat(500))],
        }];
        let tools = [Tool::function(
            "read_file",
            "d".repeat(500),
            serde_json::json!({}),
        )];
        assert_eq!(
            ContextFitter::new(Some(window), &system, &[])
                .fit(&history)
                .dropped,
            4
        );
        assert_eq!(
            ContextFitter::new(Some(window), &[], &tools)
                .fit(&history)
                .dropped,
            4
        );
    }

    #[test]
    fn reported_usage_calibrates_the_estimate() {
        let history = history();
        let bytes = total(&history);
        let mut fitter = ContextFitter::new(Some(bytes as u64 / 2), &[], &[]);
        assert_eq!(fitter.fit(&history).dropped, 0);

        // Every byte turned out to be a token, so the history is twice the window.
        let usage = Usage {
            input_tokens: bytes as u64,
            ..Usage::default()
        };
        fitter.record_usage(bytes, &usage);
        assert_eq!(fitter.fit(&history).dropped, 8);
    }

    #[test]
    fn overflow_shrinks_later_requests() {
        let history = history();
        let mut fitter = ContextFitter::new(None, &[], &[]);

        let fitted = fitter.fit(&history);
        let sent = fitter.request_bytes(&fitted.messages);
        assert!(fitter.record_overflow(&fitted, sent));
        let fitted = fitter.fit(&history);
        assert_eq!(fitted.dropped, 4);

        let sent = fitter.request_bytes(&fitted.messages);
        assert!(fitter.record_overflow(&fitted, sent));
        let fitted = fitter.fit(&history);
        assert_eq!(fitted.dropped, 8);
        assert!(fitted.exhausted);

        // Only the latest turn is left: nothing more can go.
        let sent = fitter.request_bytes(&fitted.messages);
        assert!(!fitter.record_overflow(&fitted, sent));
    }

    #[test]
    fn overflow_on_a_single_turn_cannot_be_helped() {
        let history = vec![Message::user("only")];
        let mut fitter = ContextFitter::new(None, &[], &[]);
        let fitted = fitter.fit(&history);
        assert!(fitted.exhausted);
        assert!(!fitter.record_overflow(&fitted, 100));
    }
}
