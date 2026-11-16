//! Fitting a request into the model's context window by leaving out the
//! oldest whole turns, then, if the turn in progress alone is still too
//! long, the outputs of its older tool results. Only the request changes;
//! the stored history is sent in full again whenever it fits.

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

/// What a tool result's content is replaced with when it's left out.
pub(crate) const OMITTED_OUTPUT: &str = "[tool output omitted to fit the context window]";

/// A request's history after fitting.
#[derive(Debug)]
pub(crate) struct Fitted {
    pub(crate) messages: Vec<Message>,
    /// How many history messages were left out.
    pub(crate) dropped: usize,
    /// How many tool results in the latest turn had their output left out.
    pub(crate) omitted_outputs: usize,
    /// Whether nothing more can be left out: every earlier turn is, and so
    /// is every tool output in the latest turn that may be.
    pub(crate) exhausted: bool,
}

/// Index of the last assistant message in `turn`: tool results from there
/// on are new to the model, so their output is never left out. 0 if none.
fn unseen_from(turn: &[Message]) -> usize {
    turn.iter()
        .rposition(|m| m.role == Role::Assistant)
        .unwrap_or(0)
}

/// Whether `turn` (the latest one) still has a tool result whose output
/// may be left out.
fn has_omittable_output(turn: &[Message]) -> bool {
    turn[..unseen_from(turn)]
        .iter()
        .flat_map(|m| &m.content)
        .any(|block| {
            matches!(block, ContentBlock::ToolResult { content, .. }
                if content.len() > OMITTED_OUTPUT.len())
        })
}

/// `history` cut to fit `max_bytes` (with `fixed_bytes` of system prompt and
/// tools on top), or `None` if it fits whole or nothing can be left out.
///
/// First leaves out the oldest turns. It cuts only just before a user
/// message, so a tool call and its results stay together, and never after
/// the latest one; the first kept user message gets a note saying how many
/// messages were left out. If the latest turn alone is still over the
/// limit, its tool results' outputs are replaced with [`OMITTED_OUTPUT`],
/// oldest first, each call keeping its (now short) result. Results after
/// the turn's last assistant message are kept: the model hasn't seen them.
fn trim(history: &[Message], fixed_bytes: usize, max_bytes: usize) -> Option<Fitted> {
    let sizes: Vec<usize> = history.iter().map(message_bytes).collect();
    let total: usize = fixed_bytes + sizes.iter().sum::<usize>();
    if total <= max_bytes {
        return None;
    }
    let latest = latest_turn_start(history);

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
    let mut size = fixed_bytes + sizes[cut..].iter().sum::<usize>();
    if cut > 0 {
        let note = omitted_note(cut);
        size += note.len();
        messages[0].content.insert(0, ContentBlock::from(note));
    }

    // Still over the limit only if even the latest turn alone is.
    let unseen = unseen_from(&messages);
    let mut omitted_outputs = 0;
    for block in messages[..unseen]
        .iter_mut()
        .flat_map(|m| m.content.iter_mut())
    {
        if size <= max_bytes {
            break;
        }
        if let ContentBlock::ToolResult { content, .. } = block
            && content.len() > OMITTED_OUTPUT.len()
        {
            size -= content.len() - OMITTED_OUTPUT.len();
            *content = OMITTED_OUTPUT.to_string();
            omitted_outputs += 1;
        }
    }

    if cut == 0 && omitted_outputs == 0 {
        return None;
    }
    let exhausted = cut == latest && !has_omittable_output(&messages);
    Some(Fitted {
        messages,
        dropped: cut,
        omitted_outputs,
        exhausted,
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
                omitted_outputs: 0,
                exhausted: latest_turn_start(history) == 0 && !has_omittable_output(history),
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
    /// returns whether a smaller one can still be built: `false` once
    /// nothing more could be left out of it.
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

    /// One turn that's a long tool loop: three calls with big results.
    fn tool_loop() -> Vec<Message> {
        let big = "x".repeat(1000);
        vec![
            Message::user("go"),
            calls("a"),
            answer("a", &big),
            calls("b"),
            answer("b", &big),
            calls("c"),
            answer("c", &big),
        ]
    }

    /// The output of every tool result in `messages`, in order.
    fn outputs(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .filter_map(|m| m.tool_result_block())
            .map(|r| r.content)
            .collect()
    }

    #[test]
    fn a_long_tool_loop_leaves_out_its_oldest_outputs_first() {
        let history = tool_loop();
        let big = "x".repeat(1000);
        // Room for everything but about one big output.
        let fitted = trim(&history, 0, total(&history) - 500).unwrap();

        assert_eq!(fitted.dropped, 0);
        assert_eq!(fitted.omitted_outputs, 1);
        assert_eq!(
            outputs(&fitted.messages),
            [OMITTED_OUTPUT, big.as_str(), big.as_str()]
        );
        assert!(!fitted.exhausted);
        assert_tool_pairs_intact(&fitted.messages);
    }

    // The results after the last call are new to the model: they stay even
    // when nothing else is left to leave out.
    #[test]
    fn the_newest_outputs_are_kept_and_then_it_is_exhausted() {
        let history = tool_loop();
        let big = "x".repeat(1000);
        let fitted = trim(&history, 0, 1).unwrap();

        assert_eq!(fitted.omitted_outputs, 2);
        assert_eq!(
            outputs(&fitted.messages),
            [OMITTED_OUTPUT, OMITTED_OUTPUT, big.as_str()]
        );
        assert!(fitted.exhausted);
        assert_tool_pairs_intact(&fitted.messages);
    }

    // Earlier turns go first; only then the latest turn's outputs.
    #[test]
    fn outputs_are_left_out_only_once_earlier_turns_are() {
        let mut history = history();
        history.truncate(8);
        history.extend(tool_loop());

        let fitted = trim(&history, 0, total(&history) - 500).unwrap();
        assert_eq!((fitted.dropped, fitted.omitted_outputs), (4, 0));

        let fitted = trim(&history, 0, 1).unwrap();
        assert_eq!((fitted.dropped, fitted.omitted_outputs), (8, 2));
        assert!(fitted.exhausted);
        assert_tool_pairs_intact(&fitted.messages);
    }

    // A provider overflow on a single long turn leaves out its outputs on
    // the retry instead of ending the turn.
    #[test]
    fn overflow_on_a_single_turn_leaves_out_outputs() {
        let history = tool_loop();
        let mut fitter = ContextFitter::new(None, &[], &[]);

        let fitted = fitter.fit(&history);
        assert!(!fitted.exhausted);
        let sent = fitter.request_bytes(&fitted.messages);
        assert!(fitter.record_overflow(&fitted, sent));

        let fitted = fitter.fit(&history);
        assert!(fitted.omitted_outputs > 0);
        assert!(fitter.request_bytes(&fitted.messages) < sent);
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
