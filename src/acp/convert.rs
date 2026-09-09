//! Converting ACP wire content into core [`ContentBlock`]s.

use agent_client_protocol::schema::ContentBlock as AcpContentBlock;

use crate::{
    core::models::ContentBlock,
    error::{Error, Result},
};

/// Converts an ACP `session/prompt` payload into content blocks for a new
/// user [`crate::core::models::Message`].
///
/// `Text` and `Image` pass through directly (images require the `image`
/// prompt capability, which `initialize` declares). `ResourceLink` agents
/// must support unconditionally per the ACP spec, but embedding its content
/// isn't implemented — it's surfaced as a text pointer instead, which the
/// agent's own file tools can follow if needed. `Audio` and embedded
/// `Resource` blocks (and anything the `#[non_exhaustive]` enum might add
/// later) aren't supported at all: reject loudly instead of silently
/// dropping part of the user's input.
pub(crate) fn convert_prompt_blocks(blocks: &[AcpContentBlock]) -> Result<Vec<ContentBlock>> {
    let mut out = Vec::new();
    for block in blocks {
        match block {
            AcpContentBlock::Text(t) => {
                out.push(ContentBlock::Text {
                    text: t.text.clone(),
                });
            }
            AcpContentBlock::Image(img) => {
                out.push(ContentBlock::Image {
                    data: img.data.clone(),
                    mime_type: img.mime_type.clone(),
                });
            }
            AcpContentBlock::ResourceLink(link) => {
                out.push(ContentBlock::Text {
                    text: format!("[referenced resource: {} ({})]", link.name, link.uri),
                });
            }
            AcpContentBlock::Audio(_) => {
                return Err(Error::Other(
                    "audio prompt content is not supported".to_string(),
                ));
            }
            AcpContentBlock::Resource(_) => {
                return Err(Error::Other(
                    "embedded resource prompt content is not supported".to_string(),
                ));
            }
            _ => {
                return Err(Error::Other("unsupported prompt content type".to_string()));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod prompt_block_tests {
    use super::*;
    use agent_client_protocol::schema::{
        AudioContent, EmbeddedResource, EmbeddedResourceResource, ImageContent, ResourceLink,
        TextContent, TextResourceContents,
    };

    #[test]
    fn text_passes_through() {
        let blocks = vec![AcpContentBlock::Text(TextContent::new("hello"))];
        let result = convert_prompt_blocks(&blocks).unwrap();
        assert_eq!(
            result,
            vec![ContentBlock::Text {
                text: "hello".into()
            }]
        );
    }

    #[test]
    fn image_passes_through() {
        let blocks = vec![AcpContentBlock::Image(ImageContent::new(
            "base64data",
            "image/png",
        ))];
        let result = convert_prompt_blocks(&blocks).unwrap();
        assert_eq!(
            result,
            vec![ContentBlock::Image {
                data: "base64data".into(),
                mime_type: "image/png".into(),
            }]
        );
    }

    #[test]
    fn resource_link_becomes_text_hint() {
        let blocks = vec![AcpContentBlock::ResourceLink(ResourceLink::new(
            "notes.txt",
            "file:///tmp/notes.txt",
        ))];
        let result = convert_prompt_blocks(&blocks).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(
            &result[0],
            ContentBlock::Text { text }
                if text.contains("notes.txt") && text.contains("file:///tmp/notes.txt")
        ));
    }

    #[test]
    fn audio_is_rejected() {
        let blocks = vec![AcpContentBlock::Audio(AudioContent::new(
            "base64data",
            "audio/wav",
        ))];
        assert!(convert_prompt_blocks(&blocks).is_err());
    }

    #[test]
    fn embedded_resource_is_rejected() {
        let blocks = vec![AcpContentBlock::Resource(EmbeddedResource::new(
            EmbeddedResourceResource::TextResourceContents(TextResourceContents::new(
                "content",
                "file:///tmp/notes.txt",
            )),
        ))];
        assert!(convert_prompt_blocks(&blocks).is_err());
    }
}
