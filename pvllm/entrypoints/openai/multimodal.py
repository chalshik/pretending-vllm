"""Turning OpenAI content parts into placeholders. R18.

Upstream: vllm/entrypoints/chat_utils.py (the multimodal-parsing half)
Tier: B

A chat message's `content` may be a string or a list of parts, and an
`{"type": "image_url"}` part is what makes a request multimodal. This turns those parts
into what the engine actually schedules on: a token sequence with placeholder runs, and
one `MultiModalFeatureSpec` per image saying where its run sits and what it hashes to.

**Nothing is fetched.** The URL is hashed, not retrieved -- and a data URL's payload is
hashed without being decoded as an image. That is the honest boundary for a simulator:
the engine's behaviour depends on how many tokens an image occupies and whether two
requests refer to the same one, and both are available without ever seeing a pixel.

The consequence is worth stating: this cannot tell you an image is malformed, too
large for the real model, or of an unsupported type. It tells you your plumbing,
scheduling, and caching behave.
"""

from __future__ import annotations

from typing import Any

from pvllm.multimodal.inputs import (
    PLACEHOLDER_TOKEN_ID,
    MultiModalFeatureSpec,
    content_hash,
)

#: How many prompt tokens an image occupies. A fixed cost, in the region real
#: vision-language models land in (LLaVA-1.5 uses 576, Qwen2-VL varies with
#: resolution). Fixed rather than derived, because deriving it would need the image.
DEFAULT_IMAGE_TOKENS = 256

#: Content part types OpenAI defines. `input_audio` and `video_url` are recognized so
#: the error names them rather than reporting an unknown type.
_KNOWN_PART_TYPES = ("text", "image_url", "input_audio", "video_url")


def parse_content(
    content: str | list[dict[str, Any]] | None,
    *,
    image_tokens: int = DEFAULT_IMAGE_TOKENS,
) -> tuple[str, list[tuple[int, MultiModalFeatureSpec]]]:
    """Split a message's content into text and image placeholders.

    Returns `(text, [(text_offset, feature)])`, where `text_offset` is the character
    position in the returned text at which the image's placeholder run belongs. The
    caller resolves that to a *token* position once the text is tokenized, because
    only it knows the tokenizer.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []

    pieces: list[str] = []
    features: list[tuple[int, MultiModalFeatureSpec]] = []
    length = 0

    for part in content:
        if not isinstance(part, dict):
            raise ValueError(
                f"a content part must be an object, got {type(part).__name__}"
            )
        kind = part.get("type")
        if kind == "text":
            text = part.get("text", "")
            pieces.append(text)
            length += len(text)
        elif kind == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if not url:
                raise ValueError("an image_url part requires image_url.url")
            features.append(
                (
                    length,
                    MultiModalFeatureSpec(
                        identifier=content_hash(url, "image"),
                        modality="image",
                        # Filled in by the caller once the text is tokenized; a
                        # character offset is not a token position.
                        position=0,
                        length=image_tokens,
                        num_embeds=image_tokens,
                    ),
                )
            )
        elif kind in _KNOWN_PART_TYPES:
            raise NotImplementedError(
                f"content parts of type {kind!r} are not implemented; only 'text' and "
                f"'image_url' are. Audio and video would need their own token counts "
                f"and encoder costs, and inventing those would produce plausible "
                f"scheduling behaviour for a modality nobody modeled."
            )
        else:
            raise ValueError(
                f"unknown content part type {kind!r}; expected one of "
                f"{list(_KNOWN_PART_TYPES)}"
            )

    return "".join(pieces), features


def build_multimodal_prompt(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    image_tokens: int = DEFAULT_IMAGE_TOKENS,
) -> tuple[list[int] | None, list[MultiModalFeatureSpec]]:
    """Token ids with placeholder runs, and the features describing them.

    Returns `(None, [])` when no message carries an image, so a text-only chat request
    takes exactly the path it did before multimodal existed -- the placeholder
    machinery costs one check on the common case.
    """
    if not _has_image(messages):
        return None, []

    token_ids: list[int] = []
    features: list[MultiModalFeatureSpec] = []

    for message in messages:
        role = message.get("role", "user")
        text, image_offsets = parse_content(
            message.get("content"), image_tokens=image_tokens
        )
        header = f"<|{role}|>\n"
        token_ids.extend(tokenizer.encode(header, add_special_tokens=False))

        cursor = 0
        for offset, feature in image_offsets:
            # Text before the image, then its placeholder run. Order matters: the
            # prefix cache hashes the sequence, so an image before its caption and
            # one after it are different prompts, as they should be.
            segment = text[cursor:offset]
            if segment:
                token_ids.extend(tokenizer.encode(segment, add_special_tokens=False))
            features.append(
                MultiModalFeatureSpec(
                    identifier=feature.identifier,
                    modality=feature.modality,
                    position=len(token_ids),
                    length=feature.length,
                    num_embeds=feature.num_embeds,
                )
            )
            token_ids.extend([PLACEHOLDER_TOKEN_ID] * feature.length)
            cursor = offset

        remainder = text[cursor:]
        if remainder:
            token_ids.extend(tokenizer.encode(remainder, add_special_tokens=False))
        token_ids.extend(tokenizer.encode("\n", add_special_tokens=False))

    token_ids.extend(tokenizer.encode("<|assistant|>\n", add_special_tokens=False))
    return token_ids, features


def _has_image(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in message["content"]
        )
        for message in messages
    )
