"""Helpers shared by the error path. C7.

Upstream: vllm/entrypoints/serve/utils/api_utils.py
Tier: B

`sanitize_message` lives here rather than beside the handlers for the reason upstream
puts it here: `create_error_response` needs it too, and the handlers import *from*
`error_response`, so a home in the handler module would be an import cycle.
"""

from __future__ import annotations

import re


def sanitize_message(message: str) -> str:
    """Strip memory addresses, traceback frames and filesystem paths from an error.

    Not cosmetic. The message goes to whoever called the endpoint, and the messages
    most likely to carry a path are the ones built from an arbitrary exception's
    `str` -- which is exactly what `to_error_response` does four times over.
    """
    message = re.sub(r" at 0x[0-9a-f]+>", ">", message)
    message = re.sub(r'\n?\s*File "[^"]+", line \d+, in \S+(\n\s+.*)?', "", message)
    message = re.sub(
        r"/(?:home|usr|opt|var|tmp|root|lib|mnt|srv)(?:/[\w.\-]+)+", "<path>", message
    )
    message = re.sub(r"(?:/[\w\-]+)+/[\w\-]+\.\w+", "<path>", message)
    return message.strip()
