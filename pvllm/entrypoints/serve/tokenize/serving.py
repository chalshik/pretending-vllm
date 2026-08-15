"""Serving /tokenize and /detokenize.

Upstream: vllm/entrypoints/serve/tokenize/serving.py
Tier: C

R2.2. Useful to a product for reasons beyond curiosity: it is how a client checks a
prompt against the context limit before paying to submit it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pvllm.tokenizers.protocol import TokenizerLike


class TokenizeRequest(BaseModel):
    model: str
    prompt: str
    add_special_tokens: bool = True


class TokenizeResponse(BaseModel):
    count: int
    max_model_len: int
    tokens: list[int] = Field(default_factory=list)


class DetokenizeRequest(BaseModel):
    model: str
    tokens: list[int]


class DetokenizeResponse(BaseModel):
    prompt: str


class OpenAIServingTokenization:
    def __init__(self, tokenizer: TokenizerLike, max_model_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len

    def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        tokens = self.tokenizer.encode(
            request.prompt, add_special_tokens=request.add_special_tokens
        )
        return TokenizeResponse(
            count=len(tokens), max_model_len=self.max_model_len, tokens=tokens
        )

    def detokenize(self, request: DetokenizeRequest) -> DetokenizeResponse:
        return DetokenizeResponse(prompt=self.tokenizer.decode(request.tokens))
