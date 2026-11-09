"""The OpenAI error object, and the refusals the HTTP door names.

Every error the door answers is `{"error": {"message", "type", "param", "code"}}` with its HTTP status — the shape
the OpenAI API and its clients use (an `openai.APIStatusError` carries it). A parameter the engine does not read is
refused by name in `param`, never ignored.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ApiError(Exception):
    """An answer that is an error: its HTTP status and the OpenAI error object."""

    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 param: Optional[str] = None, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.type = type_
        self.param = param
        self.code = code

    def body(self) -> Dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


def unsupported(param: str, why: str) -> ApiError:
    """A parameter (or value) the engine does not honour: refused by name, before anything runs."""
    return ApiError(400, f"`{param}` is not supported by this server: {why}", "invalid_request_error",
                    param, "unsupported_parameter")


def not_served(endpoint: str, model: str, family: str, why: str) -> ApiError:
    return ApiError(409, f"{endpoint}: the loaded model {model} (family {family}) does not serve it: {why}",
                    "invalid_request_error", None, "model_does_not_serve_endpoint")


def model_not_found(asked: Any, served: str) -> ApiError:
    return ApiError(404, f"The model {asked!r} is not served here; this daemon serves {served!r} (one daemon, one "
                         f"model: start another with `neurobrix serve --model {asked}`).",
                    "invalid_request_error", "model", "model_not_found")

