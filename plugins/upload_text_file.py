from __future__ import annotations

from typing import Callable, Dict, List


def plugin_info() -> Dict:
    return {
        "description": "Append uploaded text file content to the system prompt/context.",
        "params": {
            "text_file": {
                "type": "file",
                "types": ["txt", "md", "csv", "log", "json"],
                "help": "Upload a text file. Its content will be appended to the system prompt for this turn.",
            },
            "attachment_prompt": {
                "type": "text",
                "default": "User attached text content which within `<attachment>` tag:\n<attachment>{attachment}</attachment>",
                "help": "Prompt template for wrapping the uploaded text. Use {attachment} as the placeholder.",
            },
            "as_system": {
                "type": "boolean",
                "default": True,
                "help": "If true, append to system prompt; otherwise append as an extra user message.",
            },
            "max_chars": {
                "type": "number",
                "default": 200000,
                "help": "Max characters to read from the file to avoid exceeding context.",
            },
        },
    }


def _read_uploaded_text_file(uploaded_file, max_chars: int) -> str:
    if uploaded_file is None:
        return ""
    try:
        # Try to decode as UTF-8; fall back to latin-1 to avoid exceptions
        content_bytes = uploaded_file.getvalue()
        text = content_bytes.decode("utf-8", errors="replace")
        if max_chars and max_chars > 0:
            text = text[:max_chars]
        return text
    except Exception:
        return ""


def process(
    model: str,
    messages: List[Dict],
    system_prompt: str,
    model_params: Dict,
    plugin_params: Dict,
    generate_func: Callable[[str, List[Dict], str, Dict], object],
    **kwargs,
):
    uploaded_file = plugin_params.get("text_file")
    attachment_prompt: str = plugin_params.get(
        "attachment_prompt",
        "User attached text content which within `<attachment>` tag:\n<attachment>{attachment}</attachment>",
    )
    as_system: bool = bool(plugin_params.get("as_system", True))
    max_chars: int = int(plugin_params.get("max_chars", 200000))

    attachment_text = _read_uploaded_text_file(uploaded_file, max_chars)

    updated_messages = messages
    updated_system_prompt = system_prompt

    if attachment_text:
        wrapped = attachment_prompt.replace("{attachment}", attachment_text)
        if as_system:
            updated_system_prompt = (
                f"{system_prompt}{wrapped}" if system_prompt else wrapped
            )
        else:
            updated_messages = messages + [{"role": "user", "content": wrapped}]

    for chunk in generate_func(
        model, updated_messages, updated_system_prompt, model_params
    ):
        yield chunk
