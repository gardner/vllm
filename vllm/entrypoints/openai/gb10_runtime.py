# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

_GB10_OPENAI_TOOL_CALLING_RUNTIME_MESSAGE = (
    "OpenAI tool-calling runtime is not supported on GB10/SM12x in this fork: "
    "--enable-auto-tool-choice, --tool-call-parser, --tool-parser-plugin, "
    "--tool-server, and request-level tools/tool_choice select frontend tool "
    "parsers, ToolParser.adjust_request schema injection, built-in/MCP tool "
    "sessions, output parsing, or tool-call streaming outside the validated "
    "native first-path NVFP4 serving release. Disable OpenAI tool calling on "
    "GB10 until native SM12x tool-calling correctness and runtime evidence "
    "exists."
)


def _is_gb10_sm12x_cuda_platform() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_cuda() and current_platform.is_device_capability_family(
        120
    )


def _server_arg_selects_tool_calling(args: Any) -> bool:
    return any(
        (
            bool(getattr(args, "enable_auto_tool_choice", False)),
            bool(getattr(args, "tool_call_parser", None)),
            bool(getattr(args, "tool_parser_plugin", "")),
            bool(getattr(args, "tool_server", None)),
        )
    )


def reject_gb10_openai_tool_calling_server_args(args: Any) -> None:
    if _is_gb10_sm12x_cuda_platform() and _server_arg_selects_tool_calling(args):
        raise ValueError(_GB10_OPENAI_TOOL_CALLING_RUNTIME_MESSAGE)


def gb10_openai_tool_calling_request_error(request: Any) -> str | None:
    if not _is_gb10_sm12x_cuda_platform():
        return None

    tools = getattr(request, "tools", None)
    tool_choice = getattr(request, "tool_choice", None)
    if tools and tool_choice != "none":
        return _GB10_OPENAI_TOOL_CALLING_RUNTIME_MESSAGE
    return None
