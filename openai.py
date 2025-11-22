import datetime
import textwrap
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateError
from mitmproxy import contentviews, ctx
from mitmproxy.http import HTTPFlow, Request, Response
from pydantic import BaseModel


class MyEmitter(yaml.emitter.Emitter):
    def choose_scalar_style(self):
        if self.event.style and self.event.style in "|>":
            return self.event.style

        return super().choose_scalar_style()


class MyDumper(yaml.Dumper, MyEmitter):
    pass


def str_presenter(dumper: yaml.Dumper, data: str):
    lines = data.splitlines()

    text = data
    if ctx.options.wrap:
        wrapped_lines = [textwrap.fill(line, ctx.options.wrap) for line in lines]
        text = "\n".join(wrapped_lines)

    style = "|" if len(lines) > 1 else None

    return dumper.represent_scalar("tag:yaml.org,2002:str", text, style=style)


yaml.add_representer(str, str_presenter)


class Message(BaseModel):
    role: str
    content: str


class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class Tool(BaseModel):
    type: str
    function: Function


class ChatCompletionsRequest(BaseModel):
    messages: list[Message]
    model: str
    temperature: float
    top_p: float
    tools: list[dict[str, Any]]


class ToolCallFunction(BaseModel):
    arguments: str
    name: str | None = None


class ToolCall(BaseModel):
    index: int
    function: ToolCallFunction
    id: str | None = None
    type: str | None = None


class Delta(BaseModel):
    content: str | None = None
    refusal: str | None = None
    role: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None


class Choice(BaseModel):
    delta: Delta
    finish_reason: str | None = None
    index: int
    logprobs: dict[str, Any] | None = None


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_token_details: dict[str, Any]
    prompt_token_details: dict[str, Any]


class Chunk(BaseModel):
    choices: list[Choice]
    created: int
    id: str
    model: str
    object: str
    service_tier: str | None = None
    usage: dict[str, Any] | None = None


class OpenAIRequest(contentviews.Contentview):
    def prettify(self, data: bytes, metadata: contentviews.Metadata) -> str:
        request = ChatCompletionsRequest.model_validate_json(data)

        out = yaml.dump(
            request.model_dump(),
            Dumper=MyDumper,
            sort_keys=False,
            allow_unicode=True,
        )

        return out

    def render_priority(self, data: bytes, metadata: contentviews.Metadata) -> float:
        if not isinstance(metadata.flow, HTTPFlow):
            return 0

        if not isinstance(metadata.http_message, Request):
            return 0

        if (
            metadata.content_type == "application/json"
            and metadata.flow.request.path.endswith("/v1/chat/completions")
        ):
            return 2
        else:
            return 0


def strftime_now(format_str: str) -> str:
    return datetime.datetime.now().strftime(format_str)


def raise_exception(msg: str) -> None:
    raise TemplateError(msg)


class OpenAIRequestJinja(contentviews.Contentview):
    def prettify(self, data: bytes, metadata: contentviews.Metadata) -> str:
        request = ChatCompletionsRequest.model_validate_json(data)

        request_data = request.model_dump()

        env = Environment(
            loader=FileSystemLoader(Path.cwd()), autoescape=select_autoescape()
        )

        env.globals["strftime_now"] = strftime_now
        env.globals["raise_exception"] = raise_exception

        template = env.get_template(ctx.options.jinja)

        return template.render(messages=request_data["messages"])

    def render_priority(self, data: bytes, metadata: contentviews.Metadata) -> float:
        if not isinstance(metadata.flow, HTTPFlow):
            return 0

        if not isinstance(metadata.http_message, Request):
            return 0

        if (
            metadata.content_type == "application/json"
            and metadata.flow.request.path.endswith("/v1/chat/completions")
            and ctx.options.jinja
        ):
            return 3
        else:
            return 0


class OpenAIResponseStreaming(contentviews.Contentview):
    def prettify(self, data: bytes, metadata: contentviews.Metadata) -> str:
        lines = data.decode("utf-8").splitlines()

        chunks = []
        for line in lines:
            line_data = line[6:]
            if line_data.startswith("[DONE]"):
                break

            if len(line_data) == 0:
                continue

            try:
                chunk: Chunk = Chunk.model_validate_json(line_data)
            except Exception as e:
                return str(e)

            chunks.append(chunk.model_dump(exclude_unset=True))

        res = yaml.dump(
            chunks,
            Dumper=MyDumper,
            sort_keys=False,
            allow_unicode=True,
        )

        return res

    def render_priority(self, data: bytes, metadata: contentviews.Metadata) -> float:
        if not isinstance(metadata.flow, HTTPFlow):
            return 0

        if not isinstance(metadata.http_message, Response):
            return 0

        if (
            metadata.content_type == "text/event-stream"
            and metadata.flow.request.path.endswith("/v1/chat/completions")
        ):
            return 2
        else:
            return 0


class OpenAI:
    def __init__(self):
        self.num = 0

    def load(self, loader):
        loader.add_option(
            name="wrap",
            typespec=int,
            default=0,
            help="Wrap text at N characters",
        )

        loader.add_option(
            name="jinja",
            typespec=str,
            default="",
            help="Use jinja template for chat",
        )


contentviews.add(OpenAIRequest)
contentviews.add(OpenAIRequestJinja)
contentviews.add(OpenAIResponseStreaming)

addons = [
    OpenAI(),
]
