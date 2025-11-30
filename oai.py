import datetime
import textwrap
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateError
from mitmproxy import contentviews, ctx
from mitmproxy.http import HTTPFlow, Request, Response
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)
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


class LlamaCppChoiceDelta(ChoiceDelta):
    reasoning_content: str | None = None


class LlamaCppChoice(Choice):
    delta: LlamaCppChoiceDelta # pyright: ignore[reportIncompatibleVariableOverride]


class LlamaCppChatCompletionChunk(ChatCompletionChunk):
    choices: list[LlamaCppChoice] # pyright: ignore[reportIncompatibleVariableOverride]


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
        dialog = []
        for line in lines:
            line_data = line[6:]
            if line_data.startswith("[DONE]"):
                break

            if len(line_data) == 0:
                continue

            try:
                chunk = LlamaCppChatCompletionChunk.model_validate_json(line_data)
            except Exception as e:
                return str(e)

            if len(chunk.choices) > 0:
                first_choice = chunk.choices[0]

                if first_choice.delta.role is not None:
                    item = {
                        "role": first_choice.delta.role,
                        "content": [],
                        "reasoning_content": [],
                    }
                    dialog.append(item)

                if first_choice.delta.content is not None:
                    dialog[-1]["content"].append(first_choice.delta.content)

                if first_choice.delta.reasoning_content is not None:
                    dialog[-1]["reasoning_content"].append(first_choice.delta.reasoning_content)

            chunks.append(chunk.model_dump(exclude_unset=True))

        for item in dialog:
            item["content"] = "".join(item["content"])
            item["reasoning_content"] = "".join(item["reasoning_content"])

        out = {
            "dialog": dialog,
            "chunks": chunks,
        }

        res = yaml.dump(
            out,
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
