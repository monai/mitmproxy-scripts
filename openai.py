import datetime
import textwrap
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateError
from mitmproxy import contentviews, ctx
from mitmproxy.http import HTTPFlow
from mitmproxy.http import Request as HTTPRequest
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


class Request(BaseModel):
    messages: list[Message]
    model: str
    temperature: float
    top_p: float
    tools: list[dict[str, Any]]


class OpenAIRequest(contentviews.Contentview):
    def prettify(self, data: bytes, metadata: contentviews.Metadata) -> str:
        request = Request.model_validate_json(data)

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

        if not isinstance(metadata.http_message, HTTPRequest):
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
        request = Request.model_validate_json(data)

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

        if not isinstance(metadata.http_message, HTTPRequest):
            return 0

        if (
            metadata.content_type == "application/json"
            and metadata.flow.request.path.endswith("/v1/chat/completions")
            and ctx.options.jinja
        ):
            return 3
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

addons = [
    OpenAI(),
]
