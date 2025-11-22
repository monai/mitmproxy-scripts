import datetime
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
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


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


class OpenAI(contentviews.Contentview):
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
            and metadata.flow.request.path.endswith("completions")
        ):
            return 3
        else:
            return 0


def strftime_now(format_str: str) -> str:
    return datetime.datetime.now().strftime(format_str)


def raise_exception(msg: str) -> None:
    raise TemplateError(msg)


class OpenAIJinja(contentviews.Contentview):
    def prettify(self, data: bytes, metadata: contentviews.Metadata) -> str:
        request = Request.model_validate_json(data)

        request_data = request.model_dump()

        env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())

        env.globals["strftime_now"] = strftime_now
        env.globals["raise_exception"] = raise_exception

        template = env.get_template("unsloth-gpt-oss-20b.jinja")

        print(request_data["messages"])

        try:
            out = template.render(messages=request_data["messages"])
        except Exception as e:
            print(e)
            return "Error: " + str(e)

        print(out)

        return out

    def render_priority(self, data: bytes, metadata: contentviews.Metadata) -> float:
        if not isinstance(metadata.flow, HTTPFlow):
            return 0

        if not isinstance(metadata.http_message, HTTPRequest):
            return 0

        if (
            metadata.content_type == "application/json"
            and metadata.flow.request.path.endswith("completions")
        ):
            return 2
        else:
            return 0


class Jinja:
    def __init__(self):
        self.num = 0

    def load(self, loader):
        loader.add_option(
            name="jinja",
            typespec=str,
            default=False,
            help="Use jinja template for chat",
        )

    def response(self, flow):
        if ctx.options.addheader:
            self.num = self.num + 1
            flow.response.headers["count"] = str(self.num)


contentviews.add(OpenAI)
contentviews.add(OpenAIJinja)
