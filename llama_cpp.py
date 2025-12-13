import datetime
import json
import textwrap
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateError
from mitmproxy import contentviews, ctx
from mitmproxy.http import HTTPFlow, Request, Response
from openai import OpenAI, Stream
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function as OAIFunction,
)
from pydantic import BaseModel


class MyEmitter(yaml.emitter.Emitter):
    def choose_scalar_style(self):
        if (
            isinstance(self.event, yaml.ScalarEvent)
            and self.event.style
            and self.event.style in "|>"
        ):
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

def yaml_dump(data: Any) -> str:
    return yaml.dump(
        data,
        Dumper=MyDumper,
        sort_keys=False,
        allow_unicode=True,
    )

class Message(BaseModel):
    role: str
    content: str | None = None


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
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]]


class LlamaCppChoiceDelta(ChoiceDelta):
    reasoning_content: str | None = None


class LlamaCppChatCompletionMessage(ChatCompletionMessage):
    reasoning_content: str | None = None


class LlamaCppChoice(Choice):
    message: LlamaCppChatCompletionMessage


def strftime_now(format_str: str) -> str:
    return datetime.datetime.now().strftime(format_str)


def raise_exception(msg: str) -> None:
    raise TemplateError(msg)


class OpenAIContentview(contentviews.Contentview):
    @property
    def name(self) -> str:
        return "OpenAI"

    @property
    def syntax_highlight(self) -> Literal["yaml"]:
        return "yaml"

    def prettify_request(
        self, data: bytes, flow: HTTPFlow, request: httpx.Request
    ) -> str:
        chat_completion_request = ChatCompletionsRequest.model_validate_json(data)
        request_data = chat_completion_request.model_dump(exclude_unset=True)

        if ctx.options.jinja:
            env = Environment(
                loader=FileSystemLoader(Path.cwd()), autoescape=select_autoescape()
            )

            env.globals["strftime_now"] = strftime_now
            env.globals["raise_exception"] = raise_exception

            template = env.get_template(ctx.options.jinja)

            return template.render(**request_data)

        return yaml_dump(request_data)

    def prettify_response(
        self, data: bytes, flow: HTTPFlow, response: httpx.Response
    ) -> str:
        out = ChatCompletion.model_validate_json(data)
        response_data = out.model_dump(exclude_unset=True)

        return yaml_dump(response_data)

    def prettify_streaming_response(
        self, data: bytes, flow: HTTPFlow, response: httpx.Response
    ) -> str:
        openai = OpenAI(api_key="dummy")
        stream = Stream(cast_to=ChatCompletionChunk, response=response, client=openai)

        chunks = []
        choices: list[LlamaCppChoice] = []

        for chunk in stream:
            chunks.append(chunk.model_dump(exclude_unset=True))

            for choice in chunk.choices:
                try:
                    assembled_choice = choices[choice.index]
                except IndexError:
                    if choice.index != len(choices):
                        msg = f"choice index is out of order: {choice.index}"
                        raise ValueError(msg)

                    if choice.delta.role != "assistant":
                        msg = f"unexpected role: {choice.delta.role}"
                        raise ValueError(msg)

                    message = LlamaCppChatCompletionMessage(role="assistant")
                    assembled_choice = LlamaCppChoice(
                        finish_reason="stop", index=choice.index, message=message
                    )

                    choices.append(assembled_choice)

                if choice.finish_reason is not None:
                    assembled_choice.finish_reason = choice.finish_reason

                delta = LlamaCppChoiceDelta.model_validate(choice.delta.model_dump())

                if delta.content is not None:
                    assembled_choice.message.content = (
                        assembled_choice.message.content or ""
                    ) + delta.content

                if delta.reasoning_content is not None:
                    assembled_choice.message.reasoning_content = (
                        assembled_choice.message.reasoning_content or ""
                    ) + delta.reasoning_content

                if delta.refusal is not None:
                    assembled_choice.message.refusal = (
                        assembled_choice.message.refusal or ""
                    ) + delta.refusal

                if delta.tool_calls is not None:
                    if assembled_choice.message.tool_calls is None:
                        assembled_choice.message.tool_calls = []

                    for tool_call in delta.tool_calls:
                        try:
                            assembled_tool_call = cast(
                                ChatCompletionMessageFunctionToolCall,
                                assembled_choice.message.tool_calls[tool_call.index],
                            )
                        except IndexError:
                            if tool_call.index != len(
                                assembled_choice.message.tool_calls
                            ):
                                msg = f"tool_call index is out of order: {tool_call.index}"
                                raise ValueError(msg)

                            assembled_tool_call = ChatCompletionMessageFunctionToolCall(
                                id="",
                                function=OAIFunction(arguments="", name=""),
                                type="function",
                            )

                            assembled_choice.message.tool_calls.append(
                                assembled_tool_call
                            )

                        if tool_call.id is not None:
                            assembled_tool_call.id += tool_call.id

                        if tool_call.function is not None:
                            if tool_call.function.arguments is not None:
                                assembled_tool_call.function.arguments += (
                                    tool_call.function.arguments
                                )

                            if tool_call.function.name is not None:
                                assembled_tool_call.function.name += (
                                    tool_call.function.name
                                )

                        if tool_call.type is not None:
                            if tool_call.type != "function":
                                msg = f"unexpected tool_call type: {tool_call.type}"
                                raise ValueError(msg)

                            assembled_tool_call.type = tool_call.type

        out = {
            "choices": [v.model_dump(exclude_unset=True) for v in choices],
            "chunks": chunks,
        }

        res = yaml.dump(
            out,
            Dumper=MyDumper,
            sort_keys=False,
            allow_unicode=True,
        )

        return res

    def prettify(self, data: bytes, metadata: contentviews.Metadata) -> str:
        try:
            if not isinstance(metadata.flow, HTTPFlow):
                raise TypeError("flow is not HTTPFlow")

            if metadata.http_message is None:
                raise ValueError("http_message is None")

            request = httpx.Request(
                metadata.flow.request.method, metadata.flow.request.url
            )

            if isinstance(metadata.http_message, Request):
                return self.prettify_request(data, metadata.flow, request)

            if isinstance(metadata.http_message, Response):
                response = httpx.Response(
                    metadata.http_message.status_code, content=data, request=request
                )

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    try:
                        error_data = json.loads(data)

                        return yaml_dump(error_data)
                    except json.JSONDecodeError:
                        openai = OpenAI(api_key="dummy")

                        raise openai._make_status_error_from_response(response)

                if metadata.content_type == "text/event-stream":
                    return self.prettify_streaming_response(
                        data, metadata.flow, response
                    )

                if metadata.content_type == "application/json":
                    return self.prettify_response(data, metadata.flow, response)

            return data.decode("utf-8")

        except Exception as e:
            return str(e)

    def render_priority(self, data: bytes, metadata: contentviews.Metadata) -> float:
        if not isinstance(metadata.flow, HTTPFlow):
            return 0

        if metadata.http_message is None:
            return 0

        if metadata.flow.request.path.endswith("/v1/chat/completions"):
            return 2

        return 0


class LlamaCpp:
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


contentviews.add(OpenAIContentview)

addons = [
    LlamaCpp(),
]
