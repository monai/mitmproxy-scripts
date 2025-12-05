import datetime
import textwrap
from pathlib import Path
from typing import Any, cast

import yaml
from httpx import Response as XResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateError
from mitmproxy import contentviews, ctx
from mitmproxy.http import HTTPFlow, Request, Response
from openai import OpenAI, Stream
from openai.types.chat.chat_completion import Choice
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

class LlamaCppChatCompletionMessage(ChatCompletionMessage):
    reasoning_content: str | None = None

class LlamaCppChoice(Choice):
    message: LlamaCppChatCompletionMessage


class OpenAIContentview(contentviews.Contentview):
    @property
    def name(self) -> str:
        return "OpenAI"

    def prettify_request(self, data: bytes, flow: HTTPFlow, request: Request) -> str:
        chat_completion_request = ChatCompletionsRequest.model_validate_json(data)

        out = yaml.dump(
            chat_completion_request.model_dump(),
            Dumper=MyDumper,
            sort_keys=False,
            allow_unicode=True,
        )

        return out

    def prettify_response(self, data: bytes, flow: HTTPFlow, response: Response) -> str:
        return "response"

    def prettify_streaming_response(
        self, data: bytes, flow: HTTPFlow, response: Response
    ) -> str:
        openai = OpenAI(api_key="dummy")
        x_response = XResponse(response.status_code, content=data)
        stream = Stream(cast_to=ChatCompletionChunk, response=x_response, client=openai)

        chunks = []
        contents = []
        choices: dict[int, LlamaCppChoice] = {}

        for chunk in stream:
            chunks.append(chunk.model_dump(exclude_unset=True))

            for choice in chunk.choices:
                if choice.index not in choices:
                    # TODO: check if role != "assistant"
                    message = LlamaCppChatCompletionMessage(role="assistant")
                    assembled_choice = LlamaCppChoice(
                        finish_reason="stop", index=choice.index, message=message
                    )
                    choices[choice.index] = assembled_choice

                assembled_choice = choices[choice.index]

                if choice.finish_reason is not None:
                    assembled_choice.finish_reason = choice.finish_reason

                delta = LlamaCppChoiceDelta.model_validate(choice.delta.model_dump())
                if delta.content is not None:
                    if assembled_choice.message.content is None:
                        assembled_choice.message.content = delta.content
                    else:
                        assembled_choice.message.content += delta.content

                if delta.reasoning_content is not None:
                    if assembled_choice.message.reasoning_content is None:
                        assembled_choice.message.reasoning_content = delta.reasoning_content
                    else:
                        assembled_choice.message.reasoning_content += delta.reasoning_content

                if delta.refusal is not None:
                    if assembled_choice.message.refusal is None:
                        assembled_choice.message.refusal = delta.refusal
                    else:
                        assembled_choice.message.refusal += delta.refusal

                if delta.tool_calls is not None:
                    if assembled_choice.message.tool_calls is None:
                        assembled_choice.message.tool_calls = []

                    for tool_call in delta.tool_calls:
                        if tool_call.index == len(assembled_choice.message.tool_calls):
                            # TODO: check if type != "function"
                            assembled_choice.message.tool_calls.append(
                                ChatCompletionMessageFunctionToolCall(
                                    id="",
                                    function=OAIFunction(arguments="", name=""),
                                    type="function",
                                )
                            )

                        assembled_tool_call = cast(
                            ChatCompletionMessageFunctionToolCall,
                            assembled_choice.message.tool_calls[tool_call.index],
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
                            assembled_tool_call.type = tool_call.type

            if len(chunk.choices) > 0:
                first_choice = chunk.choices[0]
                try:
                    delta = LlamaCppChoiceDelta.model_validate(
                        first_choice.delta.model_dump()
                    )

                    if delta.role is not None:
                        item = {
                            "role": delta.role,
                            "content": [],
                            "reasoning_content": [],
                            "tool_calls": [],
                        }
                        contents.append(item)

                    if delta.content is not None:
                        contents[-1]["content"].append(delta.content)

                    if delta.reasoning_content is not None:
                        contents[-1]["reasoning_content"].append(
                            delta.reasoning_content
                        )

                    if delta.tool_calls is not None:
                        first_tool_call = delta.tool_calls[0]
                        if first_tool_call.id is not None:
                            contents[-1]["tool_calls"].append(
                                f"{first_tool_call.id}: {first_tool_call.function.name} {first_tool_call.function.arguments}"
                            )
                        else:
                            contents[-1]["tool_calls"].append(
                                first_tool_call.function.arguments
                            )

                except Exception as e:
                    return str(e)

        out = {
            "choices": [v.model_dump(exclude_unset=True) for k, v in choices.items()],
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

            if isinstance(metadata.http_message, Request):
                return self.prettify_request(data, metadata.flow, metadata.http_message)

            if isinstance(metadata.http_message, Response):
                if metadata.content_type == "text/event-stream":
                    return self.prettify_streaming_response(
                        data, metadata.flow, metadata.http_message
                    )

                if metadata.content_type == "application/json":
                    return self.prettify_response(
                        data, metadata.flow, metadata.http_message
                    )

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


def strftime_now(format_str: str) -> str:
    return datetime.datetime.now().strftime(format_str)


def raise_exception(msg: str) -> None:
    raise TemplateError(msg)


# class LCChatCompletionsRequestJinja(contentviews.Contentview):
#     name = "chat completions request jinja"

#     def prettify(self, data: bytes, metadata: contentviews.Metadata) -> str:
#         request = ChatCompletionsRequest.model_validate_json(data)

#         request_data = request.model_dump()

#         env = Environment(
#             loader=FileSystemLoader(Path.cwd()), autoescape=select_autoescape()
#         )

#         env.globals["strftime_now"] = strftime_now
#         env.globals["raise_exception"] = raise_exception

#         template = env.get_template(ctx.options.jinja)

#         return template.render(messages=request_data["messages"])

#     def render_priority(self, data: bytes, metadata: contentviews.Metadata) -> float:
#         if not isinstance(metadata.flow, HTTPFlow):
#             return 0

#         if not isinstance(metadata.http_message, Request):
#             return 0

#         if (
#             metadata.content_type == "application/json"
#             and metadata.flow.request.path.endswith("/v1/chat/completions")
#             and ctx.options.jinja
#         ):
#             return 3
#         else:
#             return 0


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
