import json
import logging
from typing import List

from mitmproxy import ctx, http

logger = logging.getLogger(__name__)


def parse_message(content: bytes) -> List[bytes]:
    segments: List[bytes] = []
    pos = 0
    marker = b"~m~"
    while pos < len(content):
        start = content.find(marker, pos)
        if start == -1:
            break

        len_start = start + len(marker)
        len_end = content.find(marker, len_start)
        if len_end == -1:
            raise ValueError("malformed message: missing closing marker")

        length_bytes = content[len_start:len_end]
        try:
            seg_len = int(length_bytes.decode("ascii"))
        except ValueError as exc:
            raise ValueError("malformed message: invalid length") from exc

        seg_start = len_end + len(marker)
        seg_end = seg_start + seg_len
        if seg_end > len(content):
            raise ValueError("malformed message: segment exceeds payload length")

        segments.append(content[seg_start:seg_end])
        pos = seg_end

    return segments


class TradingView:
    def load(self, loader):
        loader.add_option(
            name="file",
            typespec=str | None,
            default=None,
            help="Write log messages to file",
        )

    def websocket_message(self, flow: http.HTTPFlow):
        if flow.websocket is None:
            return

        message = flow.websocket.messages[-1]
        if message.from_client:
            return

        try:
            segments = parse_message(message.content)
        except ValueError:
            logger.exception("failed to parse message")
            segments = []

        for segment in segments:
            if segment.find(b"~position~") == -1:
                continue

            segment_data = json.loads(segment.decode("utf-8"))
            text = next(iter(segment_data["p"][1].values()))["ns"]["d"]

            container_data = json.loads(text)
            log_data = container_data["graphicsCmds"]["create"]["logs"]

            log_messages = []
            for log_item in log_data:
                for line in log_item["data"]:
                    log_message = line["m"]
                    logger.info(log_message)
                    log_messages.append(log_message)

            if ctx.options.file:
                with open(ctx.options.file, "a") as f:
                    for log_message in log_messages:
                        f.write(f"{log_message}\n")


addons = [
    TradingView(),
]
