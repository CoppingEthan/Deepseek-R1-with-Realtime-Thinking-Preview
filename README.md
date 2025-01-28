# Deepseek R1 Manifold Pipe with Real-Time Thinking

**Realtime preview of the thinking process for Deepseek R1 and Deepseek Chat support.**

```
"""
title: Deepseek R1 Manifold Pipe with Real-Time Thinking
authors: [MCode-Team, Ethan Copping]
author_url: [https://github.com/MCode-Team, https://github.com/CoppingEthan]
funding_url: https://github.com/open-webui
version: 0.2.0
required_open_webui_version: 0.5.0
license: MIT
environment_variables:
- DEEPSEEK_API_KEY (required)

# Notices
Deepseeks API continues to be unresponsive to many users for R1. Hopefully this clears up soon.
Function continues to work for Chat and Reasoner should work again soon.

# Acknowledgments
Thanks to MCode-Team, for sharing this code. This project builds upon their work and I hope it serves as a helpful modification.
"""

import os
import json
import logging
import requests
import aiohttp
import re
import time
from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Dict,
    AsyncIterator,
)
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    MODEL_MAX_TOKENS = {
        "deepseek-chat": 8192,
        "deepseek-reasoner": 8192,
    }

    class Valves(BaseModel):
        DEEPSEEK_BASE_URL: str = Field(
            default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            description="Your DeepSeek Base URL",
        )
        DEEPSEEK_API_KEY: str = Field(
            default=os.getenv("DEEPSEEK_API_KEY", ""),
            description="Your DeepSeek API key",
        )
        THINK_XML_TAG: str = Field(
            default=os.getenv("THINK_XML_TAG", "thinking"),
            description="XML tag used for thinking content",
        )

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "deepseek"
        self.name = "deepseek/"
        self.valves = self.Valves()
        self.request_id = None

    @staticmethod
    def get_model_id(model_name: str) -> str:
        """Extract just the base model name from any format"""
        return model_name.replace(".", "/").split("/")[-1]

    def get_deepseek_models(self) -> List[Dict[str, str]]:
        """Fetch available models from Deepseek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            response = requests.get(
                f"{self.valves.DEEPSEEK_BASE_URL}/models", headers=headers, timeout=10
            )
            response.raise_for_status()
            models_data = response.json()
            return [
                {"id": model["id"], "name": model["id"]}
                for model in models_data.get("data", [])
            ]
        except Exception as e:
            logging.error(f"Error getting models: {e}")
            return []

    def pipes(self) -> List[dict]:
        return self.get_deepseek_models()

    def format_thinking_tags(self, text: str) -> str:
        """Format content within thinking XML tags into markdown blockquotes."""
        pattern = r"<{}>(.*?)</{}>".format(
            re.escape(self.valves.THINK_XML_TAG), re.escape(self.valves.THINK_XML_TAG)
        )
        regex = re.compile(pattern, flags=re.DOTALL)

        def replacer(match):
            thinking_content = match.group(1).strip()
            formatted_lines = [f"> {line}" for line in thinking_content.splitlines()]
            return "\n".join(formatted_lines)

        formatted_text = regex.sub(replacer, text)
        return formatted_text

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, Generator, Iterator]:
        if not self.valves.DEEPSEEK_API_KEY:
            error_msg = "Error: DEEPSEEK_API_KEY is required"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}

        try:
            system_message, messages = pop_system_message(body.get("messages", []))
            if "model" not in body:
                raise ValueError("Model name is required")

            model_id = self.get_model_id(body["model"])
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(model_id, 8192)

            if system_message:
                messages.insert(0, {"role": "system", "content": str(system_message)})

            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": min(
                    body.get("max_tokens", max_tokens_limit), max_tokens_limit
                ),
                "temperature": float(body.get("temperature", 0.7)),
                "top_k": (
                    int(body.get("top_k", 40))
                    if body.get("top_k") is not None
                    else None
                ),
                "top_p": (
                    float(body.get("top_p", 0.9))
                    if body.get("top_p") is not None
                    else None
                ),
                "stream": body.get("stream", False),
            }
            payload = {k: v for k, v in payload.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }

            if __event_emitter__ and model_id == "deepseek-reasoner":
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "thinking...", "done": False},
                    }
                )

            if payload["stream"]:
                return self._stream_response(
                    url=f"{self.valves.DEEPSEEK_BASE_URL}/chat/completions",
                    headers=headers,
                    payload=payload,
                    __event_emitter__=__event_emitter__,
                    model_id=model_id,
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.valves.DEEPSEEK_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_msg = (
                            f"Error: HTTP {response.status}: {await response.text()}"
                        )
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": error_msg,
                                        "done": True,
                                    },
                                }
                            )
                        return {"content": error_msg, "format": "text"}

                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        message = result["choices"][0]["message"]
                        content = message.get("content") or ""
                        reasoning_content = message.get("reasoning_content") or ""
                        combined_content = ""
                        if reasoning_content:
                            combined_content += f"<{self.valves.THINK_XML_TAG}>\n{reasoning_content.strip()}\n</{self.valves.THINK_XML_TAG}>\n\n"
                        combined_content += content
                        final_response = self.format_thinking_tags(combined_content)
                        if __event_emitter__ and model_id == "deepseek-reasoner":
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": "Request completed successfully",
                                        "done": True,
                                    },
                                }
                            )
                        return final_response
                    return ""

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}

    async def _stream_response(
        self,
        url: str,
        headers: dict,
        payload: dict,
        __event_emitter__=None,
        model_id: str = "",
    ) -> AsyncIterator[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if __event_emitter__ and model_id == "deepseek-reasoner":
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": "thinking...", "done": False},
                            }
                        )
                    if response.status != 200:
                        error_msg = (
                            f"Error: HTTP {response.status}: {await response.text()}"
                        )
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": error_msg, "done": True},
                                }
                            )
                        yield error_msg
                        return

                    reasoning_content = ""
                    content = ""
                    is_thinking = False
                    last_status_update = time.time()
                    status_dots = 0

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})

                                    # Handle reasoning content (thinking phase)
                                    if (
                                        "reasoning_content" in delta
                                        and delta["reasoning_content"]
                                    ):
                                        if not is_thinking:
                                            is_thinking = True
                                            yield "> "  # Start blockquote only once

                                        delta_text = delta["reasoning_content"]
                                        # Handle new lines in thinking content
                                        if "\n" in delta_text:
                                            delta_text = delta_text.replace(
                                                "\n", "\n> "
                                            )
                                        yield delta_text
                                        reasoning_content += delta_text

                                    # Handle final response content
                                    if "content" in delta and delta["content"]:
                                        if is_thinking:
                                            is_thinking = False
                                            yield "\n\n"  # Add separation between thinking and response

                                        content_chunk = delta["content"]
                                        content += content_chunk
                                        yield content_chunk

                                    if (
                                        model_id == "deepseek-reasoner"
                                        and __event_emitter__
                                    ):
                                        current_time = time.time()
                                        if current_time - last_status_update > 1:
                                            status_dots = (status_dots % 3) + 1
                                            await __event_emitter__(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": f"thinking{'...'[:status_dots]}",
                                                        "done": False,
                                                    },
                                                }
                                            )
                                            last_status_update = current_time

                                    if (
                                        data["choices"][0].get("finish_reason")
                                        == "stop"
                                    ):
                                        if (
                                            __event_emitter__
                                            and model_id == "deepseek-reasoner"
                                        ):
                                            await __event_emitter__(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": "Request completed",
                                                        "done": True,
                                                    },
                                                }
                                            )
                                        break

                            except json.JSONDecodeError as e:
                                logging.error(
                                    f"Failed to parse streaming response: {e}"
                                )
                                continue

        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield error_msg

```
