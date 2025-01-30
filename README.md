# Deepseek R1 Reasoner and Chat with Realtime Thinking Preview

**Use Deepseek Reasoner & Chat with Realtime Preview of the Thinking Process**

```
"""
title: Deepseek R1 Reasoner and Chat with Realtime Thinking Preview
authors: Ethan Copping
author_url: https://github.com/CoppingEthan
version: 0.3.0
required_open_webui_version: 0.5.5
license: MIT
# Acknowledgments
Code used from MCode-Team & Zgccrui
"""

import json
import httpx
from typing import AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        DEEPSEEK_API_BASE_URL: str = Field(
            default="https://api.deepseek.com/v1", description="Base API endpoint URL"
        )
        DEEPSEEK_API_KEY: str = Field(
            default="", description="Authentication key for API access"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.thinking = -1
        self._emitter = None
        self.data_prefix = "data: "

    def pipes(self):
        try:
            headers = {"Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}"}
            resp = httpx.get(
                f"{self.valves.DEEPSEEK_API_BASE_URL}/models",
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                return [
                    {"id": m["id"], "name": m["id"]}
                    for m in resp.json().get("data", [])
                ]
        except Exception:
            pass
        return [
            {"id": "deepseek-chat", "name": "deepseek-chat"},
            {"id": "deepseek-reasoner", "name": "deepseek-reasoner"},
        ]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        self.thinking = -1
        self._emitter = __event_emitter__

        if not self.valves.DEEPSEEK_API_KEY:
            yield json.dumps({"error": "Missing API credentials"})
            return

        req_headers = {
            "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            request_data = body.copy()
            model_id = request_data["model"].split(".", 1)[-1]
            request_data["model"] = model_id
            is_reasoner = "reasoner" in model_id.lower()

            messages = request_data["messages"]
            for i in reversed(range(1, len(messages))):
                if messages[i - 1]["role"] == messages[i]["role"]:
                    alt_role = (
                        "user" if messages[i]["role"] == "assistant" else "assistant"
                    )
                    messages.insert(
                        i, {"role": alt_role, "content": "[Unfinished thinking]"}
                    )

            async with httpx.AsyncClient(http2=True) as client:
                async with client.stream(
                    "POST",
                    f"{self.valves.DEEPSEEK_API_BASE_URL}/chat/completions",
                    json=request_data,
                    headers=req_headers,
                    timeout=20,
                ) as resp:
                    if resp.status_code != 200:
                        error_content = (await resp.aread()).decode()[:200]
                        yield json.dumps(
                            {"error": f"API error {resp.status_code}: {error_content}"}
                        )
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue

                        stream_data = json.loads(line[6:])
                        choice = stream_data.get("choices", [{}])[0]

                        if choice.get("finish_reason"):
                            return

                        delta = choice.get("delta", {})

                        if is_reasoner:
                            state_marker = self._handle_state(delta)
                            if state_marker:
                                yield state_marker
                                if state_marker == "<think>":
                                    yield "\n"

                            content = delta.get("reasoning_content", "") or delta.get(
                                "content", ""
                            )
                            if content:
                                yield content
                        else:
                            content = delta.get("content", "")
                            if content:
                                yield content

        except Exception as e:
            yield json.dumps({"error": f"{type(e).__name__}: {str(e)}"})

    def _handle_state(self, delta: dict) -> str:
        if self.thinking == -1 and delta.get("reasoning_content"):
            self.thinking = 0
            return "<think>"

        if self.thinking == 0 and not delta.get("reasoning_content"):
            self.thinking = 1
            return "\n</think>\n\n"

        return ""


```
