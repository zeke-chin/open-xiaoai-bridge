"""xAI Realtime Voice WebSocket 协议客户端。"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from websockets.asyncio.client import ClientConnection, connect as ws_connect
from websockets.exceptions import ConnectionClosed

from core.utils.config import ConfigManager
from core.utils.logger import logger


EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]
_DEFAULT_INSTRUCTIONS = "你是一个有帮助的语音助手，请用简洁口语中文回答。"


@dataclass(frozen=True, slots=True)
class XaiRealtimeSettings:
    """单次 xAI Realtime 会话使用的配置快照。"""

    api_url: str
    api_key: str
    voice: str = "ara"
    instructions: str = _DEFAULT_INSTRUCTIONS
    sample_rate: int = 16000
    exit_keywords: tuple[str, ...] = ("退出", "停止", "再见")
    idle_timeout: float = 20.0
    aec: bool = True
    aec_delay_ms: int = 150
    greeting: bool = True

    @classmethod
    def from_config(
        cls,
        config: ConfigManager | None = None,
    ) -> "XaiRealtimeSettings":
        manager = config or ConfigManager.instance()
        raw = manager.get_app_config("xai", {})
        if not isinstance(raw, dict):
            raw = {}

        env_key = os.environ.get("XAI_API_KEY", "").strip()
        keywords = raw.get("exit_keywords", ["退出", "停止", "再见"])
        if not isinstance(keywords, (list, tuple)):
            keywords = ["退出", "停止", "再见"]

        settings = cls(
            api_url=str(
                raw.get(
                    "api_url",
                    "wss://api.x.ai/v1/realtime?model=grok-voice-latest",
                )
                or ""
            ).strip(),
            api_key=env_key or str(raw.get("api_key", "") or "").strip(),
            voice=str(raw.get("voice", "ara") or "ara").strip(),
            instructions=str(
                raw.get("instructions", _DEFAULT_INSTRUCTIONS)
                or _DEFAULT_INSTRUCTIONS
            ).strip(),
            sample_rate=int(raw.get("sample_rate", 16000)),
            exit_keywords=tuple(
                str(keyword).strip() for keyword in keywords if str(keyword).strip()
            ),
            idle_timeout=max(1.0, float(raw.get("idle_timeout", 20))),
            aec=bool(raw.get("aec", True)),
            aec_delay_ms=int(raw.get("aec_delay_ms", 150)),
            greeting=bool(raw.get("greeting", True)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        parsed = urlparse(self.api_url)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise ValueError("xai.api_url 必须是有效的 ws:// 或 wss:// 地址")
        if not self.api_key:
            raise ValueError("xAI API key 未配置，请设置 xai.api_key 或 XAI_API_KEY")
        if not self.voice:
            raise ValueError("xai.voice 不能为空")
        if self.sample_rate != 16000:
            raise ValueError("xai.sample_rate 首版固定为 16000")
        if not 0 <= self.aec_delay_ms <= 500:
            raise ValueError("xai.aec_delay_ms 必须在 0..500 之间")


class XaiRealtimeClient:
    """负责 xAI Realtime 协议收发，不承载设备音频策略。"""

    def __init__(
        self,
        settings: XaiRealtimeSettings,
        event_handler: EventHandler | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self._event_handler = event_handler
        self._ws: ClientConnection | None = None
        self._receive_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._closed = asyncio.Event()
        self._configured = False
        self._closing = False
        self.last_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self.is_connected

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._closed.is_set()

    async def connect(self, timeout: float = 15.0) -> None:
        """建立单次会话连接并等待 session.updated。"""
        if self.is_connected:
            return
        self._closing = False
        self._closed.clear()
        self._ready.clear()
        self._configured = False
        self.last_error = None

        self._ws = await ws_connect(
            self.settings.api_url,
            additional_headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            open_timeout=timeout,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=4 * 1024 * 1024,
            max_queue=32,
        )
        self._receive_task = asyncio.create_task(
            self._receive_loop(),
            name="xai-realtime-receive",
        )
        ready_waiter = asyncio.create_task(self._ready.wait())
        closed_waiter = asyncio.create_task(self._closed.wait())
        try:
            done, _ = await asyncio.wait(
                {ready_waiter, closed_waiter},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if ready_waiter in done and ready_waiter.result():
                return
            if closed_waiter in done:
                raise RuntimeError(self.last_error or "xAI WebSocket 在会话就绪前关闭")
            raise TimeoutError("等待 xAI session.updated 超时")
        except BaseException:
            await self.close()
            raise
        finally:
            for waiter in (ready_waiter, closed_waiter):
                if not waiter.done():
                    waiter.cancel()

    async def close(self) -> None:
        """关闭连接；不会自动重连。"""
        if self._closing:
            return
        self._closing = True
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception as exc:
                logger.debug(f"关闭 WebSocket 失败: {exc}", module="xAI Realtime")

        task = self._receive_task
        self._receive_task = None
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._ready.clear()
        self._closed.set()

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def append_audio(self, pcm: bytes) -> None:
        if not self.is_ready:
            raise RuntimeError("xAI session 尚未就绪，不能发送音频")
        if not pcm:
            return
        await self.send_event(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    async def cancel_response(self) -> None:
        if self.is_connected:
            await self.send_event({"type": "response.cancel"})

    async def send_event(self, event: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None or self._closed.is_set():
            raise RuntimeError("xAI WebSocket 未连接")
        await ws.send(json.dumps(event, ensure_ascii=False, separators=(",", ":")))

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except (TypeError, json.JSONDecodeError) as exc:
                    logger.warning(f"忽略无效 JSON 消息: {exc}", module="xAI Realtime")
                    continue
                if not isinstance(message, dict):
                    continue
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as exc:
            if not self._closing:
                self.last_error = f"WebSocket closed: {exc.code} {exc.reason}"
                logger.warning(self.last_error, module="xAI Realtime")
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"接收循环失败: {self.last_error}", module="xAI Realtime")
        finally:
            self._ready.clear()
            self._closed.set()
            if not self._closing:
                await self._dispatch(
                    {
                        "type": "client.closed",
                        "error": self.last_error,
                    }
                )

    async def _handle_message(self, message: dict[str, Any]) -> None:
        event_type = message.get("type")
        if event_type == "conversation.created" and not self._configured:
            self._configured = True
            await self.send_event(self._session_update_event())
        elif event_type == "session.updated":
            if self.settings.greeting:
                await self._send_greeting()
            self._ready.set()
        elif event_type == "error":
            error = message.get("error")
            self.last_error = str(error or "xAI returned an error")

        await self._dispatch(message)

    def _session_update_event(self) -> dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "instructions": self.settings.instructions,
                "voice": self.settings.voice,
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.settings.sample_rate,
                        }
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.settings.sample_rate,
                        }
                    },
                },
                "turn_detection": {"type": "server_vad"},
            },
        }

    async def _send_greeting(self) -> None:
        # 与已验证的 xAI cookbook 顺序保持一致。
        await self.send_event({"type": "input_audio_buffer.commit"})
        await self.send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "请简短地打个招呼。",
                        }
                    ],
                },
            }
        )
        await self.send_event({"type": "response.create"})

    async def _dispatch(self, message: dict[str, Any]) -> None:
        if self._event_handler is None:
            return
        result = self._event_handler(message)
        if inspect.isawaitable(result):
            await result
