"""xAI Realtime Voice WebSocket 协议客户端。"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

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
    session: dict[str, Any] = field(default_factory=dict)

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
            session=deepcopy(raw.get("session", {})),
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
        if not isinstance(self.session, dict):
            raise ValueError("xai.session 必须是字典")
        if "tools" in self.session:
            raise ValueError(
                "xai.session.tools 暂未开放；需与 Custom Function Tools 执行器一起配置"
            )

    @property
    def resumption_enabled(self) -> bool:
        resumption = self.session.get("resumption", {})
        return isinstance(resumption, dict) and resumption.get("enabled") is True


@dataclass(slots=True)
class XaiConversationState:
    """可跨 WebSocket 连接保存的服务端会话标识。

    这里只描述断线续接状态，不承担长上下文摘要或压缩策略。
    """

    conversation_id: str | None = None
    last_activity_at: float | None = None

    def record_conversation(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.touch()

    def touch(self) -> None:
        self.last_activity_at = time.time()

    def clear(self) -> None:
        self.conversation_id = None
        self.last_activity_at = None

    def is_expired(self, ttl_seconds: float = 30 * 60) -> bool:
        if not self.conversation_id or self.last_activity_at is None:
            return True
        return time.time() - self.last_activity_at >= ttl_seconds

    def connection_url(self, api_url: str, *, enabled: bool) -> str:
        """在启用且未过期时，把 conversation_id 安全写入查询参数。"""
        if not enabled or self.is_expired():
            return api_url
        parsed = urlparse(api_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["conversation_id"] = self.conversation_id or ""
        return urlunparse(parsed._replace(query=urlencode(query)))


class XaiRealtimeClient:
    """负责 xAI Realtime 协议收发，不承载设备音频策略。"""

    def __init__(
        self,
        settings: XaiRealtimeSettings,
        event_handler: EventHandler | None = None,
        state: XaiConversationState | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self._event_handler = event_handler
        self.state = state or XaiConversationState()
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

        connection_url = self.state.connection_url(
            self.settings.api_url,
            enabled=self.settings.resumption_enabled,
        )
        self._ws = await ws_connect(
            connection_url,
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
        self.state.touch()
        if event_type == "conversation.created" and not self._configured:
            conversation = message.get("conversation")
            if isinstance(conversation, dict) and conversation.get("id"):
                self.state.record_conversation(str(conversation["id"]))
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
        session = {
            "instructions": self.settings.instructions,
            "voice": self.settings.voice,
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.settings.sample_rate,
                    },
                    "transport": "json",
                    "transcription": {
                        "model": "grok-transcribe",
                        "language_hint": "zh",
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.settings.sample_rate,
                    },
                    "transport": "json",
                },
            },
            "turn_detection": {"type": "server_vad"},
        }
        _deep_merge(session, deepcopy(self.settings.session))

        # 以下字段与音箱音频管线绑定，不能被高级参数覆盖。
        session["instructions"] = self.settings.instructions
        session["voice"] = self.settings.voice
        session.setdefault("audio", {}).setdefault("input", {})["format"] = {
            "type": "audio/pcm",
            "rate": self.settings.sample_rate,
        }
        session["audio"]["input"]["transport"] = "json"
        transcription = session["audio"]["input"].setdefault("transcription", {})
        transcription["model"] = "grok-transcribe"
        session["audio"].setdefault("output", {})["format"] = {
            "type": "audio/pcm",
            "rate": self.settings.sample_rate,
        }
        session["audio"]["output"]["transport"] = "json"
        session.setdefault("turn_detection", {})["type"] = "server_vad"
        # 本地 controller 负责超时退出，避免服务端主动重新搭话与退出竞态。
        session["turn_detection"].pop("idle_timeout_ms", None)
        session.pop("tools", None)
        return {"type": "session.update", "session": session}

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


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> None:
    """递归合并高级 session 参数，列表和标量直接覆盖。"""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
