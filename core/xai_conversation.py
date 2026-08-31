"""xAI Realtime Voice 全双工会话控制器。"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from typing import Any, Callable

import numpy as np
from scipy.signal import resample_poly

import open_xiaoai_server

from core.ref import get_app, get_speaker
from core.services.audio.stream import MyStream
from core.services.protocols.typing import AudioConfig, DeviceState
from core.utils.config import ConfigManager
from core.utils.logger import logger
from core.xai_realtime import XaiRealtimeClient, XaiRealtimeSettings


_CAPTURE_RATE = 16000
_PLAYBACK_RATE = 24000
_FRAME_MS = 10
_FRAME_SAMPLES = _CAPTURE_RATE * _FRAME_MS // 1000
_FRAME_BYTES = _FRAME_SAMPLES * 2
_CAPTURE_BACKLOG_BYTES = _CAPTURE_RATE * 2 * 300 // 1000
_OUTPUT_QUEUE_FRAMES = 300 // _FRAME_MS
_UPLINK_CHUNK_FRAMES = 100 // _FRAME_MS
_HALF_DUPLEX_ECHO_GUARD_SECONDS = 0.25


def resample_16k_to_24k(pcm: bytes) -> bytes:
    """将 PCM16 mono 从 16kHz 高质量重采样到 24kHz。"""
    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype="<i2")
    converted = resample_poly(samples.astype(np.float32), 3, 2)
    return np.clip(np.rint(converted), -32768, 32767).astype("<i2").tobytes()


class XaiConversationController:
    """管理一条 xAI Realtime 会话及其设备音频生命周期。"""

    def __init__(
        self,
        settings: XaiRealtimeSettings | None = None,
        *,
        client_factory: Callable[..., XaiRealtimeClient] = XaiRealtimeClient,
        native_module: Any = open_xiaoai_server,
        stream_factory: Callable[..., MyStream] = MyStream,
    ) -> None:
        self.settings = settings or XaiRealtimeSettings.from_config()
        self._native = native_module
        self._stream_factory = stream_factory
        self.client = client_factory(self.settings, self._handle_event)

        self.active = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = asyncio.Event()
        self._stream: MyStream | None = None
        self._tasks: list[asyncio.Task] = []

        self._downlink_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=_OUTPUT_QUEUE_FRAMES
        )
        self._playback_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(
            maxsize=15
        )
        self._uplink_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=3)
        self._downlink_remainder = bytearray()
        self._capture_accumulator = bytearray()
        self._render_playback_buffer = bytearray()

        self._aec = None
        self._aec_enabled = False
        self._aec_degraded = False
        self._assistant_speaking = False
        self._user_speaking = False
        self._server_response_done = False
        self._allow_output = False
        self._response_id: str | None = None
        self._playback_token: int | None = None
        self._playback_inflight = False
        self._idle_deadline: float | None = None
        self._resume_uplink_at = 0.0
        self._after_wakeup_called = False

    def is_active(self) -> bool:
        return self.active

    async def start(self) -> None:
        if self.active:
            return
        self.active = True
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._set_device_state(DeviceState.CONNECTING)
        logger.info("进入 Grok Voice 实时对话", module="xAI Conv")

        try:
            self._initialize_aec()
            await self.client.connect()
            if self._stop_event.is_set():
                return
            self._stream = self._stream_factory(
                rate=_CAPTURE_RATE,
                channels=1,
                format=AudioConfig.FORMAT,
                input=True,
                frames_per_buffer=_FRAME_SAMPLES,
                max_buffer_bytes=_CAPTURE_BACKLOG_BYTES,
                start=True,
            )
            self._set_device_state(DeviceState.LISTENING)
            if not self.settings.greeting:
                self._arm_idle_timeout()
            self._tasks = [
                asyncio.create_task(self._audio_clock_loop(), name="xai-audio-clock"),
                asyncio.create_task(self._uplink_loop(), name="xai-uplink"),
                asyncio.create_task(self._playback_loop(), name="xai-playback"),
                asyncio.create_task(self._idle_loop(), name="xai-idle-timeout"),
            ]
            await self._stop_event.wait()
        finally:
            await self._cleanup()

    def stop(self) -> None:
        """线程安全地请求结束会话，并立即定向停止当前 PCM。"""
        if not self.active:
            return
        self.active = False
        token = self._playback_token
        self._playback_token = None
        if token is not None:
            self._native.stop_playback_session(token)
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop_event.set)
        else:
            self._stop_event.set()

    async def _cleanup(self) -> None:
        self.active = False
        self._stop_event.set()
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current and not task.done():
                task.cancel()
        for task in self._tasks:
            if task is current:
                continue
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._clear_queue(self._downlink_queue)
        self._clear_queue(self._playback_queue)
        self._clear_queue(self._uplink_queue)
        self._downlink_remainder.clear()
        self._capture_accumulator.clear()
        self._render_playback_buffer.clear()

        token = self._playback_token
        self._playback_token = None
        if token is not None:
            self._native.stop_playback_session(token)
        await self.client.close()
        with contextlib.suppress(Exception):
            await self._native.start_recording()
        self._set_device_state(DeviceState.IDLE)
        await self._call_after_wakeup()
        logger.info("退出 Grok Voice 实时对话", module="xAI Conv")

    def _initialize_aec(self) -> None:
        if not self.settings.aec:
            return
        try:
            self._aec = self._native.AecProcessor(
                _CAPTURE_RATE,
                1,
                self.settings.aec_delay_ms,
            )
            self._aec_enabled = True
            logger.info(
                f"Sonora AEC3 已启用，delay={self.settings.aec_delay_ms}ms",
                module="xAI Conv",
            )
        except Exception as exc:
            self._degrade_aec(exc)

    def _degrade_aec(self, exc: Exception) -> None:
        self._aec = None
        self._aec_enabled = False
        if not self._aec_degraded:
            self._aec_degraded = True
            logger.warning(
                f"AEC 不可用，当前会话降级为半双工: {type(exc).__name__}: {exc}",
                module="xAI Conv",
            )

    async def _audio_clock_loop(self) -> None:
        assert self._loop is not None
        next_tick = self._loop.time()
        silence = b"\x00" * _FRAME_BYTES
        while not self._stop_event.is_set():
            next_tick += _FRAME_MS / 1000
            delay = next_tick - self._loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.1:
                next_tick = self._loop.time()

            render = self._queue_get_nowait(self._downlink_queue)
            render_for_aec = render or silence
            if self._aec_enabled:
                try:
                    self._aec.feed_render(render_for_aec)
                except Exception as exc:
                    self._degrade_aec(exc)

            if render is not None:
                self._render_playback_buffer.extend(render)
                if len(self._render_playback_buffer) >= _FRAME_BYTES * 2:
                    await self._queue_playback_buffer()
            elif (
                self._server_response_done
                and len(self._render_playback_buffer) == _FRAME_BYTES
            ):
                self._render_playback_buffer.extend(silence)
                await self._queue_playback_buffer()

            capture = self._stream.read(_FRAME_SAMPLES) if self._stream else b""
            if capture:
                clean = capture
                if self._aec_enabled:
                    try:
                        clean = bytes(self._aec.process_capture(capture))
                    except Exception as exc:
                        self._degrade_aec(exc)
                        clean = capture

                if self._should_upload_capture():
                    self._capture_accumulator.extend(clean)
                    required = _FRAME_BYTES * _UPLINK_CHUNK_FRAMES
                    if len(self._capture_accumulator) >= required:
                        chunk = bytes(self._capture_accumulator[:required])
                        del self._capture_accumulator[:required]
                        self._put_drop_oldest(self._uplink_queue, chunk)
                else:
                    self._capture_accumulator.clear()

            self._maybe_finish_response()

    async def _queue_playback_buffer(self) -> None:
        pcm16 = bytes(self._render_playback_buffer[:_FRAME_BYTES * 2])
        del self._render_playback_buffer[:_FRAME_BYTES * 2]
        token = self._playback_token
        if token is None:
            return
        pcm24 = resample_16k_to_24k(pcm16)
        self._put_drop_oldest(self._playback_queue, (token, pcm24))

    async def _uplink_loop(self) -> None:
        while not self._stop_event.is_set():
            chunk = await self._uplink_queue.get()
            try:
                await self.client.append_audio(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"上行音频失败: {exc}", module="xAI Conv")
                self._stop_event.set()
            finally:
                self._uplink_queue.task_done()

    async def _playback_loop(self) -> None:
        while not self._stop_event.is_set():
            token, pcm = await self._playback_queue.get()
            self._playback_inflight = True
            try:
                await self._native.play_pcm_chunk(pcm, token)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"播放 PCM 失败: {exc}", module="xAI Conv")
                await self._interrupt_response(cancel_remote=False)
            finally:
                self._playback_inflight = False
                self._playback_queue.task_done()
            self._maybe_finish_response()

    async def _idle_loop(self) -> None:
        assert self._loop is not None
        while not self._stop_event.is_set():
            await asyncio.sleep(0.2)
            if self._idle_deadline and self._loop.time() >= self._idle_deadline:
                logger.info("等待用户说话超时", module="xAI Conv")
                self._stop_event.set()

    async def _handle_event(self, message: dict[str, Any]) -> None:
        event_type = str(message.get("type", ""))
        if event_type == "session.updated":
            self._set_device_state(DeviceState.LISTENING)
            if not self.settings.greeting:
                self._arm_idle_timeout()
            return
        if event_type == "client.closed":
            self._stop_event.set()
            return
        if event_type == "response.created":
            self._start_response(message)
            return
        if event_type == "response.output_audio.delta":
            self._handle_audio_delta(message)
            return
        if event_type == "input_audio_buffer.speech_started":
            self._user_speaking = True
            self._idle_deadline = None
            await self._interrupt_response(cancel_remote=True)
            self._set_device_state(DeviceState.LISTENING)
            return
        if event_type == "input_audio_buffer.speech_stopped":
            self._user_speaking = False
            return
        if event_type in {"response.done", "response.cancelled"}:
            self._server_response_done = True
            self._allow_output = False
            self._flush_downlink_remainder()
            self._maybe_finish_response()
            return

        transcript = self._extract_user_transcript(message)
        if transcript:
            logger.user_speech(transcript, module="xAI")
            if any(keyword in transcript for keyword in self.settings.exit_keywords):
                logger.info(f"检测到退出关键词: {transcript}", module="xAI Conv")
                self._stop_event.set()

    def _start_response(self, message: dict[str, Any]) -> None:
        previous_token = self._playback_token
        if previous_token is not None:
            self._native.stop_playback_session(previous_token)
        self._clear_queue(self._downlink_queue)
        self._clear_queue(self._playback_queue)
        self._downlink_remainder.clear()
        self._render_playback_buffer.clear()
        response = message.get("response")
        self._response_id = (
            str(response.get("id"))
            if isinstance(response, dict) and response.get("id")
            else str(message.get("response_id") or "") or None
        )
        self._playback_token = self._native.begin_playback_session()
        self._allow_output = True
        self._server_response_done = False
        self._assistant_speaking = False
        self._idle_deadline = None

    def _handle_audio_delta(self, message: dict[str, Any]) -> None:
        if not self._allow_output or self._playback_token is None:
            return
        message_response_id = message.get("response_id")
        if (
            message_response_id
            and self._response_id
            and str(message_response_id) != self._response_id
        ):
            return
        try:
            pcm = base64.b64decode(message.get("delta", ""), validate=True)
        except Exception as exc:
            logger.warning(f"忽略无效音频 delta: {exc}", module="xAI Conv")
            return
        if not pcm:
            return
        self._assistant_speaking = True
        self._set_device_state(DeviceState.SPEAKING)
        self._downlink_remainder.extend(pcm)
        while len(self._downlink_remainder) >= _FRAME_BYTES:
            frame = bytes(self._downlink_remainder[:_FRAME_BYTES])
            del self._downlink_remainder[:_FRAME_BYTES]
            self._put_drop_oldest(self._downlink_queue, frame)

    def _flush_downlink_remainder(self) -> None:
        if not self._downlink_remainder:
            return
        padded = bytes(self._downlink_remainder).ljust(_FRAME_BYTES, b"\x00")
        self._downlink_remainder.clear()
        self._put_drop_oldest(self._downlink_queue, padded)

    async def _interrupt_response(self, *, cancel_remote: bool) -> None:
        had_response = self._playback_token is not None or self._assistant_speaking
        self._allow_output = False
        self._server_response_done = True
        self._assistant_speaking = False
        self._downlink_remainder.clear()
        self._render_playback_buffer.clear()
        self._clear_queue(self._downlink_queue)
        self._clear_queue(self._playback_queue)
        token = self._playback_token
        self._playback_token = None
        if token is not None:
            self._native.stop_playback_session(token)
        if cancel_remote and had_response:
            with contextlib.suppress(Exception):
                await self.client.cancel_response()
        if self._loop:
            self._resume_uplink_at = (
                self._loop.time() + _HALF_DUPLEX_ECHO_GUARD_SECONDS
            )

    def _maybe_finish_response(self) -> None:
        if not self._server_response_done:
            return
        if self._user_speaking:
            return
        if (
            not self._downlink_queue.empty()
            or self._downlink_remainder
            or self._render_playback_buffer
            or not self._playback_queue.empty()
            or self._playback_inflight
        ):
            return
        if self._assistant_speaking:
            self._assistant_speaking = False
            if self._loop:
                self._resume_uplink_at = (
                    self._loop.time() + _HALF_DUPLEX_ECHO_GUARD_SECONDS
                )
            if self._stream and not self._aec_enabled:
                self._stream.clear_input()
        self._server_response_done = False
        self._response_id = None
        self._playback_token = None
        self._set_device_state(DeviceState.LISTENING)
        self._arm_idle_timeout(after_echo_guard=not self._aec_enabled)

    def _should_upload_capture(self) -> bool:
        if self._aec_enabled:
            return True
        if self._assistant_speaking or (
            self._playback_token is not None and not self._server_response_done
        ):
            return False
        return not self._loop or self._loop.time() >= self._resume_uplink_at

    def _arm_idle_timeout(self, *, after_echo_guard: bool = False) -> None:
        if not self._loop:
            return
        guard = _HALF_DUPLEX_ECHO_GUARD_SECONDS if after_echo_guard else 0.0
        self._idle_deadline = self._loop.time() + guard + self.settings.idle_timeout

    def _extract_user_transcript(self, message: dict[str, Any]) -> str:
        event_type = message.get("type")
        if event_type == "conversation.item.input_audio_transcription.completed":
            return str(message.get("transcript", "") or "").strip()
        if event_type != "conversation.item.added":
            return ""
        item = message.get("item")
        if not isinstance(item, dict) or item.get("role") != "user":
            return ""
        content = item.get("content")
        if not isinstance(content, list):
            return ""
        transcripts = [
            str(part.get("transcript", "")).strip()
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "input_audio"
            and part.get("transcript")
        ]
        return " ".join(transcripts)

    async def _call_after_wakeup(self) -> None:
        if self._after_wakeup_called:
            return
        self._after_wakeup_called = True
        after_wakeup = ConfigManager.instance().get_app_config(
            "wakeup.after_wakeup"
        )
        speaker = get_speaker()
        if after_wakeup and speaker:
            try:
                await after_wakeup(speaker, source="xai")
            except Exception as exc:
                logger.warning(f"after_wakeup 执行失败: {exc}", module="xAI Conv")

    def _set_device_state(self, state: str) -> None:
        app = get_app()
        if not app:
            return
        setter = getattr(app, "set_device_state", None)
        if setter:
            setter(state)
        else:
            app.device_state = state

    @staticmethod
    def _put_drop_oldest(queue: asyncio.Queue, item: Any) -> None:
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
                queue.task_done()
        queue.put_nowait(item)

    @staticmethod
    def _queue_get_nowait(queue: asyncio.Queue):
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        queue.task_done()
        return item

    @staticmethod
    def _clear_queue(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                return
