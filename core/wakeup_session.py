import asyncio

from core.ref import (
    get_app,
    get_kws,
    get_speaker,
    get_xiaozhi,
)
from core.services.protocols.typing import AbortReason
from core.utils.config import ConfigManager
from core.utils.logger import logger


class WakeupSessionManager:
    """Dispatches wakeup events to XiaoZhi or external backend controllers."""

    def __init__(self):
        self.config = ConfigManager.instance()
        self._openclaw_controller = None
        self._openclaw_task: asyncio.Task | None = None
        self._openai_controller = None
        self._openai_task: asyncio.Task | None = None
        self._qwenpaw_controller = None
        self._qwenpaw_task: asyncio.Task | None = None
        self._xai_controller = None
        self._xai_task: asyncio.Task | None = None
        self._xiaozhi_future: asyncio.Future | None = None

    def _get_loop(self):
        app = get_app()
        if app:
            return app.loop
        from core.xiaoai import XiaoAI
        return XiaoAI.async_loop

    async def _stop_device_playback(self):
        """Stop all audio playback on the device and restart recording.

        - killall tts_play.sh miplayer: stop blocking TTS (tts_play.sh + child miplayer)
        - mphelper pause: stop non-blocking TTS (mibrain text_to_speech via mediaplayer)
        - stop_playing: kill aplay (our PCM channel)
        - start_playing / start_recording: restart audio streams
        """
        speaker = get_speaker()
        if speaker:
            await speaker.stop_device_audio()
            import open_xiaoai_server
            await open_xiaoai_server.start_recording()
            return

        import open_xiaoai_server
        await open_xiaoai_server.stop_playing()
        await open_xiaoai_server.start_recording()

    def on_interrupt(self):
        logger.info("[Wakeup] XiaoAI wakeup — interrupting active sessions")

        loop = self._get_loop()

        # Stop XiaoZhi wakeup session (cancels futures + aborts server audio)
        if self._xiaozhi_future and not self._xiaozhi_future.done():
            self._xiaozhi_future.cancel()
        self._xiaozhi_future = None
        xiaozhi = get_xiaozhi()
        if xiaozhi:
            xiaozhi.stop_wakeup_session()

        # Stop external backend conversations (cancels VAD + stops TTS stream + kills aplay)
        if self._openclaw_controller and self._openclaw_controller.is_active():
            self._openclaw_controller.stop()
        if self._openclaw_task and not self._openclaw_task.done():
            loop.call_soon_threadsafe(self._openclaw_task.cancel)
        if self._openai_controller and self._openai_controller.is_active():
            self._openai_controller.stop()
        if self._openai_task and not self._openai_task.done():
            loop.call_soon_threadsafe(self._openai_task.cancel)
        if self._qwenpaw_controller and self._qwenpaw_controller.is_active():
            self._qwenpaw_controller.stop()
        if self._qwenpaw_task and not self._qwenpaw_task.done():
            loop.call_soon_threadsafe(self._qwenpaw_task.cancel)
        if self._xai_controller and self._xai_controller.is_active():
            self._xai_controller.stop()
        if self._xai_task and not self._xai_task.done():
            loop.call_soon_threadsafe(self._xai_task.cancel)

        asyncio.run_coroutine_threadsafe(self._stop_device_playback(), loop)

        from core.xiaoai import XiaoAI
        XiaoAI.stop_conversation()

    def on_wakeup(self):
        logger.info("[Wakeup] Wakeup session started")
        xiaozhi = get_xiaozhi()
        if xiaozhi:
            xiaozhi._is_first_round = True
            future = asyncio.run_coroutine_threadsafe(
                xiaozhi.start_wakeup_session(), self._get_loop()
            )
            self._xiaozhi_future = future

            def _clear_future(done_future):
                if self._xiaozhi_future is done_future:
                    self._xiaozhi_future = None

            future.add_done_callback(_clear_future)

    def on_speech(self, speech_buffer: bytes):
        """Called by VAD when speech is detected."""
        pass

    def on_silence(self):
        """Called by VAD when silence is detected."""
        pass

    def consume_xiaoai_asr_result(
        self,
        dialog_id: str,
        text: str,
        is_final,
        is_vad_begin,
    ) -> bool:
        """Route XiaoAI native ASR results to the active external backend controller."""
        for controller in (
            self._openclaw_controller,
            self._openai_controller,
            self._qwenpaw_controller,
        ):
            if controller and controller.is_active():
                return controller.consume_xiaoai_recognize_result(
                    dialog_id=dialog_id,
                    text=text,
                    is_final=is_final,
                    is_vad_begin=is_vad_begin,
                )
        return False

    async def wakeup(self, text, source):
        before_wakeup = self.config.get_app_config("wakeup.before_wakeup")
        kws = get_kws()
        logger.debug(f"[Wakeup] Received wakeup request from {source}: {text}")

        # Reset session_key to config default before each wakeup,
        # so paths that don't call set_openclaw_session_key() always use the default.
        from core.openclaw import OpenClawManager
        default_session_key = self.config.get_app_config("openclaw", {}).get(
            "session_key", "agent:main:open-xiaoai-bridge"
        )
        OpenClawManager._session_key = default_session_key
        from core.openai import OpenAIManager
        default_openai_session_key = self.config.get_app_config(
            "openai", {}
        ).get("session_key", "agent:default:open-xiaoai-bridge")
        OpenAIManager._session_key = default_openai_session_key
        from core.qwenpaw import QwenPawManager
        default_qwenpaw_session_key = self.config.get_app_config(
            "qwenpaw", {}
        ).get("session_key", "agent:default:open-xiaoai-bridge")
        QwenPawManager._session_key = default_qwenpaw_session_key

        if kws:
            kws.pause()
        should_wakeup = await before_wakeup(
            get_speaker(),
            text,
            source,
            get_app(),
        )
        if kws:
            kws.resume()
        logger.info(f"[Wakeup] before_wakeup returned: {should_wakeup}")
        if should_wakeup is not None:
            await self.reset_all_sessions()

        if should_wakeup == "openclaw":
            await self._start_openclaw_conversation()
        elif should_wakeup == "openai":
            await self._start_openai_conversation()
        elif should_wakeup == "qwenpaw":
            await self._start_qwenpaw_conversation()
        elif should_wakeup == "xai":
            app = get_app()
            if not app or not getattr(app, "_enable_xai", False):
                logger.warning(
                    "before_wakeup 返回 xai，但 XAI_ENABLE 未启用",
                    module="Wakeup",
                )
                await self._call_after_wakeup("xai")
            else:
                await self._start_xai_conversation()
        elif should_wakeup == "xiaozhi":
            self.on_wakeup()

    async def _start_openclaw_conversation(self):
        """Start an OpenClaw continuous conversation session.

        This runs independently of the XiaoZhi session state machine.
        KWS is paused during the conversation and resumed when done.
        """
        from core.openclaw_conversation import OpenClawConversationController

        kws = get_kws()
        if kws:
            kws.pause()
        try:
            self._openclaw_controller = OpenClawConversationController()
            self._openclaw_task = asyncio.create_task(self._openclaw_controller.start())
            await self._openclaw_task
        except asyncio.CancelledError:
            pass  # interrupted cleanly by on_interrupt
        except Exception as exc:
            logger.error(
                f"[Wakeup] OpenClaw conversation failed: {type(exc).__name__}: {exc}",
                module="Wakeup",
            )
        finally:
            self._openclaw_controller = None
            self._openclaw_task = None
            if kws:
                kws.resume()

    async def _start_openai_conversation(self):
        """Start an OpenAI-compatible continuous conversation session."""
        from core.openai_conversation import OpenAIConversationController

        kws = get_kws()
        if kws:
            kws.pause()
        try:
            self._openai_controller = OpenAIConversationController()
            self._openai_task = asyncio.create_task(self._openai_controller.start())
            await self._openai_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                f"[Wakeup] OpenAI conversation failed: {type(exc).__name__}: {exc}",
                module="Wakeup",
            )
        finally:
            self._openai_controller = None
            self._openai_task = None
            if kws:
                kws.resume()

    async def _start_qwenpaw_conversation(self):
        """Start a QwenPaw continuous conversation session."""
        from core.qwenpaw_conversation import QwenPawConversationController

        kws = get_kws()
        if kws:
            kws.pause()
        try:
            self._qwenpaw_controller = QwenPawConversationController()
            self._qwenpaw_task = asyncio.create_task(self._qwenpaw_controller.start())
            await self._qwenpaw_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                f"[Wakeup] QwenPaw conversation failed: {type(exc).__name__}: {exc}",
                module="Wakeup",
            )
        finally:
            self._qwenpaw_controller = None
            self._qwenpaw_task = None
            if kws:
                kws.resume()

    async def _start_xai_conversation(self):
        """启动独立的 xAI Realtime Voice 全双工会话。"""
        from core.xai_conversation import XaiConversationController

        kws = get_kws()
        if kws:
            kws.pause()
        try:
            self._xai_controller = XaiConversationController()
            self._xai_task = asyncio.create_task(self._xai_controller.start())
            await self._xai_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                f"xAI conversation failed: {type(exc).__name__}: {exc}",
                module="Wakeup",
            )
        finally:
            self._xai_controller = None
            self._xai_task = None
            if kws:
                kws.resume()

    async def _call_after_wakeup(self, source: str):
        after_wakeup = self.config.get_app_config("wakeup.after_wakeup")
        speaker = get_speaker()
        if after_wakeup and speaker:
            try:
                await after_wakeup(speaker, source=source)
            except Exception as exc:
                logger.warning(
                    f"after_wakeup failed for {source}: {exc}",
                    module="Wakeup",
                )

    async def stop_xai_conversation(self):
        """停止并等待 xAI 会话清理完成，供应用 shutdown 使用。"""
        if self._xai_controller and self._xai_controller.is_active():
            self._xai_controller.stop()
        task = self._xai_task
        if task and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2)
            except TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def reset_all_sessions(self):
        """Reset all active sessions before starting a new one.

        Stops XiaoAI continuous conversation, interrupts any active XiaoZhi
        session, and stops any external backend continuous conversation.
        """
        from core.xiaoai import XiaoAI
        from core.ref import get_xiaozhi

        # Stop XiaoAI continuous conversation
        XiaoAI.stop_conversation()

        # Interrupt active XiaoZhi session
        xiaozhi = get_xiaozhi()
        if xiaozhi and xiaozhi.is_connected():
            try:
                await xiaozhi.send_abort_speaking(AbortReason.ABORT)
            except Exception:
                pass

        # Stop OpenClaw continuous conversation (also stops its TTS stream)
        if self._openclaw_controller and self._openclaw_controller.is_active():
            self._openclaw_controller.stop()
        if self._openai_controller and self._openai_controller.is_active():
            self._openai_controller.stop()
        if self._qwenpaw_controller and self._qwenpaw_controller.is_active():
            self._qwenpaw_controller.stop()
        if self._xai_controller and self._xai_controller.is_active():
            self._xai_controller.stop()

        # Stop all audio playback on the device
        await self._stop_device_playback()

        logger.debug("[Wakeup] All sessions reset")


EventManager = WakeupSessionManager()
