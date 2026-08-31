import asyncio
import os
import unittest
from unittest import mock

import config
from core.app import MainApp
from core.ref import set_app
from core.wakeup_session import WakeupSessionManager


class _Speaker:
    def __init__(self):
        self.played = []
        self.aborted = 0

    async def play(self, **kwargs):
        self.played.append(kwargs)

    async def abort_xiaoai(self):
        self.aborted += 1


class XaiConfigRouteTest(unittest.IsolatedAsyncioTestCase):
    async def test_default_config_routes_grok_wake_words(self):
        speaker = _Speaker()
        result = await config.before_wakeup(speaker, "你好 grok", "kws", None)
        self.assertEqual("xai", result)
        self.assertEqual("Grok 来了", speaker.played[0]["text"])

        result = await config.before_wakeup(speaker, "召唤 grok", "xiaoai", None)
        self.assertEqual("xai", result)
        self.assertEqual(1, speaker.aborted)


class XaiWakeupDispatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_enabled_app_dispatches_to_xai_controller(self):
        manager = WakeupSessionManager()

        async def before_wakeup(*_args):
            return "xai"

        class Config:
            def get_app_config(self, path, default=None):
                if path == "wakeup.before_wakeup":
                    return before_wakeup
                if path in {"openclaw", "openai", "qwenpaw"}:
                    return {}
                return default

        manager.config = Config()
        manager.reset_all_sessions = mock.AsyncMock()
        manager._start_xai_conversation = mock.AsyncMock()
        app = mock.Mock(_enable_xai=True)

        with (
            mock.patch("core.wakeup_session.get_app", return_value=app),
            mock.patch("core.wakeup_session.get_kws", return_value=None),
            mock.patch("core.wakeup_session.get_speaker", return_value=_Speaker()),
        ):
            await manager.wakeup("你好 grok", "kws")

        manager._start_xai_conversation.assert_awaited_once()


class XaiStartupValidationTest(unittest.TestCase):
    def tearDown(self):
        if MainApp._instance is not None:
            MainApp._instance.loop.close()
        MainApp._instance = None
        set_app(None)

    def test_xai_rejects_disabled_audio_input_before_starting_threads(self):
        app = MainApp(enable_xiaozhi=False, enable_xai=True)
        with mock.patch.dict(os.environ, {"AUDIO_INPUT_ENABLE": "false"}):
            with self.assertRaisesRegex(RuntimeError, "xAI Realtime Voice"):
                app.run()

    def test_xai_rejects_missing_api_key_before_starting_threads(self):
        app = MainApp(enable_xiaozhi=False, enable_xai=True)
        with mock.patch.dict(
            os.environ,
            {"AUDIO_INPUT_ENABLE": "true", "XAI_API_KEY": ""},
        ):
            with self.assertRaisesRegex(ValueError, "API key"):
                app.run()


if __name__ == "__main__":
    unittest.main()
