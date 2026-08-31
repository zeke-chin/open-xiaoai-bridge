import asyncio
import base64
import json
import os
import unittest
from unittest import mock

from core.xai_realtime import XaiRealtimeClient, XaiRealtimeSettings


class _Config:
    def __init__(self, value):
        self.value = value

    def get_app_config(self, path, default=None):
        return self.value if path == "xai" else default


class _FakeWebSocket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.incoming.get()
        if value is StopAsyncIteration:
            raise StopAsyncIteration
        return value

    async def send(self, value):
        self.sent.append(json.loads(value))

    async def close(self):
        self.closed = True
        await self.incoming.put(StopAsyncIteration)


class XaiRealtimeSettingsTest(unittest.TestCase):
    def test_environment_api_key_takes_precedence(self):
        config = _Config(
            {
                "api_url": "wss://example.test/realtime",
                "api_key": "config-key",
                "aec_delay_ms": 200,
            }
        )
        with mock.patch.dict(os.environ, {"XAI_API_KEY": "env-key"}):
            settings = XaiRealtimeSettings.from_config(config)
        self.assertEqual("env-key", settings.api_key)
        self.assertEqual(200, settings.aec_delay_ms)

    def test_rejects_non_16k_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "16000"):
            XaiRealtimeSettings(
                api_url="wss://example.test/realtime",
                api_key="key",
                sample_rate=24000,
            ).validate()

    def test_tools_require_a_tool_executor(self):
        with self.assertRaisesRegex(ValueError, "Custom Function Tools"):
            XaiRealtimeSettings(
                api_url="wss://example.test/realtime",
                api_key="key",
                session={"tools": []},
            ).validate()


class XaiRealtimeClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = _FakeWebSocket()
        self.events = []
        self.settings = XaiRealtimeSettings(
            api_url="wss://example.test/realtime",
            api_key="secret",
            instructions="中文回答",
            greeting=True,
        )

    async def test_handshake_waits_for_session_updated(self):
        async def fake_connect(*args, **kwargs):
            self.assertEqual("Bearer secret", kwargs["additional_headers"]["Authorization"])
            return self.ws

        client = XaiRealtimeClient(self.settings, self.events.append)
        with mock.patch("core.xai_realtime.ws_connect", side_effect=fake_connect):
            connect_task = asyncio.create_task(client.connect(timeout=1))
            await self.ws.incoming.put(json.dumps({"type": "conversation.created"}))
            for _ in range(10):
                if self.ws.sent:
                    break
                await asyncio.sleep(0)
            self.assertFalse(connect_task.done())
            self.assertEqual("session.update", self.ws.sent[0]["type"])
            self.assertEqual(
                16000,
                self.ws.sent[0]["session"]["audio"]["input"]["format"]["rate"],
            )
            self.assertEqual(
                "grok-transcribe",
                self.ws.sent[0]["session"]["audio"]["input"]["transcription"]["model"],
            )
            await self.ws.incoming.put(json.dumps({"type": "session.updated"}))
            await connect_task

        self.assertTrue(client.is_ready)
        self.assertEqual(
            [
                "session.update",
                "input_audio_buffer.commit",
                "conversation.item.create",
                "response.create",
            ],
            [event["type"] for event in self.ws.sent],
        )
        await client.close()

    async def test_audio_is_rejected_before_ready_and_encoded_after_ready(self):
        client = XaiRealtimeClient(self.settings)
        with self.assertRaisesRegex(RuntimeError, "尚未就绪"):
            await client.append_audio(b"\x01\x02")

        async def fake_connect(*args, **kwargs):
            return self.ws

        with mock.patch("core.xai_realtime.ws_connect", side_effect=fake_connect):
            connect_task = asyncio.create_task(client.connect(timeout=1))
            await self.ws.incoming.put(json.dumps({"type": "conversation.created"}))
            await self.ws.incoming.put(json.dumps({"type": "session.updated"}))
            await connect_task
            await client.append_audio(b"\x01\x02")

        self.assertEqual(
            base64.b64encode(b"\x01\x02").decode("ascii"),
            self.ws.sent[-1]["audio"],
        )
        await client.close()

    async def test_connect_fails_when_socket_closes_before_ready(self):
        async def fake_connect(*args, **kwargs):
            return self.ws

        client = XaiRealtimeClient(self.settings)
        with mock.patch("core.xai_realtime.ws_connect", side_effect=fake_connect):
            connect_task = asyncio.create_task(client.connect(timeout=1))
            await self.ws.incoming.put(StopAsyncIteration)
            with self.assertRaisesRegex(RuntimeError, "就绪前关闭"):
                await connect_task

    async def test_advanced_session_options_preserve_audio_invariants(self):
        settings = XaiRealtimeSettings(
            api_url="wss://example.test/realtime",
            api_key="secret",
            session={
                "reasoning": {"effort": "none"},
                "turn_detection": {
                    "threshold": 0.5,
                    "idle_timeout_ms": 1000,
                },
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 48000},
                        "transcription": {"keyterms": ["Grok"]},
                    },
                    "output": {"speed": 1.2},
                },
            },
        )
        client = XaiRealtimeClient(settings)
        session = client._session_update_event()["session"]

        self.assertEqual("none", session["reasoning"]["effort"])
        self.assertEqual(0.5, session["turn_detection"]["threshold"])
        self.assertNotIn("idle_timeout_ms", session["turn_detection"])
        self.assertEqual(16000, session["audio"]["input"]["format"]["rate"])
        self.assertEqual("json", session["audio"]["output"]["transport"])
        self.assertEqual("grok-transcribe", session["audio"]["input"]["transcription"]["model"])
        self.assertEqual(["Grok"], session["audio"]["input"]["transcription"]["keyterms"])
        self.assertEqual(1.2, session["audio"]["output"]["speed"])


if __name__ == "__main__":
    unittest.main()
