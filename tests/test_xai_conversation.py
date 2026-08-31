import asyncio
import base64
import unittest

from core.xai_conversation import XaiConversationController, resample_16k_to_24k
from core.xai_realtime import XaiRealtimeSettings


class _Client:
    def __init__(self, settings, handler):
        self.handler = handler
        self.cancelled = 0
        self.uploaded = []
        self.closed = False

    async def connect(self):
        return None

    async def close(self):
        self.closed = True

    async def append_audio(self, pcm):
        self.uploaded.append(pcm)

    async def cancel_response(self):
        self.cancelled += 1


class _Native:
    def __init__(self):
        self.next_token = 1
        self.stopped = []
        self.recording_started = 0

    def begin_playback_session(self):
        token = self.next_token
        self.next_token += 1
        return token

    def stop_playback_session(self, token):
        self.stopped.append(token)

    async def play_pcm_chunk(self, pcm, token):
        return True

    async def start_recording(self):
        self.recording_started += 1


class XaiConversationControllerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.native = _Native()
        self.settings = XaiRealtimeSettings(
            api_url="wss://example.test/realtime",
            api_key="key",
            aec=False,
            greeting=False,
        )
        self.controller = XaiConversationController(
            self.settings,
            client_factory=_Client,
            native_module=self.native,
        )
        self.controller._loop = asyncio.get_event_loop()

    def test_resample_16k_to_24k_has_expected_length(self):
        pcm16 = b"\x00\x00" * 320
        self.assertEqual(480 * 2, len(resample_16k_to_24k(pcm16)))

    async def test_barge_in_drops_old_response_audio(self):
        await self.controller._handle_event(
            {"type": "response.created", "response": {"id": "old"}}
        )
        delta = base64.b64encode(b"\x01\x00" * 160).decode()
        await self.controller._handle_event(
            {
                "type": "response.output_audio.delta",
                "response_id": "old",
                "delta": delta,
            }
        )
        self.assertEqual(1, self.controller._downlink_queue.qsize())

        await self.controller._handle_event(
            {"type": "input_audio_buffer.speech_started"}
        )
        self.assertEqual(0, self.controller._downlink_queue.qsize())
        self.assertEqual([1], self.native.stopped)
        self.assertEqual(1, self.controller.client.cancelled)

        await self.controller._handle_event(
            {
                "type": "response.output_audio.delta",
                "response_id": "old",
                "delta": delta,
            }
        )
        self.assertEqual(0, self.controller._downlink_queue.qsize())

        await self.controller._handle_event(
            {"type": "response.created", "response": {"id": "new"}}
        )
        await self.controller._handle_event(
            {
                "type": "response.output_audio.delta",
                "response_id": "new",
                "delta": delta,
            }
        )
        self.assertEqual(1, self.controller._downlink_queue.qsize())

    async def test_output_queue_is_bounded_to_300ms(self):
        await self.controller._handle_event(
            {"type": "response.created", "response": {"id": "r1"}}
        )
        delta = base64.b64encode(b"\x01\x00" * 160 * 40).decode()
        await self.controller._handle_event(
            {
                "type": "response.output_audio.delta",
                "response_id": "r1",
                "delta": delta,
            }
        )
        self.assertEqual(30, self.controller._downlink_queue.qsize())

    async def test_audio_delta_alias_is_supported(self):
        await self.controller._handle_event(
            {"type": "response.created", "response": {"id": "r1"}}
        )
        delta = base64.b64encode(b"\x01\x00" * 160).decode()
        await self.controller._handle_event(
            {"type": "response.audio.delta", "response_id": "r1", "delta": delta}
        )
        self.assertEqual(1, self.controller._downlink_queue.qsize())

    async def test_response_done_drains_already_buffered_audio(self):
        await self.controller._handle_event(
            {"type": "response.created", "response": {"id": "r1"}}
        )
        self.controller._render_playback_buffer.extend(b"\x01\x00" * 320)
        await self.controller._handle_event({"type": "response.done"})
        await self.controller._queue_playback_buffer()

        self.assertEqual(1, self.controller._playback_queue.qsize())

    def test_finished_response_arms_idle_timeout_only_once(self):
        self.controller._server_response_done = True
        self.controller._assistant_speaking = True
        self.controller._maybe_finish_response()
        deadline = self.controller._idle_deadline

        self.controller._maybe_finish_response()
        self.assertEqual(deadline, self.controller._idle_deadline)
        self.assertFalse(self.controller._server_response_done)

    async def test_both_transcript_events_can_exit_session(self):
        await self.controller._handle_event(
            {
                "type": "conversation.item.added",
                "item": {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "transcript": "请退出"}
                    ],
                },
            }
        )
        self.assertTrue(self.controller._stop_event.is_set())

        self.controller._stop_event.clear()
        await self.controller._handle_event(
            {
                "type": "conversation.item.input_audio_transcription.updated",
                "transcript": "再见",
            }
        )
        self.assertTrue(self.controller._stop_event.is_set())

    async def test_assistant_transcript_is_accumulated_until_response_done(self):
        await self.controller._handle_event(
            {"type": "response.created", "response": {"id": "r1"}}
        )
        await self.controller._handle_event(
            {"type": "response.output_audio_transcript.delta", "delta": "你"}
        )
        await self.controller._handle_event(
            {"type": "response.output_audio_transcript.delta", "delta": "好"}
        )
        self.assertEqual("你好", self.controller._assistant_transcript)
        await self.controller._handle_event({"type": "response.done"})
        self.assertEqual("", self.controller._assistant_transcript)

    async def test_start_uploads_100ms_chunks_and_restores_recording(self):
        task = asyncio.create_task(self.controller.start())
        for _ in range(50):
            if self.controller._stream is not None:
                break
            await asyncio.sleep(0.01)
        self.assertIsNotNone(self.controller._stream)

        self.controller._stream.input(b"\x01\x00" * 160 * 10)
        for _ in range(50):
            if self.controller.client.uploaded:
                break
            await asyncio.sleep(0.01)

        self.controller.stop()
        await task
        self.assertEqual(3200, len(self.controller.client.uploaded[0]))
        self.assertTrue(self.controller.client.closed)
        self.assertEqual(1, self.native.recording_started)


if __name__ == "__main__":
    unittest.main()
