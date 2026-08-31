"""xAI Realtime 实时连通性检查。

用法：
    XAI_LIVE_TEST=1 python3 -m unittest tests/test_xai_live_connectivity.py -v

可选覆盖：
    XAI_API_KEY=...
    XAI_LIVE_API_URL=ws://127.0.0.1:3300/v1/realtime?model=grok-voice-latest
"""

import os
import unittest
from dataclasses import replace

from core.xai_realtime import XaiRealtimeClient, XaiRealtimeSettings


def _enabled() -> bool:
    return os.getenv("XAI_LIVE_TEST", "").strip().lower() in {"1", "true", "yes"}


@unittest.skipUnless(
    _enabled(),
    "set XAI_LIVE_TEST=1 to run live xAI Realtime connectivity tests",
)
class XaiLiveConnectivityTest(unittest.IsolatedAsyncioTestCase):
    async def test_connect_and_configure_session(self):
        settings = XaiRealtimeSettings.from_config()
        api_url = os.getenv("XAI_LIVE_API_URL", "").strip()
        if api_url:
            settings = replace(settings, api_url=api_url)

        # 连通性测试不触发开场白，避免生成和播放无意义的实时音频。
        settings = replace(settings, greeting=False)
        event_types: list[str] = []
        client = XaiRealtimeClient(
            settings,
            lambda event: event_types.append(str(event.get("type", ""))),
        )
        try:
            await client.connect(timeout=15)
            self.assertTrue(client.is_ready)
            self.assertTrue(client.state.conversation_id)
            self.assertIn("conversation.created", event_types)
            self.assertIn("session.updated", event_types)
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
