import math
import struct
import unittest

import open_xiaoai_server


class XaiNativeAecTest(unittest.TestCase):
    def test_aec_validates_frame_size_and_delay(self):
        aec = open_xiaoai_server.AecProcessor(16000, 1, 150)
        with self.assertRaisesRegex(ValueError, "320 bytes"):
            aec.feed_render(b"\x00" * 10)
        with self.assertRaises(ValueError):
            aec.set_delay_ms(501)
        self.assertEqual(150, aec.stats()["stream_delay_ms"])

    def test_aec_suppresses_synthetic_echo(self):
        aec = open_xiaoai_server.AecProcessor(16000, 1, 0)
        input_energy = 0
        output_energy = 0
        measured_samples = 0
        phase = 0

        for frame_index in range(300):
            samples = [
                int(10000 * math.sin(2 * math.pi * 440 * (phase + index) / 16000))
                for index in range(160)
            ]
            phase += 160
            pcm = struct.pack("<160h", *samples)
            aec.feed_render(pcm)
            output = struct.unpack("<160h", aec.process_capture(pcm))
            if frame_index >= 250:
                input_energy += sum(sample * sample for sample in samples)
                output_energy += sum(sample * sample for sample in output)
                measured_samples += len(samples)

        self.assertGreater(measured_samples, 0)
        self.assertLess(output_energy / input_energy, 0.1)

    def test_playback_tokens_are_generation_scoped(self):
        first = open_xiaoai_server.begin_playback_session()
        second = open_xiaoai_server.begin_playback_session()
        self.assertFalse(open_xiaoai_server.playback_session_active(first))
        self.assertTrue(open_xiaoai_server.playback_session_active(second))


if __name__ == "__main__":
    unittest.main()
