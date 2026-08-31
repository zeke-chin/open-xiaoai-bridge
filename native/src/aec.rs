use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use sonora::config::EchoCanceller;
use sonora::{AudioProcessing, Config, StreamConfig};
use std::sync::Mutex;

const SUPPORTED_SAMPLE_RATE: u32 = 16_000;
const SUPPORTED_CHANNELS: u16 = 1;

struct AecState {
    processor: AudioProcessing,
    delay_ms: i32,
}

impl AecState {
    fn new(delay_ms: i32) -> PyResult<Self> {
        if !(0..=500).contains(&delay_ms) {
            return Err(PyValueError::new_err("delay_ms must be in 0..=500"));
        }
        let stream = StreamConfig::new(SUPPORTED_SAMPLE_RATE, SUPPORTED_CHANNELS);
        let config = Config {
            echo_canceller: Some(EchoCanceller::default()),
            ..Default::default()
        };
        let mut processor = AudioProcessing::builder()
            .config(config)
            .capture_config(stream)
            .render_config(stream)
            .echo_detector(true)
            .build();
        processor
            .set_stream_delay_ms(delay_ms)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            processor,
            delay_ms,
        })
    }
}

/// 每个实时语音会话独占的 Sonora AEC3 处理器。
#[pyclass]
pub struct AecProcessor {
    state: Mutex<AecState>,
    frame_samples: usize,
}

#[pymethods]
impl AecProcessor {
    #[new]
    #[pyo3(signature = (sample_rate=16000, channels=1, delay_ms=150))]
    fn new(sample_rate: u32, channels: u16, delay_ms: i32) -> PyResult<Self> {
        if sample_rate != SUPPORTED_SAMPLE_RATE {
            return Err(PyValueError::new_err("sample_rate must be 16000"));
        }
        if channels != SUPPORTED_CHANNELS {
            return Err(PyValueError::new_err("channels must be 1"));
        }
        Ok(Self {
            state: Mutex::new(AecState::new(delay_ms)?),
            frame_samples: sample_rate as usize / 100 * channels as usize,
        })
    }

    /// 提交即将播放的 10ms PCM16 render 参考帧。
    fn feed_render(&self, pcm: &[u8]) -> PyResult<()> {
        let input = decode_frame(pcm, self.frame_samples)?;
        let mut output = vec![0i16; self.frame_samples];
        let mut state = self.lock_state()?;
        state
            .processor
            .process_render_i16(&input, &mut output)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// 对 10ms PCM16 麦克风帧执行回声消除。
    fn process_capture<'py>(&self, py: Python<'py>, pcm: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let input = decode_frame(pcm, self.frame_samples)?;
        let mut output = vec![0i16; self.frame_samples];
        let mut state = self.lock_state()?;
        state
            .processor
            .process_capture_i16(&input, &mut output)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(PyBytes::new(py, &encode_frame(&output)))
    }

    fn set_delay_ms(&self, delay_ms: i32) -> PyResult<()> {
        let mut state = self.lock_state()?;
        state
            .processor
            .set_stream_delay_ms(delay_ms)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        state.delay_ms = delay_ms;
        Ok(())
    }

    /// 清空 AEC 自适应状态，保留当前延迟配置。
    fn reset(&self) -> PyResult<()> {
        let mut state = self.lock_state()?;
        *state = AecState::new(state.delay_ms)?;
        Ok(())
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.lock_state()?;
        let stats = state.processor.statistics();
        let result = PyDict::new(py);
        result.set_item("stream_delay_ms", state.delay_ms)?;
        result.set_item("delay_ms", stats.delay_ms)?;
        result.set_item("delay_median_ms", stats.delay_median_ms)?;
        result.set_item(
            "delay_standard_deviation_ms",
            stats.delay_standard_deviation_ms,
        )?;
        result.set_item("echo_return_loss", stats.echo_return_loss)?;
        result.set_item(
            "echo_return_loss_enhancement",
            stats.echo_return_loss_enhancement,
        )?;
        result.set_item("residual_echo_likelihood", stats.residual_echo_likelihood)?;
        Ok(result)
    }
}

impl AecProcessor {
    fn lock_state(&self) -> PyResult<std::sync::MutexGuard<'_, AecState>> {
        self.state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("AEC processor lock poisoned"))
    }
}

fn decode_frame(pcm: &[u8], frame_samples: usize) -> PyResult<Vec<i16>> {
    let expected_bytes = frame_samples * 2;
    if pcm.len() != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "AEC frame must be {expected_bytes} bytes, got {}",
            pcm.len()
        )));
    }
    Ok(pcm
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
        .collect())
}

fn encode_frame(samples: &[i16]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        pcm.extend_from_slice(&sample.to_le_bytes());
    }
    pcm
}

pub fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AecProcessor>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_preserves_samples() {
        let samples = vec![-32768, -1, 0, 1, 32767];
        let encoded = encode_frame(&samples);
        let decoded = decode_frame(&encoded, samples.len()).unwrap();
        assert_eq!(samples, decoded);
    }

    #[test]
    fn rejects_wrong_frame_size() {
        let error = decode_frame(&[0; 10], 160).unwrap_err();
        assert!(error.to_string().contains("320 bytes"));
    }
}
