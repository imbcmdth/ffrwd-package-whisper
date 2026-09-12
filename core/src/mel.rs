//! The log-mel spectrogram whisper reads: 16 kHz mono samples in, 80 by 3000
//! floats out.
//!
//! The arithmetic is OpenAI's own front end. The samples are reflected at both
//! ends by half a transform, cut into 400-point frames a hop apart, windowed
//! and transformed; the power spectrum goes through the mel filterbank; the
//! result is log10, clamped eight decades below the loudest value and squeezed
//! into roughly -1 to 1. The last column is dropped, which is what makes 30 s
//! of audio exactly 3000 frames.
//!
//! The filterbank is the table whisper was trained with, compiled in as the
//! little-endian floats it is stored as. Computing it here instead would put
//! this module's idea of the mel scale between the audio and weights that were
//! fitted against that table.

/// The rate whisper listens at, and the only one this module accepts.
pub const SAMPLE_RATE: u32 = 16_000;

/// Points in one transform.
pub const N_FFT: usize = 400;

/// Samples between one frame and the next.
pub const HOP: usize = 160;

/// Mel bins the filterbank has.
pub const N_MELS: usize = 80;

/// Frequency bins one transform keeps.
pub const N_BINS: usize = N_FFT / 2 + 1;

/// Frames the model's input tensor holds.
pub const N_FRAMES: usize = 3000;

/// Samples one window covers: 30 s, which is the model's fixed input.
pub const WINDOW_SAMPLES: usize = N_FRAMES * HOP;

/// Seconds one window covers.
pub const WINDOW_SECONDS: f64 = WINDOW_SAMPLES as f64 / SAMPLE_RATE as f64;

/// How many of a window's [`N_FRAMES`] columns are audio rather than padding.
///
/// [`Plan::spectrogram`] always returns the model's full 3000, zero-filling
/// whatever the samples did not reach - the encoder needs its fixed input. A
/// caller that has to know where the audio actually stopped, as the word
/// alignment does, asks here: one frame per hop, and the last partial hop is
/// not a frame.
pub fn content_frames(samples: usize) -> usize {
    (samples / HOP).min(N_FRAMES)
}

/// The mel filterbank, 80 by 201 f32 as whisper ships it.
const FILTERS_BYTES: &[u8] = include_bytes!("../melfilters.bytes");

/// How far the transform is split by twos before the direct one takes over:
/// 400 is 16 times 25.
const RADIX2: usize = 16;
const BASE: usize = N_FFT / RADIX2;

/// The filterbank as floats, bin-major: `[mel * N_BINS + bin]`.
fn filters() -> Vec<f32> {
    let (words, _) = FILTERS_BYTES.as_chunks::<4>();
    words.iter().copied().map(f32::from_le_bytes).collect()
}

/// Position `i` lands at after the decimation that puts the 25-point
/// transforms in the order the combining stages expect.
fn scrambled(i: usize) -> usize {
    let (digit, group) = (i % RADIX2, i / RADIX2);
    // Four bits reversed, which is the decimation's own permutation.
    let mut reversed = 0;
    for bit in 0..4 {
        reversed |= ((digit >> bit) & 1) << (3 - bit);
    }
    reversed * BASE + group
}

/// The 400-point transform, planned once and run per frame.
///
/// Sixteen 25-point transforms taken directly, then four stages of butterflies
/// over them. Every twiddle is computed here rather than per frame, which is
/// what makes 3000 frames of a window cost sixteen tables' worth of arithmetic
/// and nothing else.
pub struct Plan {
    /// Where each input sample goes before the transforms run.
    order: [usize; N_FFT],
    /// The 25-point transform's own twiddles, `[row * BASE + column]`.
    base: Vec<(f32, f32)>,
    /// One stage's twiddles, innermost stage first.
    stages: Vec<Vec<(f32, f32)>>,
    /// The periodic Hann window each frame is multiplied by.
    hann: [f32; N_FFT],
    filters: Vec<f32>,
}

impl Default for Plan {
    fn default() -> Plan {
        Plan::new()
    }
}

impl Plan {
    pub fn new() -> Plan {
        let tau = std::f64::consts::TAU;

        let mut order = [0usize; N_FFT];
        for (i, slot) in order.iter_mut().enumerate() {
            *slot = scrambled(i);
        }

        let mut base = Vec::with_capacity(BASE * BASE);
        for row in 0..BASE {
            for column in 0..BASE {
                let angle = -tau * (row * column % BASE) as f64 / BASE as f64;
                base.push((angle.cos() as f32, angle.sin() as f32));
            }
        }

        let mut stages = Vec::new();
        let mut half = BASE;
        while half < N_FFT {
            let whole = half * 2;
            let twiddles = (0..half)
                .map(|k| {
                    let angle = -tau * k as f64 / whole as f64;
                    (angle.cos() as f32, angle.sin() as f32)
                })
                .collect();
            stages.push(twiddles);
            half = whole;
        }

        let mut hann = [0f32; N_FFT];
        for (i, slot) in hann.iter_mut().enumerate() {
            *slot = (0.5 - 0.5 * (tau * i as f64 / N_FFT as f64).cos()) as f32;
        }

        Plan {
            order,
            base,
            stages,
            hann,
            filters: filters(),
        }
    }

    /// One frame's power spectrum, the 201 bins a real transform keeps.
    ///
    /// `frame` is 400 samples; `re` and `im` are scratch the caller owns, so a
    /// window's frames share one allocation.
    fn power(&self, frame: &[f32], re: &mut [f32; N_FFT], im: &mut [f32; N_FFT], out: &mut [f32]) {
        for (i, &sample) in frame.iter().enumerate() {
            re[self.order[i]] = sample * self.hann[i];
        }
        im.fill(0.0);

        // The direct transforms, one per block of 25.
        let mut block_re = [0f32; BASE];
        let mut block_im = [0f32; BASE];
        for block in 0..RADIX2 {
            let at = block * BASE;
            block_re.copy_from_slice(&re[at..at + BASE]);
            for row in 0..BASE {
                let (mut sum_re, mut sum_im) = (0f32, 0f32);
                let turn = &self.base[row * BASE..(row + 1) * BASE];
                for (value, (cos, sin)) in block_re.iter().zip(turn) {
                    sum_re += value * cos;
                    sum_im += value * sin;
                }
                re[at + row] = sum_re;
                block_im[row] = sum_im;
            }
            im[at..at + BASE].copy_from_slice(&block_im);
        }

        // The butterflies, doubling the transform's length each stage.
        let mut half = BASE;
        for twiddles in &self.stages {
            let whole = half * 2;
            let mut at = 0;
            while at < N_FFT {
                for (k, &(cos, sin)) in twiddles.iter().enumerate() {
                    let (low, high) = (at + k, at + k + half);
                    let (odd_re, odd_im) = (re[high], im[high]);
                    let turned_re = odd_re * cos - odd_im * sin;
                    let turned_im = odd_re * sin + odd_im * cos;
                    re[high] = re[low] - turned_re;
                    im[high] = im[low] - turned_im;
                    re[low] += turned_re;
                    im[low] += turned_im;
                }
                at += whole;
            }
            half = whole;
        }

        for (bin, slot) in out.iter_mut().enumerate() {
            *slot = re[bin] * re[bin] + im[bin] * im[bin];
        }
    }

    /// The spectrogram of one window of 16 kHz mono samples, mel bin by mel
    /// bin: `[mel * N_FRAMES + frame]`, which is the layout the model's
    /// `input_features` tensor wants.
    ///
    /// Fewer samples than a whole window are padded out with silence, which is
    /// what whisper itself does with a short final window.
    pub fn spectrogram(&self, samples: &[f32]) -> Vec<f32> {
        let taken = samples.len().min(WINDOW_SAMPLES);
        let half = N_FFT / 2;
        // Reflected at both ends by half a transform, so the first and last
        // frames are centred on the first and last samples.
        let mut padded = vec![0f32; half + WINDOW_SAMPLES + half];
        padded[half..half + taken].copy_from_slice(&samples[..taken]);
        for k in 0..half.min(taken.saturating_sub(1)) {
            padded[half - 1 - k] = samples[k + 1];
        }
        let audio_end = half + WINDOW_SAMPLES;
        for k in 0..half {
            padded[audio_end + k] = padded[audio_end - 2 - k];
        }

        let mut mel = vec![0f32; N_MELS * N_FRAMES];
        let mut re = [0f32; N_FFT];
        let mut im = [0f32; N_FFT];
        let mut power = [0f32; N_BINS];
        for frame in 0..N_FRAMES {
            let at = frame * HOP;
            self.power(&padded[at..at + N_FFT], &mut re, &mut im, &mut power);
            for bin in 0..N_MELS {
                let row = &self.filters[bin * N_BINS..(bin + 1) * N_BINS];
                let mut sum = 0f32;
                for (weight, energy) in row.iter().zip(&power) {
                    sum += weight * energy;
                }
                mel[bin * N_FRAMES + frame] = sum.max(1e-10).log10();
            }
        }

        // Everything more than eight decades below the loudest is that quiet,
        // and the scale is squeezed into roughly -1 to 1.
        let floor = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
        for value in mel.iter_mut() {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
        mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform the plan runs, as the complex bins it produces.
    fn transform(plan: &Plan, frame: &[f32]) -> Vec<f32> {
        let mut re = [0f32; N_FFT];
        let mut im = [0f32; N_FFT];
        let mut power = [0f32; N_BINS];
        // The window is folded into the transform, so a test comparing it
        // against the direct sum divides it back out where it can.
        plan.power(frame, &mut re, &mut im, &mut power);
        power.to_vec()
    }

    /// The power spectrum of a windowed frame, summed straight out of the
    /// definition. Slow, and the only thing the planned transform is checked
    /// against.
    fn directly(plan: &Plan, frame: &[f32]) -> Vec<f32> {
        let tau = std::f64::consts::TAU;
        (0..N_BINS)
            .map(|k| {
                let (mut re, mut im) = (0f64, 0f64);
                for (n, &sample) in frame.iter().enumerate() {
                    let angle = -tau * (k * n % N_FFT) as f64 / N_FFT as f64;
                    let value = f64::from(sample * plan.hann[n]);
                    re += value * angle.cos();
                    im += value * angle.sin();
                }
                (re * re + im * im) as f32
            })
            .collect()
    }

    #[test]
    fn the_filterbank_is_eighty_mel_bins_of_two_hundred_and_one() {
        assert_eq!(FILTERS_BYTES.len(), N_MELS * N_BINS * 4);
        assert_eq!(filters().len(), N_MELS * N_BINS);
    }

    #[test]
    fn the_decimation_is_a_permutation() {
        let mut seen = vec![false; N_FFT];
        for i in 0..N_FFT {
            let at = scrambled(i);
            assert!(!seen[at], "{i} lands where something already is");
            seen[at] = true;
        }
    }

    #[test]
    fn the_hann_window_opens_and_closes_at_nothing() {
        let plan = Plan::new();
        assert!(plan.hann[0].abs() < 1e-7, "it opens at nothing");
        assert!(
            (plan.hann[N_FFT / 2] - 1.0).abs() < 1e-6,
            "and peaks in the middle"
        );
    }

    #[test]
    fn the_planned_transform_agrees_with_the_definition() {
        let plan = Plan::new();
        let frame: Vec<f32> = (0..N_FFT).map(|i| (i as f32 * 0.37).sin() * 0.5).collect();
        let planned = transform(&plan, &frame);
        let summed = directly(&plan, &frame);
        let scale = summed.iter().copied().fold(0f32, f32::max).max(1e-6);
        for (bin, (a, b)) in planned.iter().zip(&summed).enumerate() {
            assert!((a - b).abs() / scale < 1e-4, "bin {bin}: {a} against {b}");
        }
    }

    #[test]
    fn a_tone_lands_in_the_bin_it_belongs_to() {
        // 800 Hz at 16 kHz over 400 points is exactly bin 20.
        let plan = Plan::new();
        let frame: Vec<f32> = (0..N_FFT)
            .map(|i| (std::f32::consts::TAU * 20.0 * i as f32 / N_FFT as f32).sin())
            .collect();
        let power = transform(&plan, &frame);
        let loudest = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(bin, _)| bin)
            .expect("the spectrum is not empty");
        assert_eq!(loudest, 20);
    }

    #[test]
    fn a_window_of_audio_is_eighty_by_three_thousand() {
        let plan = Plan::new();
        let samples: Vec<f32> = (0..WINDOW_SAMPLES)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        assert_eq!(plan.spectrogram(&samples).len(), N_MELS * N_FRAMES);
    }

    #[test]
    fn a_short_window_is_padded_out_to_the_same_shape() {
        let plan = Plan::new();
        for samples in [0usize, 1, 1000, WINDOW_SAMPLES / 2] {
            let mel = plan.spectrogram(&vec![0.1f32; samples]);
            assert_eq!(mel.len(), N_MELS * N_FRAMES, "{samples} samples");
        }
    }

    #[test]
    fn the_spectrogram_spans_exactly_two_however_loud_the_audio_is() {
        // Eight decades clamped and divided by four: the loudest value and the
        // quietest are two apart, wherever on the scale they sit.
        let plan = Plan::new();
        let samples: Vec<f32> = (0..WINDOW_SAMPLES)
            .map(|i| if i % 3 == 0 { 0.9 } else { -0.00001 })
            .collect();
        let mel = plan.spectrogram(&samples);
        let loudest = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let quietest = mel.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            (loudest - quietest - 2.0).abs() < 1e-5,
            "{quietest} to {loudest}"
        );
    }

    #[test]
    fn silence_alone_is_the_floor_everywhere() {
        let plan = Plan::new();
        let mel = plan.spectrogram(&vec![0.0f32; WINDOW_SAMPLES]);
        let first = mel[0];
        for value in &mel {
            assert!((value - first).abs() < 1e-6, "{value} against {first}");
        }
    }
}
