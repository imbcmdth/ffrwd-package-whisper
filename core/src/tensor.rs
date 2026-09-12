//! The bytes a `wasi:nn` tensor carries, and the spelling of a failure.
//!
//! Nothing here names a wasi-nn type: a tensor's payload is a flat list of
//! little-endian numbers whichever module is holding it, so the conversions
//! are the same conversions for both exports and they are tested on the host
//! without a graph anywhere near them.

/// Floats as the little-endian bytes a tensor carries.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Whole numbers as the little-endian bytes a tensor carries.
pub fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A tensor's bytes back as the whole numbers they spell.
pub fn to_i32(bytes: &[u8]) -> Vec<i32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().copied().map(i32::from_le_bytes).collect()
}

/// A tensor's bytes back as the floats they spell. A trailing part-word is
/// dropped rather than guessed at, the same way `to_i32` drops one.
pub fn to_f32(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().copied().map(f32::from_le_bytes).collect()
}

/// A failure, in the spec's own spelling of the error code, so a message says
/// what actually went wrong rather than how a module happens to format things.
/// The caller maps its own generated `error-code` enum to `code`, which is the
/// one part of this that cannot be shared: each module's bindings mint their
/// own copy of that type.
pub fn failed(module: &str, what: &str, code: &str, data: &str) -> String {
    format!("{module}: {what}: {code} ({data})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_numbers_survive_the_trip_through_a_tensors_bytes() {
        let values = [0i32, -1, 50258, i32::MAX];
        assert_eq!(to_i32(&i32_bytes(&values)), values);
    }

    #[test]
    fn floats_survive_the_trip_through_a_tensors_bytes() {
        let values = [0.0f32, -1.5, 1e-10, f32::MAX];
        assert_eq!(to_f32(&f32_bytes(&values)), values);
    }

    #[test]
    fn a_trailing_part_word_is_dropped_rather_than_guessed_at() {
        assert_eq!(to_i32(&[1, 0, 0, 0, 9]), vec![1]);
        assert_eq!(to_f32(&[0, 0, 0, 0, 9, 9]), vec![0.0]);
        assert!(to_i32(&[1, 2, 3]).is_empty());
    }

    #[test]
    fn a_failure_names_the_module_the_step_and_the_spec_code() {
        assert_eq!(
            failed("transcribe", "compute", "runtime-error", "no kernel"),
            "transcribe: compute: runtime-error (no kernel)"
        );
    }
}
