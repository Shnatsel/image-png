//! helpers for `fn unfilter`
//!
//! The code keeps the same structure as the `simd` module
//! but does not require nightly Rust compiler features

use std::convert::TryInto;

#[inline(always)]
/// This is an equivalent of the `PaethPredictor` function from
/// [the spec](http://www.libpng.org/pub/png/spec/1.2/PNG-Filters.html#Filter-type-4-Paeth)
/// except that it simultaneously calculates the predictor for all SIMD lanes.
/// Mapping between parameter names and pixel positions can be found in
/// [a diagram here](https://www.w3.org/TR/png/#filter-byte-positions).
///
/// Examples of how different pixel types may be represented as multiple SIMD lanes:
/// - RGBA => 4 lanes of `i16x4` contain R, G, B, A
/// - RGB  => 4 lanes of `i16x4` contain R, G, B, and a ignored 4th value
///
/// The SIMD algorithm below is based on [`libpng`](https://github.com/glennrp/libpng/blob/f8e5fa92b0e37ab597616f554bee254157998227/intel/filter_sse2_intrinsics.c#L261-L280).
/// 
/// So, funny thing: rustc will only autovectorize this with width 8 or longer.
/// Width 4 (which we actually would like) results in scalar code with pathetic performance.
/// That's why we hardcode 8 instead of making it a const generic.
fn paeth_predictor(
    a: [i16; 8],
    b: [i16; 8],
    c: [i16; 8],
) -> [i16; 8] {
    let mut out = [0; 8];
    for i in 0..8 {
        out[i] = super::filter_paeth_decode_i16(a[i].into(), b[i].into(), c[i].into());
    }
    out.into()
}

/// Memory of previous pixels (as needed to unfilter `FilterType::Paeth`).
/// See also https://www.w3.org/TR/png/#filter-byte-positions
#[derive(Default)]
struct PaethState {
    /// Previous pixel in the previous row.
    c: [i16; 8],

    /// Previous pixel in the current row.
    a: [i16; 8],
}

#[inline(always)]
/// Mutates `x` as needed to unfilter `FilterType::Paeth`.
///
/// `b` is the current pixel in the previous row.  `x` is the current pixel in the current row.
/// See also https://www.w3.org/TR/png/#filter-byte-positions
fn paeth_step<const N: usize>(state: &mut PaethState, b: [u8; N], x: &mut [u8; N]) {
    // Storing the inputs.
    let mut our_b: [i16; 8] = [0; 8];
    our_b[..N].iter_mut().zip(b).for_each(|(our, b)| *our = b as i16);

    // Calculating the new value of the current pixel.
    let predictor = paeth_predictor(state.a, our_b, state.c);
    x.iter_mut().zip(predictor).for_each(|(x, p)| *x += p as u8);

    // Preparing for the next step.
    state.c = our_b;
    let mut new_a: [i16; 8] = [0; 8];
    new_a[..N].iter_mut().zip(x).for_each(|(new, a)| *new = *a as i16);
    state.a = new_a;
}

fn load3(src: &[u8]) -> [u8; 4] {
    [src[0], src[1], src[2], 0]
}

fn store3(src: [u8; 4], dest: &mut [u8]) {
    dest[0..3].copy_from_slice(&src[0..3])
}

/// Undoes `FilterType::Paeth` for `BytesPerPixel::Three`.
pub fn unfilter_paeth3(mut prev_row: &[u8], mut curr_row: &mut [u8]) {
    debug_assert_eq!(prev_row.len(), curr_row.len());
    debug_assert_eq!(prev_row.len() % 3, 0);

    let mut state = PaethState::default();
    while prev_row.len() >= 4 {
        // `u8x4` requires working with `[u8;4]`, but we can just load and ignore the first
        // byte from the next triple.  This optimization technique mimics the algorithm found
        // in
        // https://github.com/glennrp/libpng/blob/f8e5fa92b0e37ab597616f554bee254157998227/intel/filter_sse2_intrinsics.c#L130-L131
        let b = prev_row[..4].try_into().unwrap();
        let mut x = curr_row[..4].try_into().unwrap();

        paeth_step(&mut state, b, &mut x);

        // We can speculate that writing 4 bytes might be more efficient (just as with using
        // `u8x4::from_slice` above), but we can't use that here, because we can't clobber the
        // first byte of the next pixel in the `curr_row`.
        store3(x, curr_row);

        prev_row = &prev_row[3..];
        curr_row = &mut curr_row[3..];
    }
    // Can't use `u8x4::from_slice` for the last `[u8;3]`.
    let b = load3(prev_row);
    let mut x = load3(curr_row);
    paeth_step(&mut state, b, &mut x);
    store3(x, curr_row);
}

fn load6(src: &[u8]) -> [u8; 8] {
    [src[0], src[1], src[2], src[3], src[4], src[5], 0, 0]
}

fn store6(src: [u8; 8], dest: &mut [u8]) {
    dest[0..6].copy_from_slice(&src[0..6])
}

/// Undoes `FilterType::Paeth` for `BytesPerPixel::Six`.
pub fn unfilter_paeth6(mut prev_row: &[u8], mut curr_row: &mut [u8]) {
    debug_assert_eq!(prev_row.len(), curr_row.len());
    debug_assert_eq!(prev_row.len() % 6, 0);

    let mut state = PaethState::default();
    while prev_row.len() >= 8 {
        // `u8x8` requires working with `[u8;8]`, but we can just load and ignore the first two
        // bytes from the next pixel.  This optimization technique mimics the algorithm found
        // in
        // https://github.com/glennrp/libpng/blob/f8e5fa92b0e37ab597616f554bee254157998227/intel/filter_sse2_intrinsics.c#L130-L131
        let b = prev_row[..8].try_into().unwrap();
        let mut x = curr_row[..8].try_into().unwrap();

        paeth_step(&mut state, b, &mut x);

        // We can speculate that writing 8 bytes might be more efficient (just as with using
        // `u8x8::from_slice` above), but we can't use that here, because we can't clobber the
        // first bytes of the next pixel in the `curr_row`.
        store6(x, curr_row);

        prev_row = &prev_row[6..];
        curr_row = &mut curr_row[6..];
    }
    // Can't use `u8x8::from_slice` for the last `[u8;6]`.
    let b = load6(prev_row);
    let mut x = load6(curr_row);
    paeth_step(&mut state, b, &mut x);
    store6(x, curr_row);
}
