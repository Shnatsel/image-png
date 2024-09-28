//! helpers for `fn unfilter`
//!
//! The code keeps the same structure as the `simd` module
//! but does not require nightly Rust compiler features

use std::simd::num::{SimdInt, SimdUint};
use std::simd::{u8x4, u8x8, LaneCount, Simd, SimdElement, SupportedLaneCount};

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
fn paeth_predictor<const N: usize>(
    a: Simd<i16, N>,
    b: Simd<i16, N>,
    c: Simd<i16, N>,
) -> Simd<i16, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut out = [0; N];
    for i in 0..N {
        out[i] = super::filter_paeth_decode_i16(a[i].into(), b[i].into(), c[i].into());
    }
    out.into()
}

/// Memory of previous pixels (as needed to unfilter `FilterType::Paeth`).
/// See also https://www.w3.org/TR/png/#filter-byte-positions
#[derive(Default)]
struct PaethState<T, const N: usize>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
{
    /// Previous pixel in the previous row.
    c: Simd<T, N>,

    /// Previous pixel in the current row.
    a: Simd<T, N>,
}

/// Mutates `x` as needed to unfilter `FilterType::Paeth`.
///
/// `b` is the current pixel in the previous row.  `x` is the current pixel in the current row.
/// See also https://www.w3.org/TR/png/#filter-byte-positions
fn paeth_step<const N: usize>(state: &mut PaethState<i16, N>, b: Simd<u8, N>, x: &mut Simd<u8, N>)
where
    LaneCount<N>: SupportedLaneCount,
{
    // Storing the inputs.
    let b = b.cast::<i16>();

    // Calculating the new value of the current pixel.
    let predictor = paeth_predictor(state.a, b, state.c);
    *x += predictor.cast::<u8>();

    // Preparing for the next step.
    state.c = b;
    state.a = x.cast::<i16>();
}

fn load3(src: &[u8]) -> u8x4 {
    u8x4::from_array([src[0], src[1], src[2], 0])
}

fn store3(src: u8x4, dest: &mut [u8]) {
    dest[0..3].copy_from_slice(&src.to_array()[0..3])
}

/// Undoes `FilterType::Paeth` for `BytesPerPixel::Three`.
pub fn unfilter_paeth3(mut prev_row: &[u8], mut curr_row: &mut [u8]) {
    debug_assert_eq!(prev_row.len(), curr_row.len());
    debug_assert_eq!(prev_row.len() % 3, 0);

    let mut state = PaethState::<i16, 4>::default();
    while prev_row.len() >= 4 {
        // `u8x4` requires working with `[u8;4]`, but we can just load and ignore the first
        // byte from the next triple.  This optimization technique mimics the algorithm found
        // in
        // https://github.com/glennrp/libpng/blob/f8e5fa92b0e37ab597616f554bee254157998227/intel/filter_sse2_intrinsics.c#L130-L131
        let b = u8x4::from_slice(prev_row);
        let mut x = u8x4::from_slice(curr_row);

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

fn load6(src: &[u8]) -> u8x8 {
    u8x8::from_array([src[0], src[1], src[2], src[3], src[4], src[5], 0, 0])
}

fn store6(src: u8x8, dest: &mut [u8]) {
    dest[0..6].copy_from_slice(&src.to_array()[0..6])
}

/// Undoes `FilterType::Paeth` for `BytesPerPixel::Six`.
pub fn unfilter_paeth6(mut prev_row: &[u8], mut curr_row: &mut [u8]) {
    debug_assert_eq!(prev_row.len(), curr_row.len());
    debug_assert_eq!(prev_row.len() % 6, 0);

    let mut state = PaethState::<i16, 8>::default();
    while prev_row.len() >= 8 {
        // `u8x8` requires working with `[u8;8]`, but we can just load and ignore the first two
        // bytes from the next pixel.  This optimization technique mimics the algorithm found
        // in
        // https://github.com/glennrp/libpng/blob/f8e5fa92b0e37ab597616f554bee254157998227/intel/filter_sse2_intrinsics.c#L130-L131
        let b = u8x8::from_slice(prev_row);
        let mut x = u8x8::from_slice(curr_row);

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
