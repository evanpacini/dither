// BEGIN: Diffusion matrix trait and implementations
pub trait DiffusionMatrix {
    fn enumerate(&self) -> Box<dyn EnumerateDiffusionMatrix + '_>;
}

pub struct DiffusionMatrixImpl<const M: usize, const N: usize> {
    offset: i32,
    weights: [[f32; N]; M],
}

impl<const M: usize, const N: usize> DiffusionMatrix for DiffusionMatrixImpl<M, N> {
    fn enumerate(&self) -> Box<dyn EnumerateDiffusionMatrix + '_> {
        Box::new(EnumerateDiffusionMatrixImpl { matrix: self, x: 0, y: 0 })
    }
}
// END: Diffusion matrix trait and implementations

// BEGIN: EnumerateDiffMatrix trait and implementations
pub trait EnumerateDiffusionMatrix {
    fn next_a(&mut self) -> Option<(i32, u32, f32)>;
}

impl Iterator for Box<dyn EnumerateDiffusionMatrix + '_> {
    type Item = (i32, u32, f32);

    fn next(&mut self) -> Option<Self::Item> {
        self.next_a()
    }
}

pub struct EnumerateDiffusionMatrixImpl<'a, const M: usize, const N: usize> {
    matrix: &'a DiffusionMatrixImpl<M, N>,
    x: usize,
    y: usize,
}

impl<const M: usize, const N: usize> EnumerateDiffusionMatrixImpl<'_, M, N> {
    fn next_weight(&mut self) {
        self.x += 1;
        if self.x >= N {
            self.x = 0;
            self.y += 1;
        }
    }
}

impl<const M: usize, const N: usize> EnumerateDiffusionMatrix
for EnumerateDiffusionMatrixImpl<'_, M, N>
{
    fn next_a(&mut self) -> Option<(i32, u32, f32)> {
        loop {
            if self.y >= M {
                return None;
            }
            let weight = self.matrix.weights[self.y][self.x];
            if weight != 0.0 {
                let x = self.x as i32 + self.matrix.offset;
                let y = self.y as u32;
                self.next_weight();
                return Some((x, y, weight));
            }
            self.next_weight();
        }
    }
}
// END: EnumerateDiffMatrix trait and implementations

// BEGIN: Diffusion matrices
/// Atkinson, W.D. (1984). Revealed in a private correspondence with John Balestrieri in January 2003 (unpublished).
pub const ATKINSON: DiffusionMatrixImpl<3, 4> = DiffusionMatrixImpl {
    offset: -1,
    weights: [
        [0., 0., 1.0 / 8.0, 1.0 / 8.0],
        [1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 0.],
        [0., 1.0 / 8.0, 0., 0.],
    ],
};

/// Burkes, D. (1988). _Presentation of the Burkes error filter for use in preparing continuous-tone
/// images for presentation on bi-level devices._
/// In LIB 15 (Publications), CIS Graphics Support Forum. (unpublished).
pub const BURKES: DiffusionMatrixImpl<2, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [0., 0., 0., 8.0 / 32.0, 4.0 / 32.0],
        [2.0 / 32.0, 4.0 / 32.0, 8.0 / 32.0, 4.0 / 32.0, 2.0 / 32.0],
    ],
};

/// Fan, Z. (1991). _A Simple Modification of Error Diffusion Weights._ IS&T's 44th Annual Conference
pub const FAN: DiffusionMatrixImpl<2, 4> = DiffusionMatrixImpl {
    offset: -2,
    weights: [[0., 0., 0., 7.0 / 16.0], [1.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 0.]],
};

/// Fedoseev, V. (2015). _Kernel weights optimization for error diffusion halftoning method_
/// Proceedings of SPIE 9445, Seventh International Conference on Machine Vision (ICMV 2014);
/// https://doi.org/10.1117/12.2180540
pub const FEDOSEEV_15: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [0., 0., 0., 0.5423, 0.0533],
        [0.0246, 0.2191, 0.4715, -0.0023, -0.1241],
        [-0.0065, -0.0692, 0.0168, -0.0952, -0.0304],
    ],
};
pub const FEDOSEEV_16: DiffusionMatrixImpl<2, 2> =
    DiffusionMatrixImpl { offset: 0, weights: [[0., 0.4364], [0.5636, 0.]] };
pub const FEDOSEEV_17: DiffusionMatrixImpl<2, 3> =
    DiffusionMatrixImpl { offset: -1, weights: [[0., 0., 0.4473], [0.1654, 0.3872, 0.]] };
pub const FEDOSEEV_18: DiffusionMatrixImpl<3, 3> = DiffusionMatrixImpl {
    offset: -1,
    weights: [[0., 0., 0.5221], [0.1854, 0.4689, 0.], [0., 0., -0.1763]],
};
pub const FEDOSEEV_19: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [0., 0., 0., 0.5, 0.0625],
        [0.015625, 0.25, 0.5, -0.001953125, -0.125],
        [-0.00390625, -0.0625, 0.015625, -0.125, -0.03125],
    ],
};
pub const FEDOSEEV_20: DiffusionMatrixImpl<3, 3> = DiffusionMatrixImpl {
    offset: -1,
    weights: [[0., 0., 0.5], [0.125, 0.5, 0.], [0., 0., -0.125]],
};

/// Floyd, R.W. & Steinberg, L. (1975). _An adaptive algorithm for spatial grey scale._
/// Society of Information Display Symposium, Digest of Technical Papers, 36–37.
pub const FLOYD_STEINBERG: DiffusionMatrixImpl<2, 3> = DiffusionMatrixImpl {
    offset: -1,
    weights: [[0., 0., 7.0 / 16.0], [3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0]],
};

/// Jarvis, J., Judice, C., & Ninke, W. (1976). _A survey of techniques for the display of continuous
/// tone pictures on bilevel displays._ Computer Graphics and Image Processing, 5(1), 13–40.
/// https://doi.org/10.1016/s0146-664x(76)80003-2
pub const JARVIS_JUDICE_NINKE: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [0., 0., 0., 7.0 / 48.0, 5.0 / 48.0],
        [3.0 / 48.0, 5.0 / 48.0, 7.0 / 48.0, 5.0 / 48.0, 3.0 / 48.0],
        [1.0 / 48.0, 3.0 / 48.0, 5.0 / 48.0, 3.0 / 48.0, 1.0 / 48.0],
    ],
};

/// Kolpatzik, B.W. & Bouman, C.A. (1992). _Optimized error diffusion for image display_
/// Journal of Electronic Imaging, 1(3). https://doi.org/10.1117/12.232863
pub const KOLPATZIK_BOUMAN_LUMINANCE: DiffusionMatrixImpl<2, 3> =
    DiffusionMatrixImpl { offset: -1, weights: [[0., 0., 0.777], [-0.009, 0.7861, -0.6098]] };
pub const KOLPATZIK_BOUMAN_CHROMINANCE: DiffusionMatrixImpl<2, 3> =
    DiffusionMatrixImpl { offset: -1, weights: [[0., 0., 0.8767], [0.0359, 0.8205, -0.7376]] };

/// Pigeon, S. (2013). _Dithering._ Harder, Better, Faster, Stronger.
/// https://hbfs.wordpress.com/2013/12/31/dithering/
pub const PIGEON: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [0., 0., 0., 2.0 / 14.0, 1.0 / 14.0],
        [0., 2.0 / 14.0, 2.0 / 14.0, 2.0 / 14.0, 0.],
        [1.0 / 14.0, 0., 1.0 / 14.0, 0., 1.0 / 14.0],
    ],
};

/// Shiau, J., & Fan, Z. (1996). _Set of easily implementable coefficients in error diffusion with reduced worm artifacts._
/// Proceedings of SPIE 2658, Color Imaging: Device-Independent Color, Color Hard Copy, and Graphic Arts;
/// https://doi.org/10.1117/12.236968
pub const SHIAU_FAN: DiffusionMatrixImpl<2, 5> = DiffusionMatrixImpl {
    offset: -3,
    weights: [[0., 0., 0., 0., 8.0 / 16.0], [1.0 / 16.0, 1.0 / 16.0, 2.0 / 16.0, 4.0 / 16.0, 0.]],
};

/// Sierra, F. (1989). In LIB 17 (Developer's Den), CIS Graphics Support Forum (unpublished).
pub const SIERRA: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [0., 0., 0., 5.0 / 32.0, 3.0 / 32.0],
        [2.0 / 32.0, 4.0 / 32.0, 5.0 / 32.0, 4.0 / 32.0, 2.0 / 32.0],
        [0., 2.0 / 32.0, 3.0 / 32.0, 2.0 / 32.0, 0.],
    ],
};
pub const SIERRA_TWO_ROW: DiffusionMatrixImpl<2, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [0., 0., 0., 4.0 / 16.0, 3.0 / 16.0],
        [1.0 / 16.0, 2.0 / 16.0, 3.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    ],
};
pub const SIERRA_LITE: DiffusionMatrixImpl<2, 3> =
    DiffusionMatrixImpl { offset: -1, weights: [[0., 0., 2.0 / 4.0], [1.0 / 4.0, 1.0 / 4.0, 0.]] };

/// Stucki, P. (1981). _MECCA - A Multiple-Error Correction Computation Algorithm for Bi-Level Image Hardcopy Reproduction._
/// Research Report RZ1060, IBM Zurich Research Laboratory.
pub const STUCKI: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [0., 0., 0., 8.0 / 42.0, 4.0 / 42.0],
        [2.0 / 42.0, 4.0 / 42.0, 8.0 / 42.0, 4.0 / 42.0, 2.0 / 42.0],
        [1.0 / 42.0, 2.0 / 42.0, 4.0 / 42.0, 2.0 / 42.0, 1.0 / 42.0],
    ],
};

/// Wong, P. W., & Allebach, J. P. (1997). _Optimum error-diffusion kernel design_
/// Proceedings of SPIE 3018,  Color Imaging: Device-Independent Color, Color Hard Copy, and Graphic Arts II;
/// https://doi.org/10.1117/12.271597
pub const WONG_ALLEBACH: DiffusionMatrixImpl<2, 3> =
    DiffusionMatrixImpl { offset: -1, weights: [[0., 0., 0.2911], [0.1373, 0.3457, 0.2258]] };
// END: Diffusion matrices
