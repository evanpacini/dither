// BEGIN: Diffusion matrix trait and implementations
pub trait DiffusionMatrix {
    fn enumerate(&self) -> Box<dyn EnumerateDiffusionMatrix + '_>;
}

pub struct DiffusionMatrixImpl<const M: usize, const N: usize> {
    offset: i32,
    weights: [[Option<f32>; N]; M],
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
            if let Some(weight) = self.matrix.weights[self.y][self.x] {
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
        [None, None, Some(1.0 / 8.0), Some(1.0 / 8.0)],
        [Some(1.0 / 8.0), Some(1.0 / 8.0), Some(1.0 / 8.0), None],
        [None, Some(1.0 / 8.0), None, None],
    ],
};

/// Burkes, D. (1988). _Presentation of the Burkes error filter for use in preparing continuous-tone
/// images for presentation on bi-level devices._
/// In LIB 15 (Publications), CIS Graphics Support Forum. (unpublished).
pub const BURKES: DiffusionMatrixImpl<2, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(8.0 / 32.0), Some(4.0 / 32.0)],
        [Some(2.0 / 32.0), Some(4.0 / 32.0), Some(8.0 / 32.0), Some(4.0 / 32.0), Some(2.0 / 32.0)],
    ],
};

/// Floyd, R.W. & Steinberg, L. (1975). _An adaptive algorithm for spatial grey scale._
/// Society of Information Display Symposium, Digest of Technical Papers, 36–37.
pub const FLOYD_STEINBERG: DiffusionMatrixImpl<2, 3> = DiffusionMatrixImpl {
    offset: -1,
    weights: [
        [None, None, Some(7.0 / 16.0)],
        [Some(3.0 / 16.0), Some(5.0 / 16.0), Some(1.0 / 16.0)],
    ],
};

/// Jarvis, J., Judice, C., & Ninke, W. (1976). _A survey of techniques for the display of continuous
/// tone pictures on bilevel displays._ Computer Graphics and Image Processing, 5(1), 13–40.
/// https://doi.org/10.1016/s0146-664x(76)80003-2
pub const JARVIS_JUDICE_NINKE: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(7.0 / 48.0), Some(5.0 / 48.0)],
        [Some(3.0 / 48.0), Some(5.0 / 48.0), Some(7.0 / 48.0), Some(5.0 / 48.0), Some(3.0 / 48.0)],
        [Some(1.0 / 48.0), Some(3.0 / 48.0), Some(5.0 / 48.0), Some(3.0 / 48.0), Some(1.0 / 48.0)],
    ],
};

/// Pigeon, S. (2013). _Dithering._ Harder, Better, Faster, Stronger.
/// https://hbfs.wordpress.com/2013/12/31/dithering/
pub const PIGEON: DiffusionMatrixImpl<2, 4> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(7.0 / 16.0)],
        [Some(1.0 / 16.0), Some(3.0 / 16.0), Some(5.0 / 16.0), None],
    ],
};

/// Shiau, J., & Fan, Z. (1996). _Set of easily implementable coefficients in error diffusion with reduced worm artifacts._
/// Proceedings of SPIE 2658, Color Imaging: Device-Independent Color, Color Hard Copy, and Graphic Arts,
/// https://doi.org/10.1117/12.236968
pub const SHIAU_FAN: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(8.0 / 48.0), Some(4.0 / 48.0)],
        [Some(2.0 / 48.0), Some(4.0 / 48.0), Some(8.0 / 48.0), Some(4.0 / 48.0), Some(2.0 / 48.0)],
        [None, Some(2.0 / 48.0), Some(4.0 / 48.0), Some(2.0 / 48.0), None],
    ],
};

/// Sierra, F. (1989). In LIB 17 (Developer's Den), CIS Graphics Support Forum (unpublished).
pub const SIERRA: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(5.0 / 32.0), Some(3.0 / 32.0)],
        [Some(2.0 / 32.0), Some(4.0 / 32.0), Some(5.0 / 32.0), Some(4.0 / 32.0), Some(2.0 / 32.0)],
        [None, Some(2.0 / 32.0), Some(3.0 / 32.0), Some(2.0 / 32.0), None],
    ],
};

/// Sierra, F. (1990). In LIB 17 (Developer's Den), CIS Graphics Support Forum (unpublished).
pub const SIERRA_TWO_ROW: DiffusionMatrixImpl<2, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(4.0 / 16.0), Some(3.0 / 16.0)],
        [Some(1.0 / 16.0), Some(2.0 / 16.0), Some(3.0 / 16.0), Some(2.0 / 16.0), Some(1.0 / 16.0)],
    ],
};

/// Sierra, F. (1990). In LIB 17 (Developer's Den), CIS Graphics Support Forum (unpublished).
pub const SIERRA_LITE: DiffusionMatrixImpl<2, 3> = DiffusionMatrixImpl {
    offset: -1,
    weights: [[None, None, Some(2.0 / 4.0)], [Some(1.0 / 4.0), Some(1.0 / 4.0), None]],
};

/// Stucki, P. (1981). _MECCA - A Multiple-Error Correction Computation Algorithm for Bi-Level Image Hardcopy Reproduction._
/// Research Report RZ1060, IBM Zurich Research Laboratory.
pub const STUCKI: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(8.0 / 42.0), Some(4.0 / 42.0)],
        [Some(2.0 / 42.0), Some(4.0 / 42.0), Some(8.0 / 42.0), Some(4.0 / 42.0), Some(2.0 / 42.0)],
        [Some(1.0 / 42.0), Some(2.0 / 42.0), Some(4.0 / 42.0), Some(2.0 / 42.0), Some(1.0 / 42.0)],
    ],
};
// END: Diffusion matrices
