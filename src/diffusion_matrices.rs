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
pub const ATKINSON: DiffusionMatrixImpl<3, 4> = DiffusionMatrixImpl {
    offset: -1,
    weights: [
        [None, None, Some(1.0 / 8.0), Some(1.0 / 8.0)],
        [Some(1.0 / 8.0), Some(1.0 / 8.0), Some(1.0 / 8.0), None],
        [None, Some(1.0 / 8.0), None, None],
    ],
};

pub const BURKES: DiffusionMatrixImpl<2, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(8.0 / 32.0), Some(4.0 / 32.0)],
        [Some(2.0 / 32.0), Some(4.0 / 32.0), Some(8.0 / 32.0), Some(4.0 / 32.0), Some(2.0 / 32.0)],
    ],
};

pub const FLOYD_STEINBERG: DiffusionMatrixImpl<2, 3> = DiffusionMatrixImpl {
    offset: -1,
    weights: [
        [None, None, Some(7.0 / 16.0)],
        [Some(3.0 / 16.0), Some(5.0 / 16.0), Some(1.0 / 16.0)],
    ],
};

pub const JARVIS_JUDICE_NINKE: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(7.0 / 48.0), Some(5.0 / 48.0)],
        [Some(3.0 / 48.0), Some(5.0 / 48.0), Some(7.0 / 48.0), Some(5.0 / 48.0), Some(3.0 / 48.0)],
        [Some(1.0 / 48.0), Some(3.0 / 48.0), Some(5.0 / 48.0), Some(3.0 / 48.0), Some(1.0 / 48.0)],
    ],
};

// https://hbfs.wordpress.com/2013/12/31/dithering/
pub const PIGEON: DiffusionMatrixImpl<2, 4> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(7.0 / 16.0)],
        [Some(1.0 / 16.0), Some(3.0 / 16.0), Some(5.0 / 16.0), None],
    ],
};

// https://doi-org.tudelft.idm.oclc.org/10.1117/12.236968
pub const SHIAU_FAN: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(8.0 / 48.0), Some(4.0 / 48.0)],
        [Some(2.0 / 48.0), Some(4.0 / 48.0), Some(8.0 / 48.0), Some(4.0 / 48.0), Some(2.0 / 48.0)],
        [None, Some(2.0 / 48.0), Some(4.0 / 48.0), Some(2.0 / 48.0), None],
    ],
};

pub const SIERRA: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(5.0 / 32.0), Some(3.0 / 32.0)],
        [Some(2.0 / 32.0), Some(4.0 / 32.0), Some(5.0 / 32.0), Some(4.0 / 32.0), Some(2.0 / 32.0)],
        [None, Some(2.0 / 32.0), Some(3.0 / 32.0), Some(2.0 / 32.0), None],
    ],
};

pub const SIERRA_TWO_ROW: DiffusionMatrixImpl<2, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(4.0 / 16.0), Some(3.0 / 16.0)],
        [Some(1.0 / 16.0), Some(2.0 / 16.0), Some(3.0 / 16.0), Some(2.0 / 16.0), Some(1.0 / 16.0)],
    ],
};

pub const SIERRA_LITE: DiffusionMatrixImpl<2, 3> = DiffusionMatrixImpl {
    offset: -1,
    weights: [[None, None, Some(2.0 / 4.0)], [Some(1.0 / 4.0), Some(1.0 / 4.0), None]],
};

pub const STUCKI: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(8.0 / 42.0), Some(4.0 / 42.0)],
        [Some(2.0 / 42.0), Some(4.0 / 42.0), Some(8.0 / 42.0), Some(4.0 / 42.0), Some(2.0 / 42.0)],
        [Some(1.0 / 42.0), Some(2.0 / 42.0), Some(4.0 / 42.0), Some(2.0 / 42.0), Some(1.0 / 42.0)],
    ],
};
// END: Diffusion matrices
