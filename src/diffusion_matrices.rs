use crate::DiffusionMatrixImpl;

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
    weights: [
        [None, None, Some(2.0 / 4.0)],
        [Some(1.0 / 4.0), Some(1.0 / 4.0), None],
    ],
};

pub const STUCKI: DiffusionMatrixImpl<3, 5> = DiffusionMatrixImpl {
    offset: -2,
    weights: [
        [None, None, None, Some(8.0 / 42.0), Some(4.0 / 42.0)],
        [Some(2.0 / 42.0), Some(4.0 / 42.0), Some(8.0 / 42.0), Some(4.0 / 42.0), Some(2.0 / 42.0)],
        [Some(1.0 / 42.0), Some(2.0 / 42.0), Some(4.0 / 42.0), Some(2.0 / 42.0), Some(1.0 / 42.0)],
    ],
};
