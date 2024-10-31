use image::{DynamicImage, ImageBuffer, Luma};

// BEGIN: Define a trait that will encapsulate both threshold matrices and functions
pub trait OrderedThreshold {
    fn get(&self, x: u32, y: u32) -> f32;
}

impl OrderedThreshold for ImageBuffer<Luma<f32>, Vec<f32>> {
    fn get(&self, x: u32, y: u32) -> f32 {
        self.get_pixel(x % self.width(), y % self.height()).0[0]
    }
}

struct ThresholdMatrix {
    matrix: Vec<Vec<f32>>,
}

impl OrderedThreshold for ThresholdMatrix {
    fn get(&self, x: u32, y: u32) -> f32 {
        self.matrix[y as usize % self.matrix.len()][x as usize % self.matrix[0].len()]
    }
}

impl From<&DynamicImage> for ThresholdMatrix {
    fn from(img: &DynamicImage) -> Self {
        Self {
            matrix: img
                .to_luma32f()
                .to_vec()
                .chunks(img.width() as usize)
                .map(|row| row.to_vec())
                .collect(),
        }
    }
}

impl OrderedThreshold for fn(u32, u32) -> f32 {
    fn get(&self, x: u32, y: u32) -> f32 {
        self(x, y)
    }
}

// END: Define a trait that will encapsulate both threshold matrices and functions

// BEGIN: Define threshold functions
/// Flat threshold
pub const FLAT: fn(u32, u32) -> f32 = |_, _| 0.5;

/// Random threshold
pub const RANDOM: fn(u32, u32) -> f32 = |_, _| rand::random::<f32>();

/// Interleaved Gradient Noise (IGN) threshold
/// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
pub const IGN: fn(u32, u32) -> f32 =
    |x, y| (52.9829189 * 0.06711056_f32.mul_add(x as f32, 0.00583715 * y as f32).fract()).fract();
// END: Define threshold functions
