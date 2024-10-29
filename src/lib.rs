use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};

pub mod diffusion_matrices;

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

/// Quantise an image to binary (black and white) using a threshold of 128.
pub fn threshold_quantise(img: &DynamicImage) -> GrayImage {
    let mut img = img.to_luma16();
    for pixel in img.pixels_mut() {
        pixel.0[0] = if pixel.0[0] < 32768 { 0 } else { 65535 };
    }
    DynamicImage::ImageLuma16(img).to_luma8()
}

/// Quantise an image to binary (black and white) using a random threshold. (Random dithering)
pub fn random_dither_quantise(img: &DynamicImage) -> GrayImage {
    let mut img = img.to_luma16();
    for pixel in img.pixels_mut() {
        let threshold = rand::random::<u16>();
        pixel.0[0] = if pixel.0[0] < threshold { 0 } else { 65535 };
    }
    DynamicImage::ImageLuma16(img).to_luma8()
}

/// Quantise an image to binary (black and white) using IGN threshold.
pub fn ign_quantise(img: &DynamicImage) -> GrayImage {
    let mut img = img.to_luma16();
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let threshold =
            ((52.9829189 * 0.06711056_f32.mul_add(x as f32, 0.00583715 * y as f32).fract()).fract()
                * 65535.0)
                .round() as u16;
        pixel.0[0] = if pixel.0[0] < threshold { 0 } else { 65535 };
    }
    DynamicImage::ImageLuma16(img).to_luma8()
}

/// Quantise an image to binary (black and white) using ordered dithering.
pub fn ordered_dither_quantise(
    img: &DynamicImage, matrix: &ImageBuffer<Luma<u16>, Vec<u16>>,
) -> GrayImage {
    let mut img = img.to_luma16();
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let threshold = matrix.get_pixel(x % matrix.width(), y % matrix.height()).0[0];
        pixel.0[0] = if pixel.0[0] < threshold { 0 } else { 65535 };
    }
    DynamicImage::ImageLuma16(img).to_luma8()
}

/// Quantise an image to binary (black and white) using error diffusion.
pub fn error_diffusion_quantise(img: &DynamicImage, matrix: &dyn DiffusionMatrix) -> GrayImage {
    let mut img = img.to_luma32f();
    for y in 0..img.height() {
        for x in 0..img.width() {
            let old_pixel = img.get_pixel(x, y).0[0];
            let new_pixel = if old_pixel < 0.5 { 0.0 } else { 1.0 };
            let error = old_pixel - new_pixel;
            img.put_pixel(x, y, Luma([new_pixel]));
            for (dx, dy, weight) in matrix.enumerate() {
                let nx = x as i32 + dx;
                let ny = y + dy;
                if nx < 0 || !img.in_bounds(nx as u32, ny) {
                    continue;
                }
                let pixel = img.get_pixel(nx as u32, ny).0[0];
                let new_pixel = pixel + error * weight;
                img.put_pixel(nx as u32, ny, Luma([new_pixel]));
            }
        }
    }
    DynamicImage::from(img).to_luma8()
}
