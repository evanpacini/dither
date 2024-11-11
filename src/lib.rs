use image::{DynamicImage, GenericImageView, GrayImage, Luma};

pub mod diffusion_matrices;
pub mod ordered_thresholds;

/// Quantise an image to binary (black and white) using error diffusion.
pub fn error_diffusion_quantise(
    img: &DynamicImage, matrix: &dyn diffusion_matrices::DiffusionMatrix, serpentine: bool,
) -> GrayImage {
    let mut img = img.to_luma32f();
    let mut reversed = false;
    for y in 0..img.height() {
        for x in 0..img.width() {
            let x = if reversed { img.width() - x - 1 } else { x };
            let old_pixel = img.get_pixel(x, y).0[0];
            let new_pixel = if old_pixel < 0.5 { 0.0 } else { 1.0 };
            let error = old_pixel - new_pixel;
            img.put_pixel(x, y, Luma([new_pixel]));
            for (dx, dy, weight) in matrix.enumerate() {
                let nx = x.wrapping_add_signed(if reversed { -dx } else { dx });
                let ny = y + dy;
                if !img.in_bounds(nx, ny) {
                    continue;
                }
                let pixel = img.get_pixel(nx, ny).0[0];
                let new_pixel = pixel + error * weight;
                img.put_pixel(nx, ny, Luma([new_pixel]));
            }
        }
        reversed = serpentine && !reversed;
    }
    DynamicImage::from(img).to_luma8()
}

/// Quantise an image to binary (black and white) using ordered dithering.
pub fn ordered_dither_quantise(
    img: &DynamicImage, thresh: &dyn ordered_thresholds::OrderedThreshold,
) -> GrayImage {
    let mut img = img.to_luma32f();
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        pixel.0[0] = if pixel.0[0] < thresh.get(x, y) { 0.0 } else { 1.0 };
    }
    DynamicImage::from(img).to_luma8()
}
