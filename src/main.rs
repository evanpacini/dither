use dither::{diffusion_matrices, ordered_thresholds};
use image::codecs::pnm::PnmEncoder;
use image::codecs::pnm::PnmSubtype::Bitmap;
use image::codecs::pnm::SampleEncoding::Binary;
use image::ExtendedColorType::L8;
use image::{EncodableLayout, ImageEncoder, ImageReader};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Load image
    let in_img = ImageReader::open("input/lena.ppm")?.decode()?;

    // Error diffusion and save
    let diffusion_matrices: [(&str, &dyn diffusion_matrices::DiffusionMatrix); 10] = [
        ("atkinson", &diffusion_matrices::ATKINSON),
        ("burkes", &diffusion_matrices::BURKES),
        ("floyd_steinberg", &diffusion_matrices::FLOYD_STEINBERG),
        ("jarvis_judice_ninke", &diffusion_matrices::JARVIS_JUDICE_NINKE),
        ("pigeon", &diffusion_matrices::PIGEON),
        ("shiau_fan", &diffusion_matrices::SHIAU_FAN),
        ("sierra", &diffusion_matrices::SIERRA),
        ("sierra_two_row", &diffusion_matrices::SIERRA_TWO_ROW),
        ("sierra_lite", &diffusion_matrices::SIERRA_LITE),
        ("stucki", &diffusion_matrices::STUCKI),
    ];
    for (name, matrix) in diffusion_matrices {
        let err_img = dither::error_diffusion_quantise(&in_img, matrix);
        PnmEncoder::new(std::fs::File::create(format!("output/lena_ed_{}.pbm", name))?)
            .with_subtype(Bitmap(Binary))
            .write_image(err_img.as_bytes(), err_img.width(), err_img.height(), L8)?;
    }

    // Ordered dithering and save
    let ordered_thresholds: [(&str, &dyn ordered_thresholds::OrderedThreshold); 18] = [
        ("flat", &ordered_thresholds::FLAT),
        ("random", &ordered_thresholds::RANDOM),
        ("ign", &ordered_thresholds::IGN),
        ("bayer_2", &image::open("textures/bayer/2x2.png")?.to_luma32f()),
        ("bayer_4", &image::open("textures/bayer/4x4.png")?.to_luma32f()),
        ("bayer_8", &image::open("textures/bayer/8x8.png")?.to_luma32f()),
        ("bayer_16", &image::open("textures/bayer/16x16.png")?.to_luma32f()),
        ("bayer_32", &image::open("textures/bayer/32x32.png")?.to_luma32f()),
        ("bayer_64", &image::open("textures/bayer/64x64.png")?.to_luma32f()),
        ("bayer_128", &image::open("textures/bayer/128x128.png")?.to_luma32f()),
        ("bayer_256", &image::open("textures/bayer/256x256.png")?.to_luma32f()),
        ("blue_noise_16", &image::open("textures/blue_noise/16x16/16bpc_0.png")?.to_luma32f()),
        ("blue_noise_32", &image::open("textures/blue_noise/32x32/16bpc_0.png")?.to_luma32f()),
        ("blue_noise_64", &image::open("textures/blue_noise/64x64/16bpc_0.png")?.to_luma32f()),
        ("blue_noise_128", &image::open("textures/blue_noise/128x128/16bpc_0.png")?.to_luma32f()),
        ("blue_noise_256", &image::open("textures/blue_noise/256x256/16bpc_0.png")?.to_luma32f()),
        ("blue_noise_512", &image::open("textures/blue_noise/512x512/8bpc_0.png")?.to_luma32f()),
        ("blue_noise_1024", &image::open("textures/blue_noise/1024x1024/8bpc_0.png")?.to_luma32f()),
    ];
    for (name, thresh) in ordered_thresholds {
        let res = dither::ordered_dither_quantise(&in_img, thresh);
        PnmEncoder::new(std::fs::File::create(format!("output/lena_{name}.pbm"))?)
            .with_subtype(Bitmap(Binary))
            .write_image(res.as_bytes(), res.width(), res.height(), L8)?;
    }

    // Success
    Ok(())
}
