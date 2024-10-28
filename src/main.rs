use image::codecs::pnm::PnmSubtype::Bitmap;
use image::codecs::pnm::SampleEncoding::Binary;
use image::codecs::pnm::PnmEncoder;
use image::ExtendedColorType::L8;
use image::{EncodableLayout, ImageEncoder, ImageReader};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Load image
    let in_img = ImageReader::open("input/lena.ppm")?.decode()?;

    // Threshold quantisation and save
    let bin_img = dither::threshold_quantise(&in_img);
    PnmEncoder::new(std::fs::File::create("output/lena_threshold.pbm")?)
        .with_subtype(Bitmap(Binary))
        .write_image(bin_img.as_bytes(), bin_img.width(), bin_img.height(), L8)?;

    // Random dithering and save
    let rand_img = dither::random_dither_quantise(&in_img);
    PnmEncoder::new(std::fs::File::create("output/lena_random_dither.pbm")?)
        .with_subtype(Bitmap(Binary))
        .write_image(rand_img.as_bytes(), rand_img.width(), rand_img.height(), L8)?;

    // IGN threshold and save
    let ign_img = dither::ign_quantise(&in_img);
    PnmEncoder::new(std::fs::File::create("output/lena_ign.pbm")?)
        .with_subtype(Bitmap(Binary))
        .write_image(ign_img.as_bytes(), ign_img.width(), ign_img.height(), L8)?;

    // Ordered dithering and save
    let threshold_matrices: [&str; 15] = [
        "bayer/2x2", "bayer/4x4", "bayer/8x8", "bayer/16x16", "bayer/32x32", "bayer/64x64",
        "bayer/128x128", "bayer/256x256", "blue_noise/16x16/16bpc_0", "blue_noise/32x32/16bpc_0",
        "blue_noise/64x64/16bpc_0", "blue_noise/128x128/16bpc_0", "blue_noise/256x256/16bpc_0",
        "blue_noise/512x512/8bpc_0", "blue_noise/1024x1024/8bpc_0"
    ];
    for matrix in threshold_matrices {
        let matrix_img = image::open(format!("textures/{matrix}.png"))?.into_luma16();
        let matrix_img = dither::ordered_dither_quantise(&in_img, &matrix_img);
        let matrix = matrix.replace("/", "_");
        PnmEncoder::new(std::fs::File::create(format!("output/lena_{matrix}.pbm"))?)
            .with_subtype(Bitmap(Binary))
            .write_image(matrix_img.as_bytes(), matrix_img.width(), matrix_img.height(), L8)?;
    }

    // Error diffusion and save
    let diffusion_matrices: [(&str, &dyn dither::DiffusionMatrix); 8] = [
        ("atkinson", &dither::ATKINSON),
        ("burkes", &dither::BURKES),
        ("floyd_steinberg", &dither::FLOYD_STEINBERG),
        ("jarvis_judice_ninke", &dither::JARVIS_JUDICE_NINKE),
        ("sierra", &dither::SIERRA),
        ("sierra_two_row", &dither::SIERRA_TWO_ROW),
        ("sierra_lite", &dither::SIERRA_LITE),
        ("stucki", &dither::STUCKI)
    ];
    for (name, matrix) in diffusion_matrices {
        let err_img = dither::error_diffusion_quantise(&in_img, matrix);
        PnmEncoder::new(std::fs::File::create(format!("output/lena_{}.pbm", name))?)
            .with_subtype(Bitmap(Binary))
            .write_image(err_img.as_bytes(), err_img.width(), err_img.height(), L8)?;
    }

    // Success
    Ok(())
}