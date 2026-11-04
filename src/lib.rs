//! # kaptchaocr — CAPTCHA OCR
//!
//! Library + API server. This file is the library core; see the [`manifest`] module for the model manifest.

pub mod click;
pub mod manifest;
pub mod math;
pub mod trajectory;

/// Initialize content recognition with the default fine-tuned model (`ft-small`).
#[cfg(feature = "inline-model")]
pub fn kaptchaocr_classification() -> anyhow::Result<KaptchaOcr<'static>> {
    KaptchaOcr::new(
        include_bytes!("../model/finetuned/small_rec.onnx"),
        serde_json::from_str(include_str!("../model/finetuned/small_rec.json")).unwrap(),
    )
}

/// Initialize content recognition and set the ORT `IntraOpNumThreads`.
#[cfg(feature = "inline-model")]
pub fn kaptchaocr_classification_with_num_threads(
    num_threads: usize,
) -> anyhow::Result<KaptchaOcr<'static>> {
    KaptchaOcr::new_with_num_threads(
        include_bytes!("../model/finetuned/small_rec.onnx"),
        serde_json::from_str(include_str!("../model/finetuned/small_rec.json")).unwrap(),
        num_threads,
    )
}

/// Initialize content recognition with the default fine-tuned model.
#[cfg(not(feature = "inline-model"))]
pub fn kaptchaocr_classification() -> anyhow::Result<KaptchaOcr<'static>> {
    KaptchaOcr::new(
        std::fs::read("model/finetuned/small_rec.onnx")?,
        serde_json::from_str(&std::fs::read_to_string("model/finetuned/small_rec.json")?)?,
    )
}

/// Initialize content recognition with the default fine-tuned model on CUDA.
#[cfg(feature = "cuda")]
pub fn kaptchaocr_classification_cuda(device_id: i32) -> anyhow::Result<KaptchaOcr<'static>> {
    KaptchaOcr::new_cuda(
        include_bytes!("../model/finetuned/small_rec.onnx"),
        serde_json::from_str(include_str!("../model/finetuned/small_rec.json")).unwrap(),
        device_id,
    )
}

/// Initialize object detection with the default fine-tuned detector (`ft-small-det`).
#[cfg(feature = "inline-model")]
pub fn kaptchaocr_detection() -> anyhow::Result<KaptchaOcr<'static>> {
    let mut model = KaptchaOcr::new_model(include_bytes!("../model/finetuned/small_det.onnx"))?;
    model.set_det_type(DetType::Db);
    Ok(model)
}

/// Initialize object detection and set the ORT `IntraOpNumThreads`.
#[cfg(feature = "inline-model")]
pub fn kaptchaocr_detection_with_num_threads(
    num_threads: usize,
) -> anyhow::Result<KaptchaOcr<'static>> {
    let mut model = KaptchaOcr::new_model_with_num_threads(
        include_bytes!("../model/finetuned/small_det.onnx"),
        num_threads,
    )?;
    model.set_det_type(DetType::Db);
    Ok(model)
}

/// Initialize object detection with the default fine-tuned detector.
#[cfg(not(feature = "inline-model"))]
pub fn kaptchaocr_detection() -> anyhow::Result<KaptchaOcr<'static>> {
    let mut model = KaptchaOcr::new_model(std::fs::read("model/finetuned/small_det.onnx")?)?;
    model.set_det_type(DetType::Db);
    Ok(model)
}

/// Initialize object detection with the default fine-tuned detector on CUDA.
#[cfg(feature = "cuda")]
pub fn kaptchaocr_detection_cuda(device_id: i32) -> anyhow::Result<KaptchaOcr<'static>> {
    let mut model = KaptchaOcr::new_model_cuda(
        include_bytes!("../model/finetuned/small_det.onnx"),
        device_id,
    )?;
    model.set_det_type(DetType::Db);
    Ok(model)
}

/// Slider matching.
pub fn slide_match<I1, I2>(target_image: I1, background_image: I2) -> anyhow::Result<SlideBBox>
where
    I1: AsRef<[u8]>,
    I2: AsRef<[u8]>,
{
    let target_image = image::load_from_memory(target_image.as_ref())?;
    let background_image = image::load_from_memory(background_image.as_ref())?;

    anyhow::ensure!(
        background_image.width() >= target_image.width(),
        "the background image width must be greater than or equal to the target image width"
    );

    anyhow::ensure!(
        background_image.height() >= target_image.height(),
        "the background image height must be greater than or equal to the target image height"
    );

    let target_image = target_image.to_rgba8();

    // crop the image, keeping only the opaque part
    let width = target_image.width();
    let height = target_image.height();
    let mut start_x = width;
    let mut start_y = height;
    let mut end_x = 0;
    let mut end_y = 0;

    for x in 0..width {
        for y in 0..height {
            let p = target_image[(x, y)];

            if p[3] != 0 {
                if x < start_x {
                    start_x = x;
                }

                if y < start_y {
                    start_y = y;
                }

                if x > end_x {
                    end_x = x;
                }

                if y > end_y {
                    end_y = y;
                }
            }
        }
    }

    let cropped_image = if start_x > end_x || start_y > end_y {
        // no opaque pixels at all
        target_image
    } else {
        image::imageops::crop_imm(
            &target_image,
            start_x,
            start_y,
            end_x - start_x + 1,
            end_y - start_y + 1,
        )
        .to_image()
    };

    // convert the image to grayscale
    let target_image = image::imageops::grayscale(&cropped_image);

    // Run canny edge detection, then do the same for the background image.
    // Use match_template for template matching to get the result matrix;
    // then use find_extremes to find the max and min values in the matrix,
    // and take the location `loc` of the max value to compute the target position from the target image size and `loc`
    let target_image = imageproc::edges::canny(&target_image, 100.0, 200.0);
    let background_image = imageproc::edges::canny(&background_image.to_luma8(), 100.0, 200.0);
    let result =
        imageproc::template_matching::find_extremes(&imageproc::template_matching::match_template(
            &background_image,
            &target_image,
            imageproc::template_matching::MatchTemplateMethod::CrossCorrelationNormalized,
        ));

    Ok(SlideBBox {
        target_x: start_x,
        target_y: start_y,
        x1: result.max_value_location.0,
        y1: result.max_value_location.1,
        x2: result.max_value_location.0 + target_image.width(),
        y2: result.max_value_location.1 + target_image.height(),
    })
}

/// Slider matching.
pub fn slide_match_with_path<P1, P2>(
    target_image: P1,
    background_image: P2,
) -> anyhow::Result<SlideBBox>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    slide_match(
        std::fs::read(target_image)?,
        std::fs::read(background_image)?,
    )
}

/// If the small image has little background, simple slider matching can be used.
pub fn simple_slide_match<I1, I2>(
    target_image: I1,
    background_image: I2,
) -> anyhow::Result<SlideBBox>
where
    I1: AsRef<[u8]>,
    I2: AsRef<[u8]>,
{
    let target_image = image::load_from_memory(target_image.as_ref())?;
    let background_image = image::load_from_memory(background_image.as_ref())?;

    anyhow::ensure!(
        background_image.width() >= target_image.width(),
        "the background image width must be greater than or equal to the target icon width"
    );

    anyhow::ensure!(
        background_image.height() >= target_image.height(),
        "the background image height must be greater than or equal to the target icon height"
    );

    // Run canny edge detection, then do the same for the background image.
    // Use match_template for template matching to get the result matrix;
    // then use find_extremes to find the max and min values in the matrix,
    // and take the location `loc` of the max value to compute the target position from the target image size and `loc`
    let target_image = imageproc::edges::canny(&target_image.to_luma8(), 100.0, 200.0);
    let background_image = imageproc::edges::canny(&background_image.to_luma8(), 100.0, 200.0);
    let result =
        imageproc::template_matching::find_extremes(&imageproc::template_matching::match_template(
            &background_image,
            &target_image,
            imageproc::template_matching::MatchTemplateMethod::CrossCorrelationNormalized,
        ));

    Ok(SlideBBox {
        target_x: 0,
        target_y: 0,
        x1: result.max_value_location.0,
        y1: result.max_value_location.1,
        x2: result.max_value_location.0 + target_image.width(),
        y2: result.max_value_location.1 + target_image.height(),
    })
}

/// If the small image has little background, simple slider matching can be used.
pub fn simple_slide_match_with_path<P1, P2>(
    target_image: P1,
    background_image: P2,
) -> anyhow::Result<SlideBBox>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    simple_slide_match(
        std::fs::read(target_image)?,
        std::fs::read(background_image)?,
    )
}

/// Slot matching.
pub fn slide_comparison<I1, I2>(
    target_image: I1,
    background_image: I2,
) -> anyhow::Result<(u32, u32)>
where
    I1: AsRef<[u8]>,
    I2: AsRef<[u8]>,
{
    let target_image = image::load_from_memory(target_image.as_ref())?;
    let background_image = image::load_from_memory(background_image.as_ref())?;

    anyhow::ensure!(
        target_image.width() == background_image.width()
            && target_image.height() == background_image.height(),
        "image dimensions are not equal"
    );

    let image = image::RgbImage::from_vec(
        target_image.width(),
        target_image.height(),
        target_image
            .as_bytes()
            .iter()
            .zip(background_image.as_bytes().iter())
            .map(|(a, b)| if a.abs_diff(*b) > 80 { 255 } else { 0 })
            .collect(),
    )
    .unwrap();

    let mut start_x = 0;
    let mut start_y = 0;

    for i in 0..image.width() {
        let mut count = 0;

        for j in 0..image.height() {
            let pixel = image[(i, j)];

            if pixel != image::Rgb([0, 0, 0]) {
                count += 1;
            }

            if count >= 5 && start_y == 0 {
                start_y = j.saturating_sub(5);
            }
        }

        if count >= 5 {
            start_x = i + 2;
            break;
        }
    }

    Ok((start_x, start_y))
}

/// Slot matching.
pub fn slide_comparison_with_path<P1, P2>(
    target_image: P1,
    background_image: P2,
) -> anyhow::Result<(u32, u32)>
where
    P1: AsRef<std::path::Path>,
    P2: AsRef<std::path::Path>,
{
    slide_comparison(
        std::fs::read(target_image)?,
        std::fs::read(background_image)?,
    )
}

/// Determine whether the model is a custom (DIY) model.
pub fn is_diy<MODEL>(model: MODEL) -> bool
where
    MODEL: AsRef<[u8]>,
{
    // SHA-256 of the bundled fine-tuned models.
    let sha256 = sha256::digest(model.as_ref());

    sha256 != "1421f20a9f67a124b88bb57c6a797de251a68282933689143e743b96e2d07cc7"
        && sha256 != "45625442b7083d0cd96abb229ac7d3ec9182cbcffb5629cf6105f55f580ead04"
        && sha256 != "3dcf9aa37c1e8fd8c7c4e909d55e242577f4180963c08ef5787b9c7958f1aa8a"
        && sha256 != "136eb567f409b1141b3ff8a1da1e175730a790c9736cac879e8ba36ddbe5cdcd"
        && sha256 != "be9b9d84073d689b0682813e0327ffd228776aa6e8551b270d429427eaad2935"
}

/// Scan rotation angles in -15°..=15°, using horizontal-projection variance to measure "how upright" the text is; returns the best rotation angle (radians).
fn find_deskew_angle(image: &image::GrayImage) -> f32 {
    let mut best_angle = 0.0f32;
    let mut best_score = f32::MIN;

    for deg in -15..=15 {
        let theta = (deg as f32).to_radians();

        let rotated = imageproc::geometric_transformations::rotate_about_center(
            image,
            theta,
            imageproc::geometric_transformations::Interpolation::Bilinear,
            image::Luma([255]),
        );

        // horizontal projection: mean grayscale per row; larger variance → stronger "line structure" of the text rows
        let (w, h) = rotated.dimensions();
        let mut sums = Vec::with_capacity(h as usize);

        for y in 0..h {
            let mut row_sum = 0.0f32;
            for x in 0..w {
                row_sum += rotated.get_pixel(x, y)[0] as f32;
            }
            sums.push(row_sum / w as f32);
        }

        let mean = sums.iter().sum::<f32>() / sums.len() as f32;
        let variance =
            sums.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / sums.len() as f32;

        if variance > best_score {
            best_score = variance;
            best_angle = theta;
        }
    }

    best_angle
}

/// Fill the transparent parts of the image with white.
fn png_rgba_black_preprocess(image: &image::DynamicImage) -> image::DynamicImage {
    let (width, height) = image::GenericImageView::dimensions(image);
    let mut new_image = image::DynamicImage::new_rgb8(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgba_pixel = image::GenericImageView::get_pixel(image, x, y);

            let rgb_pixel = if rgba_pixel[3] == 0 {
                image::Rgba([255, 255, 255, 255])
            } else {
                image::Rgba([rgba_pixel[0], rgba_pixel[1], rgba_pixel[2], rgba_pixel[3]])
            };

            image::GenericImage::put_pixel(&mut new_image, x, y, rgb_pixel);
        }
    }

    new_image
}

/// Configuration needed for content recognition.
///
/// Example charset configs: `../model/finetuned/small_rec.json`,
/// `../model/finetuned/medium_rec.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Charset {
    /// Whether this is a cnn model.
    pub word: bool,

    /// Width and height; if width is -1 it is adjusted automatically; height must be a multiple of 16.
    pub image: [i64; 2],

    /// Number of channels.
    pub channel: i64,

    /// Charset.
    pub charset: Vec<String>,

    /// Optional normalization mean (PaddleOCR style), e.g. `[0.5, 0.5, 0.5]`.
    /// When provided, normalize per channel as `(x/255 - mean)/std`; otherwise the official / DIY defaults are used.
    #[serde(default)]
    pub mean: Option<[f32; 3]>,

    /// Optional normalization standard deviation, used together with `mean`.
    #[serde(default)]
    pub std: Option<[f32; 3]>,

    /// Whether the channel order is BGR (PaddleOCR uses BGR by default).
    #[serde(default)]
    pub bgr: Option<bool>,
}

impl std::str::FromStr for Charset {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

/// Preprocessing options; all disabled by default (preserving the original behavior).
///
/// These options act on the original image before resizing / channel conversion, and suit noisy CAPTCHAs with complex backgrounds.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreprocessOptions {
    /// OTSU adaptive threshold binarization (converted to grayscale first).
    pub binarize: bool,

    /// Median filter denoising (3x3 neighborhood).
    pub denoise: bool,

    /// Automatic deskew via horizontal projection (scans -15°..=15° for the best angle and rotates it upright).
    pub deskew: bool,

    /// Resize strategy; `ResizeStrategy::Auto` (default) follows the model's `Charset::image` config.
    pub resize: ResizeStrategy,
}

/// Resize strategy.
#[derive(Debug, Clone, Copy, Default)]
pub enum ResizeStrategy {
    /// Use the size from the model charset config (default, preserving the original behavior).
    #[default]
    Auto,

    /// Force resize to the given `(width, height)`.
    Exact(u32, u32),
}

/// Text coordinates.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BBox {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

/// Slider coordinates.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SlideBBox {
    /// x offset of the transparent part.
    pub target_x: u32,

    /// y offset of the transparent part.
    pub target_y: u32,

    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

/// Charset and probabilities.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CharacterProbability {
    pub text: Option<String>,
    pub charset: Vec<String>,
    pub probability: Vec<Vec<f32>>,
    pub confidence: Option<f64>,
}

/// Sometimes you only want `probability`, not `text` and `confidence`.
impl CharacterProbability {
    pub fn get_text(&mut self) -> &str {
        self.text.get_or_insert_with(|| {
            let mut s = String::new();

            for i in &self.probability {
                let (n, _) = i
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap();

                s += &self.charset[n];
            }

            s
        })
    }

    pub fn get_confidence(&mut self) -> f64 {
        *self.confidence.get_or_insert_with(|| {
            let mut max_sum = 0.0;
            let mut count = 0usize;

            for i in &self.probability {
                if let Some(v) = i.iter().fold(None, |acc: Option<f64>, &i| {
                    acc.map_or(Some(i as f64), |max| Some(max.max(i as f64)))
                }) {
                    max_sum += v;
                    count += 1;
                }
            }

            if count == 0 {
                0.0
            } else {
                max_sum / count as f64
            }
        })
    }
}

pub trait MapJson {
    fn json(&self) -> String;
}

impl MapJson for BBox {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for Vec<BBox> {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for SlideBBox {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for (u32, u32) {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for CharacterProbability {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl MapJson for Vec<(BBox, String)> {
    fn json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

pub trait MapBBox {
    fn to_tuple(&self) -> Vec<(u32, u32, u32, u32)>;

    fn to_vec(&self) -> Vec<Vec<u32>>;
}

impl MapBBox for Vec<BBox> {
    fn to_tuple(&self) -> Vec<(u32, u32, u32, u32)> {
        self.iter()
            .map(|v| (v.x1, v.y1, v.x2, v.y2))
            .collect::<Vec<_>>()
    }

    fn to_vec(&self) -> Vec<Vec<u32>> {
        self.iter()
            .map(|v| vec![v.x1, v.y1, v.x2, v.y2])
            .collect::<Vec<_>>()
    }
}

const NMS_THR: f32 = 0.45;
const SCORE_THR: f32 = 0.1;
const MODEL_WIDTH: u32 = 416;
const MODEL_HEIGHT: u32 = 416;
const STRIDES: [u32; 3] = [8, 16, 32];

/// Output decoding type of the object detection model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetType {
    /// Anchor-grid decoding (YOLOv5 style) for custom detectors.
    #[default]
    Yolo,
    /// DB (Differentiable Binarization) probability-map decoding (PaddleOCR family).
    Db,
}

/// Post-processing parameters for the DB detector (PaddleOCR `DBPostProcess` defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DbParams {
    /// Probability-map binarization threshold.
    pub thresh: f32,
    /// Minimum average score for a box to be kept.
    pub box_thresh: f32,
    /// Text-box expansion ratio.
    pub unclip_ratio: f32,
    /// Maximum number of candidate boxes kept.
    pub max_candidates: usize,
}

impl Default for DbParams {
    fn default() -> Self {
        Self {
            thresh: 0.2,
            box_thresh: 0.4,
            unclip_ratio: 1.4,
            max_candidates: 3000,
        }
    }
}

/// Charset range.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CharsetRange {
    /// Digits 0-9 only.
    Digit,

    /// Lowercase letters a-z only.
    Lowercase,

    /// Uppercase letters A-Z only.
    Uppercase,

    /// Lowercase a-z + uppercase A-Z.
    LowercaseUppercase,

    /// Lowercase a-z + digits 0-9.
    LowercaseDigit,

    /// Uppercase A-Z + digits 0-9.
    UppercaseDigit,

    /// Lowercase a-z + uppercase A-Z + digits 0-9.
    LowercaseUppercaseDigit,

    /// Default charset with lowercase a-z, uppercase A-Z and digits 0-9 removed.
    DefaultCharsetLowercaseUppercaseDigit,

    /// Custom charset, e.g. `0123456789+-x/=`.
    Other(String),

    /// Set the charset directly, i.e. the result of set_ranges.
    Charset(Vec<String>),
}

impl From<i32> for CharsetRange {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Digit,
            1 => Self::Lowercase,
            2 => Self::Uppercase,
            3 => Self::LowercaseUppercase,
            4 => Self::LowercaseDigit,
            5 => Self::UppercaseDigit,
            6 => Self::LowercaseUppercaseDigit,
            7 => Self::DefaultCharsetLowercaseUppercaseDigit,
            // fall back to a literal charset when out of range instead of panicking
            n => Self::Other(n.to_string()),
        }
    }
}

impl From<&str> for CharsetRange {
    fn from(value: &str) -> Self {
        CharsetRange::Other(value.to_string())
    }
}

impl From<String> for CharsetRange {
    fn from(value: String) -> Self {
        CharsetRange::Other(value)
    }
}

impl From<&String> for CharsetRange {
    fn from(value: &String) -> Self {
        CharsetRange::Other(value.clone())
    }
}

impl From<Vec<String>> for CharsetRange {
    fn from(value: Vec<String>) -> Self {
        CharsetRange::Charset(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Color {
    /// Red.
    Red,

    /// Blue.
    Blue,

    /// Green.
    Green,

    /// Yellow.
    Yellow,

    /// Orange.
    Orange,

    /// Purple.
    Purple,

    /// Cyan.
    Cyan,

    /// Black.
    Black,

    /// White.
    White,

    /// Gray.
    Gray,
}

impl<T: AsRef<str>> From<T> for Color {
    fn from(value: T) -> Self {
        match value.as_ref().to_ascii_lowercase().as_str() {
            "red" => Color::Red,
            "blue" => Color::Blue,
            "green" => Color::Green,
            "yellow" => Color::Yellow,
            "orange" => Color::Orange,
            "purple" => Color::Purple,
            "cyan" => Color::Cyan,
            "black" => Color::Black,
            "white" => Color::White,
            "gray" => Color::Gray,
            _ => panic!("unknown color: {}", value.as_ref()),
        }
    }
}

/// HSV triple, e.g. `(0, 50, 50)`.
pub type HsvValue = (u8, u8, u8);

/// An HSV range `(min_hsv, max_hsv)`.
pub type HsvRange = (HsvValue, HsvValue);

/// List of HSV ranges.
pub type HsvRanges = Vec<HsvRange>;

pub trait IntoHsvRange {
    fn into_hsv_ranges(self) -> HsvRanges;
}

impl IntoHsvRange for Color {
    fn into_hsv_ranges(self) -> HsvRanges {
        match self {
            Color::Red => vec![
                ((0, 50, 50), (10, 255, 255)),
                ((170, 50, 50), (180, 255, 255)),
            ],
            Color::Blue => vec![((100, 50, 50), (140, 255, 255))],
            Color::Green => vec![((40, 50, 50), (80, 255, 255))],
            Color::Yellow => vec![((20, 50, 50), (40, 255, 255))],
            Color::Orange => vec![((10, 50, 50), (20, 255, 255))],
            Color::Purple => vec![((140, 50, 50), (170, 255, 255))],
            Color::Cyan => vec![((80, 50, 50), (100, 255, 255))],
            Color::Black => vec![((0, 0, 0), (180, 255, 30))],
            Color::White => vec![((0, 0, 200), (180, 30, 255))],
            Color::Gray => vec![((0, 0, 30), (180, 30, 200))],
        }
    }
}

/// Color filtering.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ColorFilter {
    /// HSV ranges; each element is a (min_hsv, max_hsv) tuple.
    /// e.g. `[((0, 50, 50), (10, 255, 255))]`.
    HSVRanges(HsvRanges),

    /// Color list, e.g. `["red", "blue"]`.
    ColorRanges(Vec<Color>),

    /// A single color.
    Color(Color),
}

impl IntoHsvRange for ColorFilter {
    fn into_hsv_ranges(self) -> HsvRanges {
        match self {
            ColorFilter::HSVRanges(v) => v,
            ColorFilter::ColorRanges(v) => ColorFilter::from(v).into_hsv_ranges(),
            ColorFilter::Color(v) => v.into_hsv_ranges(),
        }
    }
}

impl IntoHsvRange for &str {
    fn into_hsv_ranges(self) -> HsvRanges {
        Color::from(self).into_hsv_ranges()
    }
}

impl IntoHsvRange for String {
    fn into_hsv_ranges(self) -> HsvRanges {
        Color::from(self).into_hsv_ranges()
    }
}

impl IntoHsvRange for &String {
    fn into_hsv_ranges(self) -> HsvRanges {
        Color::from(self).into_hsv_ranges()
    }
}

impl IntoHsvRange for HsvRange {
    fn into_hsv_ranges(self) -> HsvRanges {
        vec![self]
    }
}

impl From<&str> for ColorFilter {
    fn from(value: &str) -> Self {
        Color::from(value).into_hsv_ranges().into()
    }
}

impl From<String> for ColorFilter {
    fn from(value: String) -> Self {
        value.as_str().into()
    }
}

impl From<&String> for ColorFilter {
    fn from(value: &String) -> Self {
        value.as_str().into()
    }
}

impl From<Color> for ColorFilter {
    fn from(value: Color) -> Self {
        value.into_hsv_ranges().into()
    }
}

impl From<&[Color]> for ColorFilter {
    fn from(value: &[Color]) -> Self {
        value.to_vec().into()
    }
}

impl From<&[&str]> for ColorFilter {
    fn from(value: &[&str]) -> Self {
        value
            .iter()
            .map(|v| Color::from(*v))
            .collect::<Vec<_>>()
            .into()
    }
}

impl<T, const N: usize> From<[T; N]> for ColorFilter
where
    T: IntoHsvRange,
{
    fn from(value: [T; N]) -> Self {
        value
            .into_iter()
            .flat_map(|v| v.into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for ColorFilter
where
    T: IntoHsvRange + Clone,
{
    fn from(value: &'a [T; N]) -> Self {
        value
            .iter()
            .flat_map(|v| v.clone().into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<&[String]> for ColorFilter {
    fn from(values: &[String]) -> Self {
        values.to_vec().into()
    }
}

impl From<Vec<&str>> for ColorFilter {
    fn from(value: Vec<&str>) -> Self {
        value
            .into_iter()
            .flat_map(|v| Color::from(v).into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<Vec<String>> for ColorFilter {
    fn from(value: Vec<String>) -> Self {
        value
            .iter()
            .flat_map(|v| Color::from(v.as_str()).into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<Vec<Color>> for ColorFilter {
    fn from(value: Vec<Color>) -> Self {
        value
            .into_iter()
            .flat_map(|v| v.into_hsv_ranges())
            .collect::<Vec<_>>()
            .into()
    }
}

impl From<HsvRanges> for ColorFilter {
    fn from(value: HsvRanges) -> Self {
        ColorFilter::HSVRanges(value)
    }
}

impl ColorFilter {
    /// Filter by color; e.g. ColorFilter::from("green").filter(image) keeps only green.
    pub fn filter<I>(&self, image: I) -> anyhow::Result<image::DynamicImage>
    where
        I: AsRef<[u8]>,
    {
        let image = image::load_from_memory(image.as_ref())?.to_rgb8();
        let (width, height) = image.dimensions();
        let mut array = ndarray::Array3::<u8>::zeros((height as usize, width as usize, 3));

        for (x, y, pixel) in image.enumerate_pixels() {
            let y = y as usize;
            let x = x as usize;
            array[[y, x, 0]] = pixel[0];
            array[[y, x, 1]] = pixel[1];
            array[[y, x, 2]] = pixel[2];
        }

        let mut hsv = ndarray::Array3::<u8>::zeros((height as usize, width as usize, 3));

        for y in 0..height as usize {
            for x in 0..width as usize {
                let r = array[[y, x, 0]] as f32 / 255.0;
                let g = array[[y, x, 1]] as f32 / 255.0;
                let b = array[[y, x, 2]] as f32 / 255.0;
                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                let v = max;
                let delta = max - min;
                let s = if max == 0.0 { 0.0 } else { delta / max };
                let h_deg = if delta == 0.0 {
                    0.0
                } else if max == r {
                    60.0 * (((g - b) / delta) % 6.0)
                } else if max == g {
                    60.0 * (((b - r) / delta) + 2.0)
                } else {
                    60.0 * (((r - g) / delta) + 4.0)
                };

                let h_deg = if h_deg < 0.0 { h_deg + 360.0 } else { h_deg };

                hsv[[y, x, 0]] = (h_deg / 2.0).round().min(180.0) as u8;
                hsv[[y, x, 1]] = (s * 255.0).round().min(255.0) as u8;
                hsv[[y, x, 2]] = (v * 255.0).round().min(255.0) as u8;
            }
        }

        let mut mask = ndarray::Array2::<bool>::from_elem((height as usize, width as usize), false);

        let ranges = match self {
            ColorFilter::HSVRanges(v) => v,
            ColorFilter::ColorRanges(v) => &v
                .iter()
                .flat_map(|v| v.into_hsv_ranges())
                .collect::<Vec<_>>(),
            ColorFilter::Color(v) => &v.into_hsv_ranges(),
        };

        for (lower, upper) in ranges {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let h = hsv[[y, x, 0]];
                    let s = hsv[[y, x, 1]];
                    let v = hsv[[y, x, 2]];

                    if h >= lower.0
                        && h <= upper.0
                        && s >= lower.1
                        && s <= upper.1
                        && v >= lower.2
                        && v <= upper.2
                    {
                        mask[[y, x]] = true;
                    }
                }
            }
        }

        let mut result: image::RgbImage = image::ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let yi = y as usize;
                let xi = x as usize;
                let pixel = if mask[[yi, xi]] {
                    image::Rgb([array[[yi, xi, 0]], array[[yi, xi, 1]], array[[yi, xi, 2]]])
                } else {
                    image::Rgb([255, 255, 255])
                };

                result.put_pixel(x, y, pixel);
            }
        }

        Ok(image::DynamicImage::ImageRgb8(result))
    }
}

/// Determine whether the output of a word (CTC) model is a probability.
///
/// The CTC head exported by PaddleOCR already applies softmax in inference mode (the ONNX graph ends with `Softmax`,
/// and each frame sums to 1); custom exports may still be logits. Approximated by the sum of the first frame.
fn is_probability_output<S>(values: &ndarray::ArrayBase<S, ndarray::IxDyn>) -> bool
where
    S: ndarray::Data<Elem = f32>,
{
    let (timesteps, classes) = if values.ndim() == 3 {
        (values.shape()[1], values.shape()[2])
    } else {
        (values.shape()[0], values.shape()[1])
    };

    if timesteps == 0 || classes == 0 {
        return false;
    }

    let mut sum = 0.0f32;

    for c in 0..classes {
        sum += if values.ndim() == 3 {
            values[[0, 0, c]]
        } else {
            values[[0, c]]
        };
    }

    (sum - 1.0).abs() < 0.05
}

/// CTC greedy decoding (logits output): softmax → per-frame argmax → merge consecutive repeats → drop blank (class 0).
///
/// Returns `(class_idx, confidence)`, where confidence is the maximum softmax probability over the character's consecutive frames.
/// Supports both `[1, T, C]` and `[T, C]` output shapes (batch is treated as 1; content recognition processes one image at a time).
fn ctc_greedy_decode<S>(logits: &ndarray::ArrayBase<S, ndarray::IxDyn>) -> Vec<(usize, f32)>
where
    S: ndarray::Data<Elem = f32>,
{
    ctc_greedy_decode_impl(logits, false)
}

/// CTC greedy decoding (probability output, model already applies softmax): per-frame argmax; confidence is the frame probability directly.
fn ctc_greedy_decode_probability<S>(
    probability: &ndarray::ArrayBase<S, ndarray::IxDyn>,
) -> Vec<(usize, f32)>
where
    S: ndarray::Data<Elem = f32>,
{
    ctc_greedy_decode_impl(probability, true)
}

fn ctc_greedy_decode_impl<S>(
    values: &ndarray::ArrayBase<S, ndarray::IxDyn>,
    probability_output: bool,
) -> Vec<(usize, f32)>
where
    S: ndarray::Data<Elem = f32>,
{
    let (timesteps, classes) = if values.ndim() == 3 {
        (values.shape()[1], values.shape()[2])
    } else {
        (values.shape()[0], values.shape()[1])
    };

    let mut result: Vec<(usize, f32)> = Vec::new();
    let mut last_class: usize = 0;

    for t in 0..timesteps {
        let mut max_logit = f32::NEG_INFINITY;
        let mut argmax = 0usize;

        for c in 0..classes {
            let v = if values.ndim() == 3 {
                values[[0, t, c]]
            } else {
                values[[t, c]]
            };

            if v > max_logit {
                max_logit = v;
                argmax = c;
            }
        }

        // blank (class 0) separates two identical characters before and after it
        if argmax == 0 {
            last_class = 0;
            continue;
        }

        // confidence: logits output gets a full-table softmax; probability output uses the frame probability directly
        let prob = if probability_output {
            max_logit
        } else {
            // softmax probability (numerically stable: subtract the max before exp)
            let mut sum_exp = 0.0f32;
            for c in 0..classes {
                let v = if values.ndim() == 3 {
                    values[[0, t, c]]
                } else {
                    values[[t, c]]
                };
                sum_exp += (v - max_logit).exp();
            }

            1.0 / sum_exp
        };

        if argmax == last_class {
            // consecutive frames of the same character: confidence is the maximum over this run
            if let Some((_, conf)) = result.last_mut() {
                *conf = conf.max(prob);
            }
        } else {
            result.push((argmax, prob));
            last_class = argmax;
        }
    }

    result
}

/// Greedy merge for CTC index output (the model already provides a per-frame argmax): drop blank (class 0) and merge consecutive repeats.
/// When the model provides no probabilities, confidence is recorded as 1.0 uniformly.
fn ctc_greedy_decode_indices<S>(
    indices: &ndarray::ArrayBase<S, ndarray::IxDyn>,
) -> Vec<(usize, f32)>
where
    S: ndarray::Data<Elem = i64>,
{
    let mut result = Vec::new();
    let mut last_class = 0usize;

    for &v in indices.iter() {
        let v = v as usize;

        if v == 0 {
            last_class = 0;
        } else if v != last_class {
            result.push((v, 1.0));
            last_class = v;
        }
    }

    result
}

/// Map the requested charset range to model class indices.
///
/// For word (CTC) models blank is fixed at 0: an empty string maps to blank and other characters are looked up in the model charset;
/// requested characters not in the charset are recorded as `None` (probability output -1.0).
fn map_charset_range_indices(classes: &[String], ranges: &[String]) -> Vec<Option<usize>> {
    ranges
        .iter()
        .map(|v| {
            if v.is_empty() {
                Some(0)
            } else {
                classes.iter().position(|c| c == v)
            }
        })
        .collect()
}

/// Restricted CTC greedy decoding: each frame takes its argmax only over the allowed classes
/// (blank=0 is always eligible), then merges consecutive repeats and drops blank. Returns `(class_idx, confidence)`.
///
/// Unlike `ctc_greedy_decode`: the model's argmax over the full charset may fall outside the allowed set
/// (the `charset_range` case), so each frame's choice is constrained to `allowed`. Confidence is still
/// the probability under a full-table softmax for that frame.
fn ctc_greedy_decode_restricted<S>(
    logits: &ndarray::ArrayBase<S, ndarray::IxDyn>,
    allowed: &[usize],
) -> Vec<(usize, f32)>
where
    S: ndarray::Data<Elem = f32>,
{
    let (timesteps, classes) = if logits.ndim() == 3 {
        (logits.shape()[1], logits.shape()[2])
    } else {
        (logits.shape()[0], logits.shape()[1])
    };

    let mut result: Vec<(usize, f32)> = Vec::new();
    let mut last_class: usize = 0;

    for t in 0..timesteps {
        let logit = |c: usize| -> f32 {
            if logits.ndim() == 3 {
                logits[[0, t, c]]
            } else {
                logits[[t, c]]
            }
        };

        // blank always participates in the comparison
        let mut max_logit = f32::NEG_INFINITY;
        let mut argmax = 0usize;

        if classes > 0 {
            max_logit = logit(0);
        }

        for &c in allowed {
            if c == 0 || c >= classes {
                continue;
            }

            let v = logit(c);

            if v > max_logit {
                max_logit = v;
                argmax = c;
            }
        }

        // softmax probability (numerically stable: subtract the max before exp); the denominator is still the full charset
        let mut sum_exp = 0.0f32;
        for c in 0..classes {
            sum_exp += (logit(c) - max_logit).exp();
        }
        let prob = 1.0 / sum_exp;

        if argmax == 0 {
            last_class = 0;
            continue;
        }

        if argmax == last_class {
            // consecutive frames of the same character: confidence is the maximum over this run
            if let Some((_, conf)) = result.last_mut() {
                *conf = conf.max(prob);
            }
        } else {
            result.push((argmax, prob));
            last_class = argmax;
        }
    }

    result
}

/// Take the max at each position and average them as the overall confidence.
fn average_max_confidence(probability: &[Vec<f32>]) -> f64 {
    if probability.is_empty() {
        return 0.0;
    }

    let mut max_sum = 0.0f64;
    let mut count = 0usize;

    for i in probability {
        if let Some(v) = i.iter().fold(None, |acc: Option<f64>, &v| {
            acc.map_or(Some(v as f64), |max| Some(max.max(v as f64)))
        }) {
            max_sum += v;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        max_sum / count as f64
    }
}

#[derive(Debug)]
pub struct KaptchaOcr<'a> {
    diy: bool,
    session: ort::Session,
    charset: Option<std::borrow::Cow<'a, Charset>>,
    charset_range: Vec<String>,
    det_type: DetType,
    db_params: DbParams,
}

unsafe impl<'a> Send for KaptchaOcr<'a> {}
unsafe impl<'a> Sync for KaptchaOcr<'a> {}

/// Build an ORT session, optionally setting `IntraOpNumThreads`.
fn build_session<MODEL>(model: MODEL, num_threads: Option<usize>) -> anyhow::Result<ort::Session>
where
    MODEL: AsRef<[u8]>,
{
    let builder = ort::Session::builder()?;

    let builder = match num_threads {
        Some(n) => builder.with_intra_threads(n)?,
        None => builder,
    };

    Ok(builder.commit_from_memory(model.as_ref())?)
}

/// Bundled models and custom models take different parameters, so creating a model automatically detects whether it is a custom model.
impl<'a> KaptchaOcr<'a> {
    /// Set the runtime library path.
    ///
    /// - Unix: `/etc/.../libonnxruntime.so`
    /// - Windows: `C:\\Program Files\\...\\onnxruntime.dll`
    ///
    /// No need to call this function manually, as the program locates the runtime library path automatically.
    #[cfg(feature = "load-dynamic")]
    pub fn set_onnxruntime_path<P>(path: P) -> anyhow::Result<()>
    where
        P: AsRef<std::path::Path>,
    {
        let path = path.as_ref();

        let save = std::panic::take_hook();

        std::panic::set_hook(Box::new(|_| {}));

        let result = std::panic::catch_unwind(|| {
            ort::init_from(path.to_string_lossy().to_string())
                .commit()
                .unwrap()
        });

        std::panic::set_hook(save);

        result.map_err(|v| {
            anyhow::anyhow!(
                "{}",
                v.downcast::<String>().unwrap_or(Box::new(format!(
                    "failed to load the runtime library: {}",
                    path.display()
                )))
            )
        })
    }

    /// Load the model and charset from memory; only content recognition can be used — object detection panics.
    pub fn new<MODEL>(model: MODEL, charset: Charset) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: ort::Session::builder()?.commit_from_memory(model.as_ref())?,
            charset: Some(std::borrow::Cow::Owned(charset)),
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model and charset from memory and set the ORT `IntraOpNumThreads`;
    /// only content recognition can be used — object detection panics.
    pub fn new_with_num_threads<MODEL>(
        model: MODEL,
        charset: Charset,
        num_threads: usize,
    ) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: build_session(model, Some(num_threads))?,
            charset: Some(std::borrow::Cow::Owned(charset)),
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model and charset from memory; only content recognition can be used — object detection panics.
    pub fn new_ref<MODEL>(model: MODEL, charset: &'a Charset) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: ort::Session::builder()?.commit_from_memory(model.as_ref())?,
            charset: Some(std::borrow::Cow::Borrowed(charset)),
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model and charset from memory; only content recognition can be used — object detection panics.
    #[cfg(feature = "cuda")]
    pub fn new_cuda<MODEL>(model: MODEL, charset: Charset, device_id: i32) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        let builder = ort::Session::builder()?;

        let cuda = ort::CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);

        if !ort::ExecutionProvider::is_available(&cuda)? {
            anyhow::bail!("please compile ONNX Runtime with CUDA!")
        }

        ort::ExecutionProvider::register(&cuda, &builder)?;

        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: builder.commit_from_memory(model.as_ref())?,
            charset: Some(std::borrow::Cow::Owned(charset)),
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model and charset from memory and set the ORT `IntraOpNumThreads`;
    /// only content recognition can be used — object detection panics.
    #[cfg(feature = "cuda")]
    pub fn new_cuda_with_num_threads<MODEL>(
        model: MODEL,
        charset: Charset,
        device_id: i32,
        num_threads: usize,
    ) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        let builder = ort::Session::builder()?.with_intra_threads(num_threads)?;

        let cuda = ort::CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);

        if !ort::ExecutionProvider::is_available(&cuda)? {
            anyhow::bail!("please compile ONNX Runtime with CUDA!")
        }

        ort::ExecutionProvider::register(&cuda, &builder)?;

        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: builder.commit_from_memory(model.as_ref())?,
            charset: Some(std::borrow::Cow::Owned(charset)),
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model and charset from memory; only content recognition can be used — object detection panics.
    #[cfg(feature = "cuda")]
    pub fn new_cuda_ref<MODEL>(
        model: MODEL,
        charset: &'a Charset,
        device_id: i32,
    ) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        let builder = ort::Session::builder()?;

        let cuda = ort::CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);

        if !ort::ExecutionProvider::is_available(&cuda)? {
            anyhow::bail!("please compile ONNX Runtime with CUDA!")
        }

        ort::ExecutionProvider::register(&cuda, &builder)?;

        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: builder.commit_from_memory(model.as_ref())?,
            charset: Some(std::borrow::Cow::Borrowed(charset)),
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model from memory; only object detection can be used — content recognition panics.
    pub fn new_model<MODEL>(model: MODEL) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: ort::Session::builder()?.commit_from_memory(model.as_ref())?,
            charset: None,
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model from memory and set the ORT `IntraOpNumThreads`;
    /// only object detection can be used — content recognition panics.
    pub fn new_model_with_num_threads<MODEL>(
        model: MODEL,
        num_threads: usize,
    ) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: build_session(model, Some(num_threads))?,
            charset: None,
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model from memory; only object detection can be used — content recognition panics.
    #[cfg(feature = "cuda")]
    pub fn new_model_cuda<MODEL>(model: MODEL, device_id: i32) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        let builder = ort::Session::builder()?;

        let cuda = ort::CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);

        if !ort::ExecutionProvider::is_available(&cuda)? {
            anyhow::bail!("please compile ONNX Runtime with CUDA!")
        }

        ort::ExecutionProvider::register(&cuda, &builder)?;

        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: builder.commit_from_memory(model.as_ref())?,
            charset: None,
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model from memory and set the ORT `IntraOpNumThreads`;
    /// only object detection can be used — content recognition panics.
    #[cfg(feature = "cuda")]
    pub fn new_model_cuda_with_num_threads<MODEL>(
        model: MODEL,
        device_id: i32,
        num_threads: usize,
    ) -> anyhow::Result<Self>
    where
        MODEL: AsRef<[u8]>,
    {
        let builder = ort::Session::builder()?.with_intra_threads(num_threads)?;

        let cuda = ort::CUDAExecutionProvider::default()
            .with_device_id(device_id)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);

        if !ort::ExecutionProvider::is_available(&cuda)? {
            anyhow::bail!("please compile ONNX Runtime with CUDA!")
        }

        ort::ExecutionProvider::register(&cuda, &builder)?;

        Ok(Self {
            diy: is_diy(model.as_ref()),
            session: builder.commit_from_memory(model.as_ref())?,
            charset: None,
            charset_range: Vec::new(),
            det_type: DetType::Yolo,
            db_params: DbParams::default(),
        })
    }

    /// Load the model and charset from a file; only content recognition can be used — object detection panics.
    pub fn with_model_charset<PATH1, PATH2>(model: PATH1, charset: PATH2) -> anyhow::Result<Self>
    where
        PATH1: AsRef<std::path::Path>,
        PATH2: AsRef<std::path::Path>,
    {
        Self::new(
            std::fs::read(model)?,
            serde_json::from_str(&std::fs::read_to_string(charset)?)?,
        )
    }

    /// Load the model and charset from a file; only content recognition can be used — object detection panics.
    #[cfg(feature = "cuda")]
    pub fn with_model_charset_cuda<PATH1, PATH2>(
        model: PATH1,
        charset: PATH2,
        device_id: i32,
    ) -> anyhow::Result<Self>
    where
        PATH1: AsRef<std::path::Path>,
        PATH2: AsRef<std::path::Path>,
    {
        Self::new_cuda(
            std::fs::read(model)?,
            serde_json::from_str(&std::fs::read_to_string(charset)?)?,
            device_id,
        )
    }

    /// Load the model from a file; only object detection can be used — content recognition panics.
    pub fn with_model<P>(model: P) -> anyhow::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        Self::new_model(std::fs::read(model)?)
    }

    /// Load the model from a file; only object detection can be used — content recognition panics.
    #[cfg(feature = "cuda")]
    pub fn with_model_cuda<P>(model: P, device_id: i32) -> anyhow::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        Self::new_model_cuda(std::fs::read(model)?, device_id)
    }

    /// Compute the charset range for the given ranges.
    pub fn calc_ranges<R>(&self, ranges: R) -> Vec<String>
    where
        R: Into<CharsetRange>,
    {
        let new_charset = match ranges.into() {
            CharsetRange::Digit => "0123456789"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::Lowercase => "abcdefghijklmnopqrstuvwxyz"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::Uppercase => "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::LowercaseUppercase => {
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    .chars()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
            }
            CharsetRange::LowercaseDigit => "abcdefghijklmnopqrstuvwxyz0123456789"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::UppercaseDigit => "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                .chars()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
            CharsetRange::LowercaseUppercaseDigit => {
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    .chars()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
            }
            CharsetRange::DefaultCharsetLowercaseUppercaseDigit => {
                let delete_range = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    .chars()
                    .collect::<Vec<char>>();

                self.charset
                    .as_ref()
                    .expect("only the ocr model can be used")
                    .charset
                    .clone()
                    .into_iter()
                    .filter(|v| v.chars().all(|c| !delete_range.contains(&c)))
                    .collect::<Vec<String>>()
            }
            CharsetRange::Other(v) => v.chars().map(|v| v.to_string()).collect::<Vec<String>>(),
            CharsetRange::Charset(v) => return v,
        };

        // dedupe + append an empty string (keeping the original order stable)
        let mut deduped = Vec::new();
        for v in new_charset {
            if !deduped.contains(&v) {
                deduped.push(v);
            }
        }

        deduped.push("".to_string());

        deduped
    }

    /// Limit the charset range of classification_probability; only content recognition can be used — object detection panics.
    pub fn set_ranges<R>(&mut self, ranges: R)
    where
        R: Into<CharsetRange>,
    {
        self.charset_range = self.calc_ranges(ranges)
    }

    /// Content recognition; returns probabilities over the full charset. The charset range can be limited via `set_ranges` (required for word/CTC models).
    pub fn classification_probability<I>(&self, image: I) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
    {
        self.classification_probability_with_options(image, false, None, None)
    }

    /// Content recognition; returns probabilities over the full charset. The charset range can be limited via `set_ranges` (required for word/CTC models).
    /// If `png_fix` is true, png images with a transparent black background are supported.
    pub fn classification_probability_with_png_fix<I>(
        &self,
        image: I,
        png_fix: bool,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
    {
        self.classification_probability_with_options(image, png_fix, None, None)
    }

    /// Content recognition; returns probabilities over the full charset. The charset range can be limited via `set_ranges` (required for word/CTC models).
    /// If `filter` is red, only red is recognized.
    pub fn classification_probability_with_filter<I, F>(
        &self,
        image: I,
        filter: F,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
        F: Into<ColorFilter>,
    {
        self.classification_probability_with_options(image, false, Some(filter.into()), None)
    }

    /// Content recognition; returns probabilities for the given charset range, which is limited via `ranges`.
    pub fn classification_probability_with_ranges<I, R>(
        &self,
        image: I,
        ranges: R,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
        R: Into<CharsetRange>,
    {
        self.classification_probability_with_options(image, false, None, Some(ranges.into()))
    }

    /// Content recognition; returns probabilities for the given charset range. `png_fix` supports png images with a transparent black background and `ranges` limits the charset range.
    pub fn classification_probability_with_png_fix_and_ranges<I, R>(
        &self,
        image: I,
        png_fix: bool,
        ranges: R,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
        R: Into<CharsetRange>,
    {
        self.classification_probability_with_options(image, png_fix, None, Some(ranges.into()))
    }

    /// Content recognition; returns probabilities for the given charset range. `filter` selects a color filter and `ranges` limits the charset range.
    pub fn classification_probability_with_filter_and_ranges<I, F, R>(
        &self,
        image: I,
        filter: F,
        ranges: R,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
        F: Into<ColorFilter>,
        R: Into<CharsetRange>,
    {
        self.classification_probability_with_options(
            image,
            false,
            Some(filter.into()),
            Some(ranges.into()),
        )
    }

    /// Content recognition; returns probabilities over the full charset. The charset range can be limited via set_ranges (required for word/CTC models).
    /// If png_fix is true, png images with a transparent black background are supported.
    /// If filter is red, only red is recognized.
    /// If ranges is None, the charset range set by set_ranges is used.
    pub fn classification_probability_with_options<I>(
        &self,
        image: I,
        png_fix: bool,
        filter: Option<ColorFilter>,
        ranges: Option<CharsetRange>,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
    {
        self.classification_probability_with_preprocess(
            image,
            png_fix,
            filter,
            ranges,
            &PreprocessOptions::default(),
        )
    }

    /// Content recognition; returns probabilities over the full charset and supports custom preprocessing options.
    pub fn classification_probability_with_preprocess<I>(
        &self,
        image: I,
        png_fix: bool,
        filter: Option<ColorFilter>,
        ranges: Option<CharsetRange>,
        options: &PreprocessOptions,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
    {
        let charset_ranges = match &ranges {
            Some(v) => self.calc_ranges(v.clone()),
            None => self.charset_range.clone(),
        };

        if self.charset.as_ref().map(|c| c.word).unwrap_or(false) {
            // word (CTC) models also support charset_range: take each frame's argmax over the limited characters,
            // then run CTC greedy decoding; without a range, full-table logits cannot become a meaningful per-position probability matrix
            if charset_ranges.is_empty() {
                return Err(anyhow::anyhow!(
                    "classification_probability does not support word (CTC) models \
                     without charset_range; pass ranges or use classification_with_confidence instead"
                ));
            }

            return self.classification_probability_word(
                image,
                png_fix,
                filter,
                charset_ranges,
                options,
            );
        }

        if self.diy {
            // non-word DIY models output class indices and have no probability matrix
            return Err(anyhow::anyhow!(
                "classification_probability can only use the official model"
            ));
        }

        let image = match filter {
            Some(v) => v.filter(image.as_ref())?,
            None => image::load_from_memory(image.as_ref())?,
        };

        let tensor = self.preprocess(image, png_fix, options)?;

        let charset = self.charset()?;
        let charset = &charset.charset;

        let ort_outs = &self.session.run(ort::inputs![tensor]?)?;

        let ort_outs = &ort_outs[0].try_extract_tensor()?;

        // looks like [[[1,2,3,4]], [[1,2,3,4]], [[1,2,3,4]]]
        let ort_outs = ort_outs.mapv(f32::exp);

        // looks like [[1,2,3,4], [1,2,3,4], [1,2,3,4]]
        let ort_outs_sum = ort_outs.sum_axis(ndarray::Axis(2));

        // create a zeros array with the same shape
        let mut ort_outs_probability = ndarray::Array::<f32, _>::zeros(ort_outs.raw_dim());

        for i in 0..ort_outs.shape()[0] {
            let mut a = ort_outs_probability.slice_mut(ndarray::s![i, .., ..]);
            let b = ort_outs.slice(ndarray::s![i, .., ..]);
            let c = ort_outs_sum.slice(ndarray::s![i, ..]);
            let d = &b / &c;

            a.assign(&d);
        }

        // after calling next, looks like [[1,2,3,4], [1,2,3,4], [1,2,3,4]]
        let ort_outs_probability = ort_outs_probability
            .axis_iter(ndarray::Axis(1))
            .next()
            .ok_or(anyhow::anyhow!("expect axis 1"))?;

        let mut result = Vec::new();

        // flatten
        for i in ort_outs_probability.axis_iter(ndarray::Axis(0)) {
            result.push(i.into_diag().to_vec());
        }

        if charset_ranges.is_empty() {
            // return probabilities for all characters
            Ok(CharacterProbability {
                text: None,
                charset: charset.clone(),
                confidence: Some(average_max_confidence(&result)),
                probability: result,
            })
        } else {
            // extract the requested characters' probabilities from the model output
            // if a character is not in the charset, set its probability to -1.0 to mean unknown
            let mut probability_result_index = Vec::new();

            for i in &charset_ranges {
                if let Some(v) = charset.iter().position(|v| v == i) {
                    probability_result_index.push(v);
                } else {
                    probability_result_index.push(usize::MAX);
                }
            }

            let mut probability_result = Vec::new();

            for item in &result {
                let mut inner_vec = Vec::new();

                for &i in &probability_result_index {
                    if i != usize::MAX {
                        inner_vec.push(item[i]);
                    } else {
                        inner_vec.push(-1.0);
                    }
                }

                probability_result.push(inner_vec);
            }

            Ok(CharacterProbability {
                text: None,
                charset: charset_ranges.clone(),
                confidence: Some(average_max_confidence(&probability_result)),
                probability: probability_result,
            })
        }
    }

    /// Restricted probability interface for word (CTC) models: softmax over the entire charset per frame,
    /// returning per-frame probabilities of the requested characters (including blank); the text comes from
    /// restricted CTC greedy decoding (each frame takes its argmax among the requested characters and blank, then merges consecutive repeats and drops blank).
    fn classification_probability_word<I>(
        &self,
        image: I,
        png_fix: bool,
        filter: Option<ColorFilter>,
        charset_ranges: Vec<String>,
        options: &PreprocessOptions,
    ) -> anyhow::Result<CharacterProbability>
    where
        I: AsRef<[u8]>,
    {
        let image = match filter {
            Some(v) => v.filter(image.as_ref())?,
            None => image::load_from_memory(image.as_ref())?,
        };

        let tensor = self.preprocess(image, png_fix, options)?;
        let charset = self.charset()?;
        let charset_list = &charset.charset;

        // requested characters → model class indices; empty string = blank (0), unknown characters recorded as None (probability -1.0)
        let index_map = map_charset_range_indices(charset_list, &charset_ranges);
        let allowed: Vec<usize> = index_map
            .iter()
            .flatten()
            .copied()
            .filter(|&i| i != 0)
            .collect();

        let outputs = self.session.run(ort::inputs![tensor]?)?;

        for out in outputs.values() {
            if let Ok(logits) = out.try_extract_tensor::<f32>() {
                let decoded = ctc_greedy_decode_restricted(&logits, &allowed);

                let text = decoded
                    .iter()
                    .filter(|&&(idx, _)| idx < charset_list.len())
                    .map(|&(idx, _)| charset_list[idx].to_string())
                    .collect::<String>();

                let (timesteps, class_count) = if logits.ndim() == 3 {
                    (logits.shape()[1], logits.shape()[2])
                } else {
                    (logits.shape()[0], logits.shape()[1])
                };

                let probability_output = is_probability_output(&logits);
                let mut probability = Vec::with_capacity(timesteps);

                for t in 0..timesteps {
                    let logit = |c: usize| -> f32 {
                        if logits.ndim() == 3 {
                            logits[[0, t, c]]
                        } else {
                            logits[[t, c]]
                        }
                    };

                    // probability output takes the column directly; logits output gets a full-table softmax first (numerically stable)
                    let (max_logit, sum_exp) = if probability_output {
                        (0.0, 1.0)
                    } else {
                        let mut max_logit = f32::NEG_INFINITY;

                        for c in 0..class_count {
                            max_logit = max_logit.max(logit(c));
                        }

                        let mut sum_exp = 0.0f32;

                        for c in 0..class_count {
                            sum_exp += (logit(c) - max_logit).exp();
                        }

                        (max_logit, sum_exp)
                    };

                    let prob_of = |c: usize| -> f32 {
                        if probability_output {
                            logit(c)
                        } else {
                            (logit(c) - max_logit).exp() / sum_exp
                        }
                    };

                    let row = index_map
                        .iter()
                        .map(|idx| match idx {
                            Some(i) if *i < class_count => prob_of(*i),
                            _ => -1.0,
                        })
                        .collect();

                    probability.push(row);
                }

                let confidence = average_max_confidence(&probability);

                return Ok(CharacterProbability {
                    text: Some(text),
                    charset: charset_ranges,
                    probability,
                    confidence: Some(confidence),
                });
            }
        }

        // exports providing only per-frame class indices: no probability matrix is available, so filter the decode result by the allowed characters
        for out in outputs.values() {
            if let Ok(indices) = out.try_extract_tensor::<i64>() {
                let decoded = ctc_greedy_decode_indices(&indices);

                let text = decoded
                    .iter()
                    .filter(|&&(idx, _)| idx < charset_list.len() && allowed.contains(&idx))
                    .map(|&(idx, _)| charset_list[idx].to_string())
                    .collect::<String>();

                return Ok(CharacterProbability {
                    text: Some(text),
                    charset: charset_ranges,
                    probability: Vec::new(),
                    confidence: None,
                });
            }
        }

        Err(anyhow::anyhow!(
            "word model: expected an f32 logits or i64 index output"
        ))
    }

    /// Content recognition.
    pub fn classification<I>(&self, image: I) -> anyhow::Result<String>
    where
        I: AsRef<[u8]>,
    {
        self.classification_with_options(image, false, None)
    }

    /// Content recognition.
    pub fn classification_with_path<P>(&self, path: P) -> anyhow::Result<String>
    where
        P: AsRef<std::path::Path>,
    {
        self.classification(std::fs::read(path)?)
    }

    /// Content recognition; if png_fix is true, png images with a transparent black background are supported.
    pub fn classification_with_png_fix<I>(&self, image: I, png_fix: bool) -> anyhow::Result<String>
    where
        I: AsRef<[u8]>,
    {
        self.classification_with_options(image, png_fix, None)
    }

    /// Content recognition; if png_fix is true, png images with a transparent black background are supported.
    pub fn classification_with_path_png_fix<P>(
        &self,
        path: P,
        png_fix: bool,
    ) -> anyhow::Result<String>
    where
        P: AsRef<std::path::Path>,
    {
        self.classification_with_png_fix(std::fs::read(path)?, png_fix)
    }

    /// Content recognition; if filter is red, only red is recognized.
    pub fn classification_with_filter<I, F>(&self, image: I, filter: F) -> anyhow::Result<String>
    where
        I: AsRef<[u8]>,
        F: Into<ColorFilter>,
    {
        self.classification_with_options(image, false, Some(filter.into()))
    }

    /// Content recognition; if filter is red, only red is recognized.
    pub fn classification_with_path_filter<P, F>(
        &self,
        path: P,
        filter: F,
    ) -> anyhow::Result<String>
    where
        P: AsRef<std::path::Path>,
        F: Into<ColorFilter>,
    {
        self.classification_with_filter(std::fs::read(path)?, filter)
    }

    /// Content recognition; if png_fix is true, png images with a transparent black background are supported; if filter is red, only red is recognized.
    pub fn classification_with_options<I>(
        &self,
        image: I,
        png_fix: bool,
        filter: Option<ColorFilter>,
    ) -> anyhow::Result<String>
    where
        I: AsRef<[u8]>,
    {
        self.classification_with_options_and_preprocess(
            image,
            png_fix,
            filter,
            &PreprocessOptions::default(),
        )
    }

    /// Content recognition; supports `png_fix`, `filter` and custom preprocessing options (binarize / denoise / deskew / resize).
    pub fn classification_with_options_and_preprocess<I>(
        &self,
        image: I,
        png_fix: bool,
        filter: Option<ColorFilter>,
        options: &PreprocessOptions,
    ) -> anyhow::Result<String>
    where
        I: AsRef<[u8]>,
    {
        let image = match filter {
            Some(v) => v.filter(image.as_ref())?,
            None => image::load_from_memory(image.as_ref())?,
        };

        self.classification_internal(image, png_fix, options)
    }

    /// Content recognition; returns each character with its confidence.
    ///
    /// `word` (CTC sequence) models use CTC greedy decoding: softmax → per-frame argmax → merge consecutive
    /// repeats → drop blank (class 0); confidence is the maximum softmax probability over the character's
    /// consecutive frames. Non-`word` models return the per-position argmax softmax confidence (DIY models have no probability output, recorded as 1.0).
    pub fn classification_with_confidence<I>(&self, image: I) -> anyhow::Result<Vec<(String, f32)>>
    where
        I: AsRef<[u8]>,
    {
        let image = image::load_from_memory(image.as_ref())?;
        let decoded =
            self.classification_internal_conf(image, false, &PreprocessOptions::default())?;
        let charset = self.charset()?;

        Ok(decoded
            .iter()
            .filter(|&&(idx, _)| idx < charset.charset.len())
            .map(|&(idx, conf)| (charset.charset[idx].to_string(), conf))
            .collect())
    }

    /// Get the model charset config; returns an error instead of panicking when none is configured.
    fn charset(&self) -> anyhow::Result<&Charset> {
        self.charset
            .as_ref()
            .map(|v| v.as_ref())
            .ok_or_else(|| anyhow::anyhow!("this session has no charset configured"))
    }

    /// Internal: load an image, resize, convert channels and normalize into an input tensor.
    fn preprocess(
        &self,
        image: image::DynamicImage,
        png_fix: bool,
        options: &PreprocessOptions,
    ) -> anyhow::Result<ndarray::Array4<f32>> {
        let charset = self.charset()?;
        let resize = charset.image;
        let channel = charset.channel;

        // preprocessing enhancements: denoise → binarize → deskew (all applied to the grayscale image)
        let mut image = image;

        if options.denoise {
            let gray = image.to_luma8();
            let gray = imageproc::filter::median_filter(&gray, 1, 1);
            image = image::DynamicImage::ImageLuma8(gray);
        }

        if options.binarize {
            let gray = image.to_luma8();
            let level = imageproc::contrast::otsu_level(&gray);
            let binary = imageproc::contrast::threshold(
                &gray,
                level,
                imageproc::contrast::ThresholdType::Binary,
            );
            image = image::DynamicImage::ImageLuma8(binary);
        }

        if options.deskew {
            let gray = image.to_luma8();
            let angle = find_deskew_angle(&gray);
            let rotated = imageproc::geometric_transformations::rotate_about_center(
                &gray,
                angle,
                imageproc::geometric_transformations::Interpolation::Bilinear,
                image::Luma([255]),
            );
            image = image::DynamicImage::ImageLuma8(rotated);
        }

        // resize the image with ANTIALIAS (Lanczos3)
        let image = match options.resize {
            ResizeStrategy::Exact(w, h) => {
                image.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
            }
            ResizeStrategy::Auto => {
                if resize[0] == -1 {
                    // height fixed to resize[1], width scaled proportionally (also applies to CTC sequence models)
                    image.resize_exact(
                        image.width() * resize[1] as u32 / image.height(),
                        resize[1] as u32,
                        image::imageops::FilterType::Lanczos3,
                    )
                } else {
                    image.resize_exact(
                        resize[0] as u32,
                        resize[1] as u32,
                        image::imageops::FilterType::Lanczos3,
                    )
                }
            }
        };

        // convert the image to a tensor (explicit per-pixel fill, avoiding interleaved RGB bytes being mistaken for planar)
        let channel = channel as usize;
        let width = image.width() as usize;
        let height = image.height() as usize;
        let mut tensor = ndarray::Array::from_shape_vec(
            (1, channel, height, width),
            vec![0f32; channel * height * width],
        )?;

        let mean = self.charset.as_ref().and_then(|c| c.mean);
        let std = self.charset.as_ref().and_then(|c| c.std);
        let bgr = self.charset.as_ref().and_then(|c| c.bgr).unwrap_or(false);

        if channel == 1 {
            let gray = image.to_luma8();

            for i in 0..height {
                for j in 0..width {
                    let now = gray.get_pixel(j as u32, i as u32)[0] as f32;

                    tensor[[0, 0, i, j]] = if let (Some(m), Some(s)) = (mean, std) {
                        ((now / 255f32) - m[0]) / s[0]
                    } else if self.diy {
                        ((now / 255f32) - 0.456f32) / 0.224f32
                    } else {
                        ((now / 255f32) - 0.5f32) / 0.5f32
                    };
                }
            }
        } else {
            let rgb = if png_fix {
                png_rgba_black_preprocess(&image).to_rgb8()
            } else {
                image.to_rgb8()
            };

            for i in 0..height {
                for j in 0..width {
                    let p = rgb.get_pixel(j as u32, i as u32);
                    let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);

                    if let (Some(m), Some(s)) = (mean, std) {
                        // explicit normalization: PaddleOCR style, per channel (x/255 - mean)/std
                        let (c0, c2) = if bgr { (b, r) } else { (r, b) };
                        tensor[[0, 0, i, j]] = ((c0 / 255f32) - m[0]) / s[0];
                        tensor[[0, 1, i, j]] = ((g / 255f32) - m[1]) / s[1];
                        tensor[[0, 2, i, j]] = ((c2 / 255f32) - m[2]) / s[2];
                    } else if self.diy {
                        // custom models: ImageNet mean/std
                        tensor[[0, 0, i, j]] = ((r / 255f32) - 0.485f32) / 0.229f32;
                        tensor[[0, 1, i, j]] = ((g / 255f32) - 0.456f32) / 0.224f32;
                        tensor[[0, 2, i, j]] = ((b / 255f32) - 0.406f32) / 0.225f32;
                    } else {
                        // official models: uniform (x/255-0.5)/0.5 for every channel
                        tensor[[0, 0, i, j]] = ((r / 255f32) - 0.5f32) / 0.5f32;
                        tensor[[0, 1, i, j]] = ((g / 255f32) - 0.5f32) / 0.5f32;
                        tensor[[0, 2, i, j]] = ((b / 255f32) - 0.5f32) / 0.5f32;
                    }
                }
            }
        }

        Ok(tensor)
    }

    /// Internal: run content recognition on a loaded image and decode it, returning each character with its confidence.
    fn classification_internal_conf(
        &self,
        image: image::DynamicImage,
        png_fix: bool,
        options: &PreprocessOptions,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let tensor = self.preprocess(image, png_fix, options)?;

        let word = self.charset()?.word;

        if word {
            let outputs = self.session.run(ort::inputs![tensor]?)?;

            // prefer f32 output → full CTC greedy decoding (with per-character confidence);
            // PaddleOCR exports already apply softmax (each frame sums to 1), custom exports may be logits
            for out in outputs.values() {
                if let Ok(logits) = out.try_extract_tensor::<f32>() {
                    return Ok(if is_probability_output(&logits) {
                        ctc_greedy_decode_probability(&logits)
                    } else {
                        ctc_greedy_decode(&logits)
                    });
                }
            }

            // fall back to i64 argmax output → only drop blank and merge consecutive repeats
            for out in outputs.values() {
                if let Ok(indices) = out.try_extract_tensor::<i64>() {
                    return Ok(ctc_greedy_decode_indices(&indices));
                }
            }

            return Err(anyhow::anyhow!(
                "word model: expected an f32 logits or i64 index output"
            ));
        }

        let result = &self.session.run(ort::inputs![tensor]?)?[0];

        if self.diy {
            // DIY models output u32 class indices; take the per-frame argmax, then drop blank and merge consecutive repeats.
            // Not verified end-to-end with a real DIY model (no DIY model sample in the repo); confidence is recorded as 1.0.
            let result = result.try_extract_tensor::<u32>()?;
            let mut last_item = 0;
            let mut out = Vec::new();

            for &v in result.iter() {
                if v != 0 && v != last_item {
                    last_item = v;
                    out.push((v as usize, 1.0));
                }
            }

            Ok(out)
        } else {
            let result = result.try_extract_tensor::<f32>()?;
            let mut last_item = 0;
            let mut out = Vec::new();

            for row in result.rows() {
                // find the current row's argmax and its softmax confidence (numerically stable)
                let mut max = f32::NEG_INFINITY;
                let mut idx = 0usize;

                for (i, &v) in row.iter().enumerate() {
                    if v > max {
                        max = v;
                        idx = i;
                    }
                }

                let mut sum = 0.0f32;
                for &v in row.iter() {
                    sum += (v - max).exp();
                }
                let conf = 1.0 / sum;

                // filter out class 0 (blank) and consecutive repeats
                if idx != 0 && idx != last_item {
                    last_item = idx;
                    out.push((idx, conf));
                }
            }

            Ok(out)
        }
    }

    /// Internal: run content recognition on a loaded image and decode it to text.
    fn classification_internal(
        &self,
        image: image::DynamicImage,
        png_fix: bool,
        options: &PreprocessOptions,
    ) -> anyhow::Result<String> {
        let decoded = self.classification_internal_conf(image, png_fix, options)?;
        let charset = self.charset()?;
        let charset = &charset.charset;

        Ok(decoded
            .iter()
            .filter(|&&(idx, _)| idx < charset.len())
            .map(|&(idx, _)| charset[idx].to_string())
            .collect::<String>())
    }

    /// Crop the image by the coordinates, then run content recognition.
    pub fn classification_bbox<I>(
        &self,
        image: I,
        bbox: &Vec<BBox>,
    ) -> anyhow::Result<Vec<(BBox, String)>>
    where
        I: AsRef<[u8]>,
    {
        let image = image::load_from_memory(image.as_ref())?;

        let mut result = Vec::new();

        for i in bbox {
            let crop =
                image::imageops::crop_imm(&image, i.x1, i.y1, i.x2 - i.x1 + 1, i.y2 - i.y1 + 1)
                    .to_image();

            result.push((
                *i,
                self.classification_internal(crop.into(), false, &PreprocessOptions::default())?,
            ));
        }

        Ok(result)
    }

    /// Crop the image by the coordinates, then run content recognition.
    pub fn classification_bbox_with_path<P>(
        &self,
        path: P,
        bbox: &Vec<BBox>,
    ) -> anyhow::Result<Vec<(BBox, String)>>
    where
        P: AsRef<std::path::Path>,
    {
        self.classification_bbox(std::fs::read(path)?, bbox)
    }

    /// Object detection with the default NMS / score thresholds; returns bounding-box coordinates only.
    pub fn detection<I>(&self, image: I) -> anyhow::Result<Vec<BBox>>
    where
        I: AsRef<[u8]>,
    {
        Ok(self
            .detection_with_options(image, NMS_THR, SCORE_THR)?
            .into_iter()
            .map(|(bbox, _)| bbox)
            .collect())
    }

    /// Object detection with the default NMS / score thresholds; returns bounding-box coordinates and their confidence scores.
    pub fn detection_with_score<I>(&self, image: I) -> anyhow::Result<Vec<(BBox, f32)>>
    where
        I: AsRef<[u8]>,
    {
        self.detection_with_options(image, NMS_THR, SCORE_THR)
    }

    /// Object detection with custom NMS IoU and score thresholds; returns bounding-box coordinates and their confidence scores.
    pub fn detection_with_options<I>(
        &self,
        image: I,
        nms_thr: f32,
        score_thr: f32,
    ) -> anyhow::Result<Vec<(BBox, f32)>>
    where
        I: AsRef<[u8]>,
    {
        match self.det_type {
            DetType::Yolo => self.detection_yolo(image, nms_thr, score_thr),
            DetType::Db => self.detection_db(image, nms_thr, score_thr),
        }
    }

    /// Set the detection decoding type (`Yolo` or `Db`). Recognition models (rec) are unaffected.
    pub fn set_det_type(&mut self, ty: DetType) {
        self.det_type = ty;
    }

    /// Current detection decoding type.
    pub fn det_type(&self) -> DetType {
        self.det_type
    }

    /// Set the DB post-processing parameters.
    pub fn set_db_params(&mut self, params: DbParams) {
        self.db_params = params;
    }

    /// YOLO (anchor-grid) decoding: fixed 416×416 resize + gray padding, output `[1, i, 6]`.
    fn detection_yolo<I>(
        &self,
        image: I,
        nms_thr: f32,
        score_thr: f32,
    ) -> anyhow::Result<Vec<(BBox, f32)>>
    where
        I: AsRef<[u8]>,
    {
        let image = image::load_from_memory(image.as_ref())?.to_rgb8();
        let (h, w) = (image.height(), image.width());
        let r = (MODEL_WIDTH as f32 / h as f32).min(MODEL_WIDTH as f32 / w as f32);
        let new_w = (w as f32 * r) as u32;
        let new_h = (h as f32 * r) as u32;

        let resized =
            image::imageops::resize(&image, new_w, new_h, image::imageops::FilterType::Triangle);

        let mut padded =
            image::RgbImage::from_pixel(MODEL_WIDTH, MODEL_HEIGHT, image::Rgb([114, 114, 114]));

        image::GenericImage::copy_from(&mut padded, &resized, 0, 0)?;

        let mut input_tensor = ndarray::Array::from_shape_vec(
            (1, 3, MODEL_WIDTH as usize, MODEL_HEIGHT as usize),
            vec![0f32; 3 * MODEL_WIDTH as usize * MODEL_HEIGHT as usize],
        )?;

        for (x, y, p) in padded.enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;

            input_tensor[[0, 0, y, x]] = p[2] as f32;
            input_tensor[[0, 1, y, x]] = p[1] as f32;
            input_tensor[[0, 2, y, x]] = p[0] as f32;
        }

        let output = &self.session.run(ort::inputs![input_tensor]?)?[0];
        let output = output.try_extract_tensor::<f32>()?;
        let mut dets = Vec::new();
        let mut grid_offset = 0;

        for &stride in &STRIDES {
            let h_grid = MODEL_WIDTH / stride;
            let w_grid = MODEL_HEIGHT / stride;

            for gy in 0..h_grid {
                for gx in 0..w_grid {
                    let i = grid_offset + (gy * w_grid + gx) as usize;
                    let obj = output[[0, i, 4]];
                    let cls = output[[0, i, 5]];
                    let score = obj * cls;

                    if score < score_thr {
                        continue;
                    }

                    let cx = output[[0, i, 0]];
                    let cy = output[[0, i, 1]];
                    let w = output[[0, i, 2]];
                    let h = output[[0, i, 3]];
                    let x1 = (cx + gx as f32) * stride as f32;
                    let y1 = (cy + gy as f32) * stride as f32;
                    let bw = w.exp() * stride as f32;
                    let bh = h.exp() * stride as f32;
                    let mut x1 = x1 - bw / 2.0;
                    let mut y1 = y1 - bh / 2.0;
                    let mut x2 = x1 + bw;
                    let mut y2 = y1 + bh;

                    x1 /= r;
                    y1 /= r;
                    x2 /= r;
                    y2 /= r;

                    dets.push((score, [x1, y1, x2, y2]));
                }
            }
            grid_offset += (h_grid * w_grid) as usize;
        }

        let scores: Vec<f32> = dets.iter().map(|d| d.0).collect();
        let mut order: Vec<usize> = (0..scores.len()).collect();

        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

        let mut keep = Vec::new();

        while !order.is_empty() {
            let i = order[0];

            keep.push(i);

            let (x1, y1, x2, y2) = {
                let b = dets[i].1;
                (b[0], b[1], b[2], b[3])
            };

            let area_i = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);
            let mut new_order = Vec::new();

            for &j in &order[1..] {
                let (x1j, y1j, x2j, y2j) = {
                    let b = dets[j].1;
                    (b[0], b[1], b[2], b[3])
                };

                let xx1 = x1.max(x1j);
                let yy1 = y1.max(y1j);
                let xx2 = x2.min(x2j);
                let yy2 = y2.min(y2j);
                let w = (xx2 - xx1 + 1.0).max(0.0);
                let h = (yy2 - yy1 + 1.0).max(0.0);
                let inter = w * h;
                let area_j = (x2j - x1j + 1.0) * (y2j - y1j + 1.0);
                let ovr = inter / (area_i + area_j - inter);

                if ovr <= nms_thr {
                    new_order.push(j);
                }
            }
            order = new_order;
        }

        let (ow, oh) = (w as f32, h as f32);
        let mut result = Vec::new();

        for i in keep {
            let (x1, y1, x2, y2) = {
                let b = dets[i].1;
                (b[0], b[1], b[2], b[3])
            };

            let x1 = x1.clamp(0.0, ow) as u32;
            let y1 = y1.clamp(0.0, oh) as u32;
            let x2 = x2.clamp(0.0, ow) as u32;
            let y2 = y2.clamp(0.0, oh) as u32;

            result.push((crate::BBox { x1, y1, x2, y2 }, scores[i]));
        }

        Ok(result)
    }

    /// DB detection: resize + pad to a `limit_side_len` square (PaddleOCR export graphs have stable
    /// internal Resize shapes at fixed sizes), BGR + ImageNet normalization, then DB decoding of the probability map.
    fn detection_db<I>(
        &self,
        image: I,
        nms_thr: f32,
        score_thr: f32,
    ) -> anyhow::Result<Vec<(BBox, f32)>>
    where
        I: AsRef<[u8]>,
    {
        let img = image::load_from_memory(image.as_ref())?;
        let (ow, oh) = (img.width() as f32, img.height() as f32);

        // DetResizeForTest defaults to limit_side_len=736; padding to a square keeps shapes stable
        const LIMIT_SIDE_LEN: u32 = 736;
        let scale = LIMIT_SIDE_LEN as f32 / ow.max(oh);
        let nw = (ow * scale).round().max(1.0) as u32;
        let nh = (oh * scale).round().max(1.0) as u32;

        let rgb = img.to_rgb8();
        let resized = image::imageops::resize(&rgb, nw, nh, image::imageops::FilterType::Triangle);

        // content centered on a square canvas (this export graph produces a stable probability map with this layout)
        let offset_x = (LIMIT_SIDE_LEN - nw) / 2;
        let offset_y = (LIMIT_SIDE_LEN - nh) / 2;

        let mut padded = image::RgbImage::from_pixel(
            LIMIT_SIDE_LEN,
            LIMIT_SIDE_LEN,
            image::Rgb([114, 114, 114]),
        );
        image::GenericImage::copy_from(&mut padded, &resized, offset_x, offset_y)?;

        let mut tensor = ndarray::Array::from_shape_vec(
            (1, 3, LIMIT_SIDE_LEN as usize, LIMIT_SIDE_LEN as usize),
            vec![0f32; 3 * LIMIT_SIDE_LEN as usize * LIMIT_SIDE_LEN as usize],
        )?;

        for (x, y, p) in padded.enumerate_pixels() {
            tensor[[0, 0, y as usize, x as usize]] = (p[2] as f32 / 255.0 - 0.485) / 0.229;
            tensor[[0, 1, y as usize, x as usize]] = (p[1] as f32 / 255.0 - 0.456) / 0.224;
            tensor[[0, 2, y as usize, x as usize]] = (p[0] as f32 / 255.0 - 0.406) / 0.225;
        }

        let output = &self.session.run(ort::inputs![tensor]?)?[0];
        let map = output.try_extract_tensor::<f32>()?;
        let map = map.into_dimensionality::<ndarray::Ix4>()?;
        let mh = map.shape()[2] as u32;
        let mw = map.shape()[3] as u32;

        // map boxes whose origin is the square's top-left corner back to original-image coordinates
        let to_orig = move |mx: f32, my: f32| -> (f32, f32) {
            (
                ((mx - offset_x as f32) / scale).clamp(0.0, ow),
                ((my - offset_y as f32) / scale).clamp(0.0, oh),
            )
        };

        db_decode_prob_map(
            &map,
            self.db_params,
            nms_thr,
            score_thr,
            (ow, oh),
            (mw, mh),
            &to_orig,
        )
    }

    /// Object detection.
    pub fn detection_with_path<P>(&self, path: P) -> anyhow::Result<Vec<BBox>>
    where
        P: AsRef<std::path::Path>,
    {
        self.detection(std::fs::read(path)?)
    }
}

/// DB probability-map decoding (a simplified PaddleOCR `DBPostProcess`):
/// threshold binarization → connected components → bounding boxes → score filtering → expand by `unclip_ratio` → map back to the original image → NMS.
fn db_decode_prob_map(
    map: &ndarray::ArrayView4<f32>,
    params: DbParams,
    nms_thr: f32,
    score_thr: f32,
    orig: (f32, f32),
    map_dims: (u32, u32),
    to_orig: &impl Fn(f32, f32) -> (f32, f32),
) -> anyhow::Result<Vec<(BBox, f32)>> {
    let (ow, oh) = orig;
    let (mw, mh) = map_dims;

    // 1) binarization + connected components (4-neighborhood BFS), recording each component's bounding box and mean score
    let mut visited = vec![false; mw as usize * mh as usize];
    let mut comps: Vec<(u32, u32, u32, u32, f32)> = Vec::new();

    for y in 0..mh {
        for x in 0..mw {
            let idx = y as usize * mw as usize + x as usize;
            if visited[idx] || map[[0, 0, y as usize, x as usize]] <= params.thresh {
                continue;
            }

            let mut stack = vec![(x, y)];
            visited[idx] = true;
            let (mut minx, mut maxx, mut miny, mut maxy) = (x, x, y, y);
            let (mut sum, mut cnt) = (0.0f32, 0u32);

            while let Some((cx, cy)) = stack.pop() {
                sum += map[[0, 0, cy as usize, cx as usize]];
                cnt += 1;

                for (dx, dy) in [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= mw as i32 || ny >= mh as i32 {
                        continue;
                    }
                    let nx = nx as u32;
                    let ny = ny as u32;
                    let ni = ny as usize * mw as usize + nx as usize;
                    if !visited[ni] && map[[0, 0, ny as usize, nx as usize]] > params.thresh {
                        visited[ni] = true;
                        minx = minx.min(nx);
                        maxx = maxx.max(nx);
                        miny = miny.min(ny);
                        maxy = maxy.max(ny);
                        stack.push((nx, ny));
                    }
                }
            }

            if cnt == 0 {
                continue;
            }

            comps.push((minx, miny, maxx, maxy, sum / cnt as f32));
        }
    }

    // 2) score filtering (stricter of box_thresh and the caller's score_thr); sort by score and truncate the candidate count
    comps.retain(|c| c.4 >= params.box_thresh && c.4 >= score_thr);
    comps.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    comps.truncate(params.max_candidates);

    // 3) unclip expansion + map back to original-image coordinates + clamp to bounds
    let unclip = params.unclip_ratio;
    let mut boxes: Vec<(f32, [f32; 4])> = Vec::new();

    for (x1, y1, x2, y2, score) in comps {
        let w = (x2 - x1) as f32;
        let h = (y2 - y1) as f32;
        let dw = w * (unclip - 1.0) / 2.0;
        let dh = h * (unclip - 1.0) / 2.0;

        let (x1, y1) = to_orig(x1 as f32 - dw, y1 as f32 - dh);
        let (x2, y2) = to_orig(x2 as f32 + dw, y2 as f32 + dh);

        if x2 - x1 < 1.0 || y2 - y1 < 1.0 {
            continue;
        }

        boxes.push((score, [x1, y1, x2, y2]));
    }

    // 4) NMS (same IoU filtering as the YOLO path)
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| boxes[b].0.partial_cmp(&boxes[a].0).unwrap());
    let mut keep = Vec::new();

    while !order.is_empty() {
        let i = order[0];
        keep.push(i);

        let (x1, y1, x2, y2) = {
            let b = boxes[i].1;
            (b[0], b[1], b[2], b[3])
        };
        let area_i = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);
        let mut new_order = Vec::new();

        for &j in &order[1..] {
            let (x1j, y1j, x2j, y2j) = {
                let b = boxes[j].1;
                (b[0], b[1], b[2], b[3])
            };

            let xx1 = x1.max(x1j);
            let yy1 = y1.max(y1j);
            let xx2 = x2.min(x2j);
            let yy2 = y2.min(y2j);
            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;
            let area_j = (x2j - x1j + 1.0) * (y2j - y1j + 1.0);
            let ovr = inter / (area_i + area_j - inter);

            if ovr <= nms_thr {
                new_order.push(j);
            }
        }

        order = new_order;
    }

    // 5) output (clamped to the original image bounds)
    let mut result = Vec::with_capacity(keep.len());
    for i in keep {
        let (x1, y1, x2, y2) = {
            let b = boxes[i].1;
            (b[0], b[1], b[2], b[3])
        };
        let x1 = x1.clamp(0.0, ow) as u32;
        let y1 = y1.clamp(0.0, oh) as u32;
        let x2 = x2.clamp(0.0, ow) as u32;
        let y2 = y2.clamp(0.0, oh) as u32;
        result.push((BBox { x1, y1, x2, y2 }, boxes[i].0));
    }

    Ok(result)
}

// cargo test --no-default-features --features download-binaries
#[cfg(test)]
mod tests {
    use image::Pixel;

    use super::*;

    fn read_image(path: &str) -> Vec<u8> {
        std::fs::read(path).unwrap_or_else(|e| {
            panic!("failed to read image {}: {}", path, e);
        })
    }

    #[test]
    fn classification_probability() {
        let mut kaptchaocr = kaptchaocr_classification().unwrap();

        // CharsetRange::LowercaseUppercase: uppercase and lowercase letters
        kaptchaocr.set_ranges(3);

        let mut result = kaptchaocr
            .classification_probability(read_image("image/3.png"))
            .unwrap();

        println!("result: {}", result.get_text());
        println!("confidence: {}", result.get_confidence());

        // whoa, that looks like too much data — careful, it might hang!
        // println!("probability: {}", result.json());
    }

    #[test]
    fn classification_filter() {
        let kaptchaocr = kaptchaocr_classification().unwrap();

        println!(
            "{}",
            kaptchaocr
                .classification_with_filter(read_image("image/4.png"), "green")
                .unwrap()
        );

        println!(
            "{}",
            kaptchaocr
                .classification_with_filter(
                    read_image("image/4.png"),
                    [((40, 50, 50), (80, 255, 255))]
                )
                .unwrap()
        );
    }

    #[test]
    fn classification() {
        let kaptchaocr = kaptchaocr_classification().unwrap();

        println!(
            "{}",
            kaptchaocr
                .classification(read_image("image/1.png"))
                .unwrap()
        );

        println!(
            "{}",
            kaptchaocr
                .classification(read_image("image/2.png"))
                .unwrap()
        );

        println!(
            "{}",
            kaptchaocr
                .classification(read_image("image/3.png"))
                .unwrap()
        );

        println!(
            "{}",
            kaptchaocr
                .classification(read_image("image/4.png"))
                .unwrap()
        );

        println!(
            "{}",
            kaptchaocr
                .classification(read_image("image/su.png"))
                .unwrap()
        );
    }

    #[test]
    fn classification_bbox() {
        let input = read_image("image/6.jpg");
        let kaptchaocr = kaptchaocr_detection().unwrap();
        let result = kaptchaocr.detection(input.clone()).unwrap();
        let kaptchaocr = kaptchaocr_classification().unwrap();
        let result = kaptchaocr.classification_bbox(input, &result).unwrap();

        println!("{:?}", result);
    }

    #[test]
    fn detection() {
        let kaptchaocr = kaptchaocr_detection().unwrap();
        let input = read_image("image/5.jpg");
        let result = kaptchaocr.detection(input.clone()).unwrap();

        println!("{:?}", result);

        // draw red boxes
        let mut image = image::load_from_memory(&input).unwrap().to_rgb8();

        for v in result {
            for i in v.x1..=v.x2 {
                image[(i, v.y1)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.x1..=v.x2 {
                image[(i, v.y2)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.y1..=v.y2 {
                image[(v.x1, i)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.y1..=v.y2 {
                image[(v.x2, i)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }
        }

        image.save("./output1.jpg").unwrap();

        let input = read_image("image/6.jpg");
        let result = kaptchaocr.detection(input.clone()).unwrap();

        println!("{:?}", result);

        // draw red boxes
        let mut image = image::load_from_memory(&input).unwrap().to_rgb8();

        for v in result {
            for i in v.x1..=v.x2 {
                image[(i, v.y1)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.x1..=v.x2 {
                image[(i, v.y2)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.y1..=v.y2 {
                image[(v.x1, i)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }

            for i in v.y1..=v.y2 {
                image[(v.x2, i)] = *image::Rgb::from_slice(&[237, 28, 36]);
            }
        }

        image.save("./output2.jpg").unwrap();
    }

    #[test]
    fn slide_match() {
        let result =
            crate::slide_match(read_image("image/hk.png"), read_image("image/bg.png")).unwrap();

        println!("{:?}", result);

        let result =
            crate::slide_match(read_image("image/a.png"), read_image("image/b.png")).unwrap();

        println!("{:?}", result);

        let result =
            crate::simple_slide_match(read_image("image/a.png"), read_image("image/b.png"))
                .unwrap();

        println!("{:?}", result);
    }

    #[test]
    fn comparison_match() {
        let result =
            crate::slide_comparison(read_image("image/c.jpg"), read_image("image/d.jpg")).unwrap();
        println!("{:?}", result);
    }

    #[test]
    fn info() {
        println!(
            "current_dir: {}",
            std::env::current_dir().unwrap().display()
        );

        #[cfg(not(feature = "inline-model"))]
        println!("not feature inline-model");
    }

    #[test]
    fn classification_no_charset_returns_error() {
        // calling classification on a session without a charset should return an error, not panic
        let mut kaptchaocr = kaptchaocr_classification().unwrap();
        kaptchaocr.charset = None;

        let result = kaptchaocr.classification(read_image("image/1.png"));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no charset configured"));
    }

    #[test]
    fn ctc_greedy_decode_collapses_repeats() {
        // 3 timesteps, 2 classes (excluding blank), class1 appears consecutively → merged into one
        let logits = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3, 3]),
            vec![
                -2.0, 2.0, -1.0, // t0: class1
                -2.0, 2.0, -1.0, // t1: class1 (repeat)
                -1.0, -2.0, 2.0, // t2: class2
            ],
        )
        .unwrap();

        let decoded = ctc_greedy_decode(&logits);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].0, 1);
        assert_eq!(decoded[1].0, 2);
        assert!(decoded[0].1 > 0.5 && decoded[0].1 <= 1.0);
    }

    #[test]
    fn ctc_greedy_decode_blank_separates() {
        // blank(class 0) separates: class1, blank, class1 → two independent class1 entries
        let logits = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 4, 3]),
            vec![
                -2.0, 2.0, -1.0, // t0: class1
                -2.0, 2.0, -1.0, // t1: class1 (repeat)
                2.0, -2.0, -1.0, // t2: blank
                -2.0, 2.0, -1.0, // t3: class1 (new run)
            ],
        )
        .unwrap();

        let decoded = ctc_greedy_decode(&logits);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].0, 1);
        assert_eq!(decoded[1].0, 1);
    }

    #[test]
    fn ctc_greedy_decode_indices_collapses() {
        // index output: drop blank and merge consecutive repeats, but identical characters separated by blank are kept
        let indices =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 6]), vec![0, 1, 1, 0, 1, 2])
                .unwrap();

        let decoded = ctc_greedy_decode_indices(&indices);
        let classes: Vec<usize> = decoded.iter().map(|&(c, _)| c).collect();
        assert_eq!(classes, vec![1, 1, 2]);
        assert!(decoded.iter().all(|&(_, c)| c == 1.0));
    }

    #[test]
    fn ctc_greedy_decode_handles_batch3d_and_2d() {
        // [1, T, C] and [T, C] give identical results
        let data = vec![
            -2.0, 2.0, -1.0, // t0: class1
            -1.0, -2.0, 2.0, // t1: class2
        ];

        let d3 = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 3]), data.clone()).unwrap();
        let d2 = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), data).unwrap();

        let r3 = ctc_greedy_decode(&d3);
        let r2 = ctc_greedy_decode(&d2);

        assert_eq!(r3.len(), r2.len());
        assert_eq!(r3[0].0, r2[0].0);
        assert_eq!(r3[1].0, r2[1].0);
    }

    #[test]
    fn is_probability_output_distinguishes_probs_from_logits() {
        // each frame sums to 1 → probability output (PaddleOCR export)
        let probs = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 2, 3]),
            vec![0.1, 0.2, 0.7, 0.2, 0.2, 0.6],
        )
        .unwrap();
        assert!(is_probability_output(&probs));

        // raw logits (sum not 1) → not a probability output
        let logits = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 2, 3]),
            vec![-2.0, 2.0, -1.0, -1.0, -2.0, 2.0],
        )
        .unwrap();
        assert!(!is_probability_output(&logits));
    }

    #[test]
    fn ctc_greedy_decode_probability_uses_frame_probability() {
        // probability output: confidence uses the frame probability directly (rather than another softmax yielding ~0)
        let probs = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 3, 3]),
            vec![
                0.9, 0.05, 0.05, // t0: blank
                0.05, 0.9, 0.05, // t1: class1
                0.05, 0.1, 0.85, // t2: class2
            ],
        )
        .unwrap();

        let decoded = ctc_greedy_decode_probability(&probs);

        assert_eq!(
            decoded.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!((decoded[0].1 - 0.9).abs() < 1e-6);
        assert!((decoded[1].1 - 0.85).abs() < 1e-6);
    }

    #[test]
    fn map_charset_range_indices_maps_blank_unknown_and_chars() {
        let classes: Vec<String> = ["", "A", "B", "中"].iter().map(|v| v.to_string()).collect();
        let ranges: Vec<String> = ["B", "", "Z", "中"].iter().map(|v| v.to_string()).collect();

        // empty string = blank(0), unknown characters = None
        assert_eq!(
            map_charset_range_indices(&classes, &ranges),
            vec![Some(2), Some(0), None, Some(3)]
        );
    }

    #[test]
    fn ctc_greedy_decode_restricted_ignores_disallowed_argmax() {
        // the global argmax of t0/t2 is class3 (not in the allowed set), so restricted decoding takes
        // the highest allowed class; t1's blank is highest, separating the two class1s → two independent characters
        let logits = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 3, 4]),
            vec![
                0.0, 2.0, 0.0, 3.0, // t0: class3 highest globally, class1 second
                5.0, 0.0, 0.0, 0.0, // t1: blank
                0.0, 4.0, 0.0, 2.0, // t2: class1
            ],
        )
        .unwrap();

        let decoded = ctc_greedy_decode_restricted(&logits, &[1, 2]);

        assert_eq!(
            decoded.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![1, 1]
        );
    }

    #[test]
    fn ctc_greedy_decode_restricted_keeps_blank_when_dominant() {
        // blank is highest in every frame: no characters should be output even when allowed classes exist
        let logits = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 2, 3]),
            vec![5.0, 2.0, 0.0, 4.0, 1.0, 0.0],
        )
        .unwrap();

        assert!(ctc_greedy_decode_restricted(&logits, &[1, 2]).is_empty());
    }

    #[test]
    fn ctc_greedy_decode_restricted_collapses_repeats() {
        // allowed class class2 appears in two consecutive frames → merged into one; class3 is globally higher but not in the allowed set
        let logits = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, 2, 4]),
            vec![
                0.0, 0.0, 3.0, 4.0, // t0
                0.0, 0.0, 2.0, 1.0, // t1
            ],
        )
        .unwrap();

        let decoded = ctc_greedy_decode_restricted(&logits, &[2]);

        assert_eq!(decoded.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn word_model_probability_with_ranges_restricts_decode() {
        // the fine-tuned PP-OCRv6 rec is a word (CTC) model: charset_range should restrict the decoded characters
        let model = std::fs::read("model/finetuned/small_rec.onnx").unwrap();
        let charset: Charset = serde_json::from_str(
            &std::fs::read_to_string("model/finetuned/small_rec.json").unwrap(),
        )
        .unwrap();
        let kaptchaocr = KaptchaOcr::new(model, charset).unwrap();
        let image = || read_image("image/3.png");

        // regular recognition without a range (CTC greedy)
        let baseline = kaptchaocr.classification(image()).unwrap();
        assert!(!baseline.is_empty());

        // restricted to the baseline's charset: restricted decoding should match unrestricted decoding exactly
        let mut result = kaptchaocr
            .classification_probability_with_ranges(image(), CharsetRange::Other(baseline.clone()))
            .unwrap();
        assert_eq!(result.get_text(), baseline);

        let mut unique: Vec<char> = baseline.chars().collect();
        unique.sort_unstable();
        unique.dedup();
        assert!(result
            .probability
            .iter()
            .all(|row| row.len() == unique.len() + 1));

        // the PP-OCRv6 export already applies softmax: confidence should be a plausible probability, not ~1e-4 from another softmax
        let confidence = result.confidence.unwrap();
        assert!(
            confidence > 0.05,
            "confidence {confidence} looks like a re-softmaxed probability output"
        );

        // after removing the baseline's first character, it must not appear in the decoded result
        let removed = baseline.chars().next().unwrap();
        let filtered: String = baseline.chars().filter(|&c| c != removed).collect();
        let mut result = kaptchaocr
            .classification_probability_with_ranges(image(), CharsetRange::Other(filtered.clone()))
            .unwrap();
        let text = result.get_text().to_string();
        assert!(!text.contains(removed));
        assert!(text.chars().all(|c| filtered.contains(c)));

        // without a range, word models return a clear error rather than a wrong result
        let err = kaptchaocr.classification_probability(image()).unwrap_err();
        assert!(err.to_string().contains("charset_range"));
    }

    fn test_ocr() -> KaptchaOcr<'static> {
        let mut kaptchaocr = kaptchaocr_classification().unwrap();
        kaptchaocr.charset = Some(std::borrow::Cow::Owned(Charset {
            word: false,
            image: [-1, 32],
            channel: 1,
            charset: vec!["a".to_string(), "b".to_string()],
            mean: None,
            std: None,
            bgr: None,
        }));
        kaptchaocr
    }

    #[test]
    fn preprocess_channel1_shape_and_normalize() {
        // 1 channel: output (1, 1, H, W), normalization (x/255-0.5)/0.5
        let mut kaptchaocr = test_ocr();
        kaptchaocr.charset = Some(std::borrow::Cow::Owned(Charset {
            word: false,
            image: [4, 4],
            channel: 1,
            charset: vec!["a".to_string(), "b".to_string()],
            mean: None,
            std: None,
            bgr: None,
        }));

        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            4,
            4,
            image::Rgb([128, 128, 128]),
        ));

        let tensor = kaptchaocr
            .preprocess(image, false, &PreprocessOptions::default())
            .unwrap();

        assert_eq!(tensor.shape(), &[1, 1, 4, 4]);
        assert!((tensor[[0, 0, 0, 0]] - ((128.0 / 255.0) - 0.5) / 0.5).abs() < 1e-5);
    }

    #[test]
    fn preprocess_channel3_shape_and_channels() {
        // 3 channels: output (1, 3, H, W), with RGB correctly de-interleaved (no longer misreading interleaved bytes)
        let mut rgb = image::RgbImage::new(2, 1);
        rgb.put_pixel(0, 0, image::Rgb([10, 20, 30]));
        rgb.put_pixel(1, 0, image::Rgb([40, 50, 60]));

        let mut kaptchaocr = test_ocr();
        kaptchaocr.charset = Some(std::borrow::Cow::Owned(Charset {
            word: false,
            image: [2, 1],
            channel: 3,
            charset: vec!["a".to_string(), "b".to_string()],
            mean: None,
            std: None,
            bgr: None,
        }));

        let tensor = kaptchaocr
            .preprocess(
                image::DynamicImage::ImageRgb8(rgb),
                false,
                &PreprocessOptions::default(),
            )
            .unwrap();

        assert_eq!(tensor.shape(), &[1, 3, 1, 2]);

        let n = |v: u8| (v as f32 / 255.0 - 0.5) / 0.5;

        assert!((tensor[[0, 0, 0, 0]] - n(10)).abs() < 1e-5);
        assert!((tensor[[0, 1, 0, 0]] - n(20)).abs() < 1e-5);
        assert!((tensor[[0, 2, 0, 0]] - n(30)).abs() < 1e-5);
        assert!((tensor[[0, 0, 0, 1]] - n(40)).abs() < 1e-5);
        assert!((tensor[[0, 1, 0, 1]] - n(50)).abs() < 1e-5);
        assert!((tensor[[0, 2, 0, 1]] - n(60)).abs() < 1e-5);
    }

    #[test]
    fn preprocess_channel3_png_fix_white_fill() {
        // png_fix: transparent black areas filled with white; the 3-channel path does not panic
        let mut rgba = image::RgbaImage::new(1, 1);
        rgba.put_pixel(0, 0, image::Rgba([0, 0, 0, 0]));

        let mut kaptchaocr = test_ocr();
        kaptchaocr.charset = Some(std::borrow::Cow::Owned(Charset {
            word: false,
            image: [1, 1],
            channel: 3,
            charset: vec!["a".to_string(), "b".to_string()],
            mean: None,
            std: None,
            bgr: None,
        }));

        let tensor = kaptchaocr
            .preprocess(
                image::DynamicImage::ImageRgba8(rgba),
                true,
                &PreprocessOptions::default(),
            )
            .unwrap();

        assert_eq!(tensor.shape(), &[1, 3, 1, 1]);
        // transparent areas filled with white 255 → (1.0 - 0.5) / 0.5 = 1.0
        assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((tensor[[0, 1, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((tensor[[0, 2, 0, 0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn preprocess_options_denoise_binarize_deskew() {
        // preprocessing options run through and the output shape matches the model config
        let kaptchaocr = test_ocr();
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            64,
            64,
            image::Rgb([200, 200, 200]),
        ));

        let mut kaptchaocr = kaptchaocr;
        kaptchaocr.charset = Some(std::borrow::Cow::Owned(Charset {
            word: false,
            image: [64, 64],
            channel: 1,
            charset: vec!["a".to_string(), "b".to_string()],
            mean: None,
            std: None,
            bgr: None,
        }));

        let options = PreprocessOptions {
            denoise: true,
            binarize: true,
            deskew: true,
            resize: ResizeStrategy::Auto,
        };

        let tensor = kaptchaocr.preprocess(image, false, &options).unwrap();
        assert_eq!(tensor.shape(), &[1, 1, 64, 64]);
    }

    #[test]
    fn detection_with_options_returns_scores() {
        let kaptchaocr = kaptchaocr_detection().unwrap();
        let input = read_image("image/5.jpg");

        let with_score = kaptchaocr.detection_with_score(input.clone()).unwrap();
        let plain = kaptchaocr.detection(input).unwrap();

        assert_eq!(with_score.len(), plain.len());
        assert!(with_score.iter().all(|(_, s)| s.is_finite() && *s >= 0.0));
        assert_eq!(with_score[0].0.x1, plain[0].x1);
    }

    #[test]
    fn detection_thresholds_change_recall() {
        let kaptchaocr = kaptchaocr_detection().unwrap();
        let input = read_image("image/5.jpg");

        let loose = kaptchaocr
            .detection_with_options(input.clone(), 0.45, 0.001)
            .unwrap();
        let strict = kaptchaocr.detection_with_options(input, 0.45, 0.9).unwrap();

        assert!(loose.len() >= strict.len());
    }

    #[test]
    fn find_deskew_angle_returns_zero_for_straight() {
        // for an unrotated text line, the best angle should be near 0
        let mut gray = image::GrayImage::from_pixel(60, 40, image::Luma([255]));
        for x in 0..60 {
            for y in 15..25 {
                gray.put_pixel(x, y, image::Luma([0]));
            }
        }

        let angle = find_deskew_angle(&gray).to_degrees();
        assert!(angle.abs() <= 3.0, "expected near 0°, got {angle}°");
    }

    #[test]
    fn average_max_confidence_matches_get_confidence() {
        let prob = vec![vec![0.1f32, 0.9f32], vec![0.7f32, 0.3f32]];

        let mut cp = CharacterProbability {
            text: None,
            charset: vec!["a".to_string(), "b".to_string()],
            probability: prob.clone(),
            confidence: None,
        };

        assert_eq!(average_max_confidence(&prob), cp.get_confidence());
    }

    #[test]
    fn db_decode_finds_blob() {
        let (mw, mh) = (40u32, 20u32);
        let mut a = ndarray::Array4::<f32>::zeros((1, 1, mh as usize, mw as usize));

        // one rectangular high-probability region
        for y in 5..10 {
            for x in 10..20 {
                a[[0, 0, y, x]] = 1.0;
            }
        }

        let params = DbParams {
            thresh: 0.5,
            box_thresh: 0.4,
            unclip_ratio: 1.4,
            max_candidates: 100,
        };

        let id = |x: f32, y: f32| (x.clamp(0.0, mw as f32), y.clamp(0.0, mh as f32));

        let boxes = db_decode_prob_map(
            &a.view(),
            params,
            0.5,
            0.0,
            (mw as f32, mh as f32),
            (mw, mh),
            &id,
        )
        .unwrap();

        assert_eq!(boxes.len(), 1, "expected exactly one box, got {:?}", boxes);

        let (bbox, score) = boxes[0];
        assert!(score >= 0.4);
        assert!(bbox.x1 <= 10, "x1={}", bbox.x1);
        assert!(bbox.x2 >= 19, "x2={}", bbox.x2);
        assert!(bbox.y1 <= 5, "y1={}", bbox.y1);
        assert!(bbox.y2 >= 9, "y2={}", bbox.y2);
    }

    #[test]
    fn manifest_loads_and_builds_finetuned_models() {
        let m = crate::manifest::ModelManifest::load("model/models.json").unwrap();
        assert!(!m.models.is_empty());

        let rec = m
            .models
            .iter()
            .find(|x| x.task == "rec" && x.name == "ft-small")
            .expect("ft-small rec in manifest");
        let det = m
            .models
            .iter()
            .find(|x| x.task == "det" && x.name == "ft-small-det")
            .expect("ft-small-det det in manifest");

        let rec_model = rec.build().unwrap();
        let det_model = det.build().unwrap();

        assert_eq!(det_model.det_type(), DetType::Db);

        let img = read_image("image/1.png");
        let text = rec_model
            .classification_with_options(img, false, None)
            .unwrap();
        assert!(!text.is_empty(), "ft-small should recognize a sample");
    }

    #[test]
    fn manifest_profiles_and_finetuned_entries() {
        let m = crate::manifest::ModelManifest::load("model/models.json").unwrap();

        assert_eq!(m.profiles.get("fast").map(String::as_str), Some("ft-tiny"));
        assert_eq!(
            m.profiles.get("balanced").map(String::as_str),
            Some("ft-small")
        );
        assert_eq!(
            m.profiles.get("accurate").map(String::as_str),
            Some("ft-medium")
        );

        // the first rec entry in the manifest = the balanced default model.
        let first_rec = m.models.iter().find(|x| x.task == "rec").unwrap();
        assert_eq!(first_rec.name, "ft-small");

        for name in ["ft-tiny", "ft-small", "ft-medium"] {
            let entry = m.models.iter().find(|x| x.name == name).expect(name);
            assert_eq!(entry.task, "rec");
            assert!(
                std::path::Path::new(&entry.path).exists(),
                "missing {}",
                entry.path
            );
            let charset = entry.charset.as_deref().expect("charset path");
            assert!(std::path::Path::new(charset).exists(), "missing {charset}");
        }
    }
}
