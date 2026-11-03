use base64::prelude::*;
use clap::ArgGroup;
use clap::Parser;
use enable_ansi_support::enable_ansi_support;
use kaptchaocr::*;
use lru::LruCache;
use rmcp::model::CallToolResult;
use rmcp::model::ClientNotification;
use rmcp::model::ClientRequest;
use rmcp::model::Content;
use rmcp::model::ErrorCode;
use rmcp::model::InitializeResult;
use rmcp::model::JsonRpcError;
use rmcp::model::JsonRpcNotification;
use rmcp::model::JsonRpcRequest;
use rmcp::model::JsonRpcResponse;
use rmcp::model::JsonRpcVersion2_0;
use rmcp::model::ListToolsResult;
use rmcp::ErrorData;
use salvo::catcher::Catcher;
use salvo::http::request;
use salvo::http::ReqBody;
use salvo::http::ResBody;
use salvo::oapi::extract::JsonBody;
use salvo::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::net::IpAddr;
use std::num::NonZero;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::sync::Semaphore;
use tokio::task::spawn_blocking;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;
use tracing_subscriber::EnvFilter;

static ARGS: OnceLock<Args> = OnceLock::new();
static OCR: OnceLock<KaptchaOcr> = OnceLock::new();
static DET: OnceLock<KaptchaOcr> = OnceLock::new();
static CACHE: LazyLock<Mutex<LruCache<String, Vec<String>>>> =
    LazyLock::new(|| Mutex::new(LruCache::new(NonZero::new(64).unwrap())));

/// Name of the currently loaded recognition model (manifest entry name or built-in name), used in the `/solve` response.
static MODEL_NAME: OnceLock<String> = OnceLock::new();

/// Records the current model name (repeated calls are ignored).
fn set_model_name(name: impl Into<String>) {
    let _ = MODEL_NAME.set(name.into());
}

/// Semaphore for concurrent ONNX inference, defaults to the number of CPU cores (`--max-concurrency`).
static INFERENCE_SEM: OnceLock<Arc<Semaphore>> = OnceLock::new();
/// Semaphore capacity (set together with INFERENCE_SEM), used by `/metrics`.
static INFERENCE_CAP: OnceLock<usize> = OnceLock::new();

/// Global request count / error count / queue length (used by `/metrics`).
static REQ_COUNT: AtomicU64 = AtomicU64::new(0);
static ERR_COUNT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TASKS: AtomicU64 = AtomicU64::new(0);

/// Per-IP sliding-window rate limiting (`--rate-limit`; 0 disables it).
static RATE_LIMIT: LazyLock<Mutex<HashMap<IpAddr, Vec<Instant>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Runs blocking ONNX inference under the semaphore and counts requests / errors / active tasks.
async fn run_blocking_inference<T, F>(f: F) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
{
    let permit = match INFERENCE_SEM.get() {
        Some(sem) => Some(sem.clone().acquire_owned().await?),
        None => None,
    };

    ACTIVE_TASKS.fetch_add(1, Ordering::SeqCst);
    REQ_COUNT.fetch_add(1, Ordering::SeqCst);

    let result = spawn_blocking(f).await;

    ACTIVE_TASKS.fetch_sub(1, Ordering::SeqCst);
    drop(permit);

    match result {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => {
            ERR_COUNT.fetch_add(1, Ordering::SeqCst);
            Err(e)
        }
        Err(e) => {
            ERR_COUNT.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::anyhow!("blocking task failed: {e}"))
        }
    }
}

/// Checks whether the request carries the correct API key.
fn check_auth(req: &salvo::http::Request) -> bool {
    let Some(expected) = ARGS.get().and_then(|a| a.api_key.as_ref()) else {
        return true;
    };

    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    if let Some(auth) = auth_header {
        if let Some(token) = auth.strip_prefix("Bearer ") {
            if token == expected {
                return true;
            }
        }
    }

    req.headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == expected)
}

/// Rate limiting + authentication middleware.
#[handler]
async fn auth_and_rate_limit(
    req: &mut salvo::Request,
    depot: &mut Depot,
    res: &mut Response,
    ctrl: &mut FlowCtrl,
) {
    // Authentication (/status, /healthz, /metrics and docs are exempt)
    let path = req.uri().path().to_string();
    let public_paths = ["/status", "/healthz", "/metrics", "/docs", "/docs.json"];
    let is_public = public_paths.iter().any(|p| path.starts_with(p));

    if !is_public && !check_auth(req) {
        res.render(Json(APIResponse::<Option<String>> {
            code: 401,
            msg: "unauthorized: missing or invalid API key".to_string(),
            data: None,
        }));
        ctrl.skip_rest();
        return;
    }

    // Rate limiting
    if let Some(limit) = ARGS
        .get()
        .and_then(|a| (a.rate_limit > 0).then_some(a.rate_limit))
    {
        let ip = req.remote_addr().clone().into_std().map(|a| a.ip());
        let window = Duration::from_secs(10);

        if let Some(ip) = ip {
            let mut map = RATE_LIMIT.lock().await;
            let hits = map.entry(ip).or_default();

            // Drop entries outside the window
            hits.retain(|t| t.elapsed() < window);

            if hits.len() >= limit {
                res.render(Json(APIResponse::<Option<String>> {
                    code: 429,
                    msg: "rate limit exceeded, please slow down".to_string(),
                    data: None,
                }));
                ctrl.skip_rest();
                return;
            }

            hits.push(Instant::now());
        }
    }

    // Continue to the next handler
    ctrl.call_next(req, depot, res).await;
}

#[derive(Parser, Debug, Clone)]
#[clap(
    group(
        ArgGroup::new("ocr_old")
            .args(&["ocr", "old"])
            .multiple(false)
    ),
    group(
        ArgGroup::new("mcp_only_mcp")
            .args(&["mcp", "only_mcp"])
            .multiple(false)
))]
struct Args {
    /// Listen address.
    #[arg(long, default_value_t = { "0.0.0.0:8000".to_string() })]
    address: String,

    /// Enable MCP protocol support; mutually exclusive with only_mcp.
    #[arg(long)]
    mcp: bool,

    /// Enable the MCP protocol only, without the regular routes; mutually exclusive with mcp.
    #[arg(long)]
    only_mcp: bool,

    /// Enable content recognition; mutually exclusive with old.
    #[arg(long)]
    ocr: bool,

    /// Enable content recognition with the legacy model; mutually exclusive with ocr.
    #[arg(long)]
    old: bool,

    /// Enable object detection.
    #[arg(long)]
    det: bool,

    /// Enable slider and gap recognition.
    #[arg(long)]
    slide: bool,

    /// Global default charset for probability-based recognition.  
    /// If the API does not provide a charset, this value is used.  
    /// Values 0~7 select a built-in charset;
    /// any other value is a custom charset, e.g. "0123456789+-x/=".  
    /// If unset, the full charset is used without restriction.
    #[arg(long)]
    ocr_charset_range: Option<String>,

    /// Path to the content recognition model and its charset.
    /// If the inline-model feature is enabled (the default), you can ignore this option unless you want a custom model.
    /// The model and charset paths must share the same base name, e.g. `model/stock_common.onnx` + `model/stock_common.json`.
    #[arg(long, default_value_t = { "model/stock_common.onnx".to_string() })]
    ocr_path: String,

    /// Path to the object detection model.
    /// If the inline-model feature is enabled (the default), you can ignore this option unless you want a custom model.
    #[arg(long, default_value_t = { "model/stock_det.onnx".to_string() })]
    det_path: String,

    /// Provide your domain to automatically obtain an SSL certificate,
    /// i.e. HTTPS support.
    #[arg(long)]
    acme: Option<String>,

    /// Maximum number of concurrent ONNX inference tasks, defaults to the number of CPU cores.
    /// 0 uses the number of CPU cores.
    #[arg(long, default_value_t = 0)]
    max_concurrency: usize,

    /// Allowed requests per IP per 10 seconds; 0 means unlimited.
    #[arg(long, default_value_t = 0)]
    rate_limit: usize,

    /// Optional API key; when set, all REST / MCP requests must carry
    /// an `Authorization: Bearer <key>` or `x-api-key: <key>` header.
    #[arg(long)]
    api_key: Option<String>,

    /// ORT `IntraOpNumThreads` (threads used per inference); 0 uses the ORT default.
    #[arg(long, default_value_t = 0)]
    num_threads: usize,

    /// Optional TOML config file; can set model paths / charset / thread count, etc.
    /// Command-line arguments take precedence.
    #[arg(long)]
    config: Option<String>,

    /// Optional model manifest file (`model/models.json`); loads rec / det models from the manifest,
    /// so any ONNX model can be plugged in without code changes. Takes precedence over --ocr-path / --det-path.
    #[arg(long)]
    manifest: Option<String>,

    /// Named profile in the model manifest (the `profiles` field of `model/models.json`),
    /// e.g. fast / balanced / accurate. Must be used together with --manifest.
    #[arg(long)]
    profile: Option<String>,

    /// Select a rec model by name from the manifest (defaults to the first rec entry),
    /// e.g. --rec-model ft-small. Takes precedence over --profile.
    #[arg(long)]
    rec_model: Option<String>,

    /// Select a det model by name from the manifest (defaults to the first det entry),
    /// e.g. --det-model ppocrv6-tiny-det.
    #[arg(long)]
    det_model: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct OCRRequest {
    /// Image to recognize, base64-encoded.
    image: String,

    /// If png_fix is true, PNG images with a transparent (black) background are supported.
    png_fix: Option<bool>,

    /// Whether to return probability data.
    probability: Option<bool>,

    /// Restrict the character range for this OCR call only.
    /// Values 0 to 7 select a built-in charset;
    /// any other value is a custom charset, e.g. `"0123456789+-x/="`.
    charset_range: Option<String>,

    /// Color filter, e.g. `red`, `["red", "blue"]` or `[[[0, 50, 50], [10, 255, 255]]]`.
    color_filter: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct OCRResponse {
    /// Recognition result.
    text: String,

    /// Probability data.
    probability: Option<Vec<Vec<f32>>>,

    /// Charset corresponding to the probability matrix (stable order).
    charset: Option<Vec<String>>,

    /// Overall recognition confidence (average of the per-character max probabilities); only returned when probabilities are requested.
    confidence: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SolveRequest {
    /// Image to recognize, base64-encoded.
    image: String,

    /// Math mode: `auto` (default; returns a solution when the text is an expression) or `off`.
    math: Option<String>,

    /// If png_fix is true, PNG images with a transparent (black) background are supported.
    png_fix: Option<bool>,

    /// Whether to return probability data.
    probability: Option<bool>,

    /// Restrict the character range for this call only; same semantics as `charset_range` in `/ocr`.
    charset_range: Option<String>,

    /// Color filter, e.g. `red`, `["red", "blue"]` or `[[[0, 50, 50], [10, 255, 255]]]`.
    color_filter: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct MathAnswer {
    /// Normalized expression, e.g. `5+?=8`.
    expression: String,

    /// Computed result.
    result: f64,

    /// Answer that can be entered directly, e.g. `"3"`.
    answer: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SolveResponse {
    /// Recognition result.
    text: String,

    /// Expression solution (null when `math` is off or the text is not an expression).
    math: Option<MathAnswer>,

    /// Probability data.
    probability: Option<Vec<Vec<f32>>>,

    /// Charset corresponding to the probability matrix (stable order).
    charset: Option<Vec<String>>,

    /// Overall recognition confidence; only returned when probabilities are requested.
    confidence: Option<f64>,

    /// Name of the currently loaded model (`--manifest` entry name or built-in name).
    model: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SolveClickRequest {
    /// Click captcha image, base64-encoded.
    image: String,

    /// Click prompt, e.g. a Chinese "click in order: Ming, Tian, Hui" prompt or `click in order: A B C`.
    /// When omitted, only the recognized character boxes are returned (`points` is empty) and the client does the matching itself.
    prompt: Option<String>,

    /// Minimum score for a detected box; defaults to 0.3.
    score_threshold: Option<f32>,

    /// Restrict the recognized character range for this call only; same semantics as `charset_range` in `/ocr`.
    charset_range: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct ClickBoxResponse {
    /// Bounding box coordinates.
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,

    /// Character recognized in this box.
    ch: String,

    /// Detection score.
    score: f32,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct ClickPointResponse {
    /// Click x coordinate (box center, relative to the captcha image).
    x: f32,

    /// Click y coordinate (box center, relative to the captcha image).
    y: f32,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SolveClickResponse {
    /// Parsed target characters (in click order); null when `prompt` is not provided.
    prompt: Option<String>,

    /// Click coordinates ordered by the prompt (box center, relative to the captcha image).
    points: Vec<ClickPointResponse>,

    /// All detected character boxes (in reading order: top to bottom, then left to right).
    boxes: Vec<ClickBoxResponse>,

    /// Characters from the prompt that could not be matched to a box.
    unmatched: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct DETRequest {
    /// Image to recognize, base64-encoded.
    image: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct DETResponse {
    /// Bounding box coordinates, e.g. `[[x1, y1, x2, y2], [x1, y1, x2, y2]]`.
    bboxes: Vec<Vec<u32>>,

    /// Confidence scores, one for each bounding box.
    scores: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SlideMatchRequest {
    /// Slider image, base64-encoded.
    target_image: String,

    /// Background image, base64-encoded.
    background_image: String,

    /// Whether this is a simple slider.
    simple_target: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SlideMatchResponse {
    /// Target position coordinates, e.g. `[x1, y1, x2, y2]`.
    target: Vec<u32>,

    /// x offset of the transparent part.
    target_x: u32,

    /// y offset of the transparent part.
    target_y: u32,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SlideComparisonRequest {
    /// Slider image, base64-encoded.
    target_image: String,

    /// Background image, base64-encoded.
    background_image: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SlideComparisonResponse {
    /// Gap x offset.
    x: u32,

    /// Gap y offset.
    y: u32,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SlideSolveRequest {
    /// Slider image, base64-encoded.
    target_image: String,

    /// Background image, base64-encoded.
    background_image: String,

    /// Whether this is a simple slider.
    simple_target: Option<bool>,

    /// Slider handle start x (page coordinates, optional); when provided, the drag distance is returned.
    handle_x: Option<f64>,

    /// Slider handle start y (page coordinates, optional; only used for client-side replay).
    handle_y: Option<f64>,

    /// Calibration offset (pixels), added to the gap x.
    offset: Option<f64>,

    /// Drag duration (milliseconds); derived from the distance by default.
    duration_ms: Option<f64>,

    /// Random seed, used to reproduce the trajectory.
    seed: Option<u64>,

    /// Whether to return the trajectory; defaults to true.
    trajectory: Option<bool>,
}

/// Trajectory sample point (offset and time relative to the slider start).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SolveTrajectoryPoint {
    /// x offset from the start position (pixels).
    x: f64,

    /// y offset from the start position (pixels).
    y: f64,

    /// Time relative to the start (milliseconds).
    t: f64,
}

impl From<kaptchaocr::trajectory::TrajectoryPoint> for SolveTrajectoryPoint {
    fn from(point: kaptchaocr::trajectory::TrajectoryPoint) -> Self {
        Self {
            x: point.x,
            y: point.y,
            t: point.t,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct SlideSolveResponse {
    /// Target position coordinates, e.g. `[x1, y1, x2, y2]`.
    target: Vec<u32>,

    /// x offset of the transparent part.
    target_x: u32,

    /// y offset of the transparent part.
    target_y: u32,

    /// Gap x coordinate (match position, before the calibration offset).
    gap_x: u32,

    /// Gap y coordinate.
    gap_y: u32,

    /// Drag distance (pixels); when handle_x is provided, the distance between the gap and the handle.
    distance: f64,

    /// Total drag duration (milliseconds).
    duration_ms: f64,

    /// Drag trajectory (relative offsets); empty when `trajectory=false`.
    trajectory: Vec<SolveTrajectoryPoint>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct StatusResponse {
    /// Service status.
    service_status: String,

    /// Enabled features.
    enabled_features: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct APIResponse<T> {
    code: u16,
    msg: String,
    data: Option<T>,
}

/// Parses the `color_filter` request field.
fn parse_color_filter(value: Option<Value>) -> anyhow::Result<Option<ColorFilter>> {
    value
        .map(|v| serde_json::from_value::<ColorFilter>(v).map_err(Into::into))
        .transpose()
}

/// Parses the `charset_range` request field (0~7 selects a built-in charset, anything else is a custom charset).
async fn resolve_charset_range(value: Option<&str>) -> anyhow::Result<Option<CharsetRange>> {
    let Some(value) = value else {
        return Ok(None);
    };

    let ocr_charset_range = match value {
        "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" => CharsetRange::from(value.parse::<i32>()?),
        value => CharsetRange::from(value),
    };

    Ok(Some(CharsetRange::Charset(
        CACHE
            .lock()
            .await
            .get_or_insert(value.to_string(), || {
                OCR.get().unwrap().calc_ranges(ocr_charset_range)
            })
            .clone(),
    )))
}

/// Performs content recognition (shared by `/ocr` and `/solve`).
async fn classify(
    image: &str,
    png_fix: bool,
    probability: bool,
    charset_range: Option<CharsetRange>,
    color_filter: Option<ColorFilter>,
) -> anyhow::Result<(
    String,
    Option<Vec<Vec<f32>>>,
    Option<Vec<String>>,
    Option<f64>,
)> {
    let image = BASE64_STANDARD.decode(image)?;

    if charset_range.is_some() || probability {
        let mut result = run_blocking_inference(move || {
            OCR.get().unwrap().classification_probability_with_options(
                image,
                png_fix,
                color_filter,
                charset_range,
            )
        })
        .await?;

        let probability = probability.then_some(result.probability.clone());
        let charset = probability.as_ref().map(|_| result.charset.clone());
        let confidence = probability
            .as_ref()
            .map(|_| result.confidence.unwrap_or_default());
        let text = result.get_text().to_string();

        Ok((text, probability, charset, confidence))
    } else {
        let text = run_blocking_inference(move || {
            OCR.get()
                .unwrap()
                .classification_with_options(image, png_fix, color_filter)
        })
        .await?;

        Ok((text, None, None, None))
    }
}

#[endpoint(responses((status_code = 200, body = APIResponse<OCRResponse>)))]
async fn route_ocr(req: JsonBody<OCRRequest>, res: &mut Response) -> anyhow::Result<()> {
    let png_fix = req.png_fix.unwrap_or_default();
    let probability = req.probability.unwrap_or_default();
    let color_filter = parse_color_filter(req.color_filter.clone())?;
    let charset_range = resolve_charset_range(req.charset_range.as_deref()).await?;

    let (text, probability, charset, confidence) = classify(
        &req.image,
        png_fix,
        probability,
        charset_range,
        color_filter,
    )
    .await?;

    let response = OCRResponse {
        text,
        probability,
        charset,
        confidence,
    };

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(response),
    };

    debug!(
        "ocr response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));

    Ok(())
}

#[endpoint(responses((status_code = 200, body = APIResponse<SolveResponse>)))]
async fn route_solve(req: JsonBody<SolveRequest>, res: &mut Response) -> anyhow::Result<()> {
    let png_fix = req.png_fix.unwrap_or_default();
    let probability = req.probability.unwrap_or_default();
    let color_filter = parse_color_filter(req.color_filter.clone())?;
    let charset_range = resolve_charset_range(req.charset_range.as_deref()).await?;

    let (text, probability, charset, confidence) = classify(
        &req.image,
        png_fix,
        probability,
        charset_range,
        color_filter,
    )
    .await?;

    let math = if req.math.as_deref().unwrap_or("auto") == "off" {
        None
    } else {
        kaptchaocr::math::solve(&text).map(|solved| MathAnswer {
            expression: solved.expression,
            result: solved.result,
            answer: solved.answer,
        })
    };

    let response = SolveResponse {
        text,
        math,
        probability,
        charset,
        confidence,
        model: MODEL_NAME.get().cloned(),
    };

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(response),
    };

    debug!(
        "solve response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));

    Ok(())
}

/// `/solve/click`: detects character boxes, recognizes each box, and orders the clicks according to the prompt.
#[endpoint(responses((status_code = 200, body = APIResponse<SolveClickResponse>)))]
async fn route_solve_click(
    req: JsonBody<SolveClickRequest>,
    res: &mut Response,
) -> anyhow::Result<()> {
    let image_bytes = BASE64_STANDARD.decode(&req.image)?;
    let score_threshold = req.score_threshold.unwrap_or(0.3);
    let charset_range = resolve_charset_range(req.charset_range.as_deref()).await?;

    let targets = req
        .prompt
        .as_deref()
        .map(kaptchaocr::click::parse_prompt)
        .filter(|prompt| !prompt.is_empty());
    let prompt_text = targets
        .as_ref()
        .map(|prompt| prompt.iter().collect::<String>());

    let (points, boxes, unmatched) = run_blocking_inference(move || {
        let image = image::load_from_memory(&image_bytes)?;
        let detections = DET.get().unwrap().detection_with_score(&image_bytes)?;

        let mut hits: Vec<(BBox, f32, String)> = Vec::new();

        for (bbox, score) in detections {
            if score < score_threshold {
                continue;
            }

            let x1 = bbox.x1.min(image.width().saturating_sub(1));
            let y1 = bbox.y1.min(image.height().saturating_sub(1));
            let x2 = bbox.x2.min(image.width());
            let y2 = bbox.y2.min(image.height());

            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            // Single-character recognition needs a little context: a tight crop (e.g. a thin `l`) can be recognized as an empty string.
            // Expand by 15% of the larger side, clamped to 2–4 px (to avoid catching neighboring characters, e.g. in row-arranged captchas);
            // the detection box itself is unchanged.
            let margin = ((x2 - x1).max(y2 - y1) as f32 * 0.15).round() as u32;
            let margin = margin.clamp(2, 4);
            let crop_x1 = x1.saturating_sub(margin);
            let crop_y1 = y1.saturating_sub(margin);
            let crop_x2 = (x2 + margin).min(image.width());
            let crop_y2 = (y2 + margin).min(image.height());

            let crop = image.crop_imm(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);
            let mut png = std::io::Cursor::new(Vec::new());
            crop.write_to(&mut png, image::ImageFormat::Png)?;

            let text = match &charset_range {
                Some(range) => {
                    let mut result = OCR.get().unwrap().classification_probability_with_options(
                        png.into_inner(),
                        false,
                        None,
                        Some(range.clone()),
                    )?;
                    result.get_text().to_string()
                }
                None => {
                    OCR.get()
                        .unwrap()
                        .classification_with_options(png.into_inner(), false, None)?
                }
            };

            hits.push((bbox, score, text));
        }

        // Returned boxes are in reading order; the indexes used for matching keep detection order.
        let mut sorted: Vec<usize> = (0..hits.len()).collect();
        sorted.sort_by_key(|&i| (hits[i].0.y1, hits[i].0.x1));

        let boxes: Vec<ClickBoxResponse> = sorted
            .iter()
            .map(|&i| ClickBoxResponse {
                x1: hits[i].0.x1,
                y1: hits[i].0.y1,
                x2: hits[i].0.x2,
                y2: hits[i].0.y2,
                ch: hits[i].2.clone(),
                score: hits[i].1,
            })
            .collect();

        let recognized: Vec<String> = hits.iter().map(|(_, _, text)| text.clone()).collect();

        let (order, unmatched) = match &targets {
            Some(prompt) => {
                let matched = kaptchaocr::click::match_prompt(prompt, &recognized);
                (matched.order, matched.unmatched)
            }
            None => (Vec::new(), Vec::new()),
        };

        let points: Vec<ClickPointResponse> = order
            .iter()
            .map(|&i| {
                let b = hits[i].0;
                ClickPointResponse {
                    x: (b.x1 + b.x2) as f32 / 2.0,
                    y: (b.y1 + b.y2) as f32 / 2.0,
                }
            })
            .collect();

        Ok((points, boxes, unmatched))
    })
    .await?;

    let response = SolveClickResponse {
        prompt: prompt_text,
        points,
        boxes,
        unmatched: unmatched.iter().map(|c| c.to_string()).collect(),
    };

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(response),
    };

    debug!(
        "solve click response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));

    Ok(())
}

#[endpoint(responses((status_code = 200, body = APIResponse<DETResponse>)))]
async fn route_det(req: JsonBody<DETRequest>, res: &mut Response) -> anyhow::Result<()> {
    let image = BASE64_STANDARD.decode(&req.image)?;
    let detections =
        run_blocking_inference(|| DET.get().unwrap().detection_with_score(image)).await?;
    let bboxes = detections
        .iter()
        .map(|(b, _)| vec![b.x1, b.y1, b.x2, b.y2])
        .collect();
    let scores = detections.into_iter().map(|(_, s)| s).collect();
    let response = DETResponse { bboxes, scores };

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(response),
    };

    debug!(
        "det response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));

    Ok(())
}

/// Copies the sub-route response body into the outer response; returns a generic error if the sub-route fails.
fn render_sub_response(res: &mut Response, sub_res: &mut Response) -> anyhow::Result<()> {
    match sub_res.take_body() {
        ResBody::Once(v) => {
            let value: serde_json::Value = serde_json::from_slice(&v)?;
            res.render(Json(value));
        }
        ResBody::Error(v) => {
            error!("sub-route error: {}", v.to_string());
            res.render(Json(APIResponse::<Option<String>> {
                code: 500,
                msg: "internal server error".to_string(),
                data: None,
            }));
        }
        _ => {
            res.render(Json(APIResponse::<Option<String>> {
                code: 500,
                msg: "internal server error".to_string(),
                data: None,
            }));
        }
    }

    Ok(())
}

/// Recognition via multipart/form-data upload; fields: `file` plus optional `png_fix` /
/// `probability` / `charset_range` / `color_filter`.
#[handler]
async fn route_ocr_upload(req: &mut Request, res: &mut Response) -> anyhow::Result<()> {
    let form = req
        .form_data()
        .await
        .map_err(|e| anyhow::anyhow!("invalid multipart form: {e}"))?;

    let file = form
        .files
        .flat_iter()
        .next()
        .map(|(_, f)| f)
        .ok_or_else(|| anyhow::anyhow!("missing image file field"))?;

    let ocr_request = OCRRequest {
        image: BASE64_STANDARD.encode(std::fs::read(file.path())?),
        png_fix: form.fields.get("png_fix").map(|s| s == "true" || s == "1"),
        probability: form
            .fields
            .get("probability")
            .map(|s| s == "true" || s == "1"),
        charset_range: form.fields.get("charset_range").cloned(),
        color_filter: form
            .fields
            .get("color_filter")
            .and_then(|s| serde_json::from_str::<Value>(s).ok()),
    };

    let mut sub_req = salvo::Request::new();
    sub_req.add_header("content-type", "application/json", true)?;
    sub_req.replace_body(ReqBody::Once(bytes::Bytes::from(serde_json::to_string(
        &ocr_request,
    )?)));

    let mut sub_res = salvo::Response::new();
    let mut sub_ctrl = salvo::FlowCtrl::default();

    route_ocr
        .handle(&mut sub_req, &mut Depot::new(), &mut sub_res, &mut sub_ctrl)
        .await;

    render_sub_response(res, &mut sub_res)
}

/// Object detection via multipart/form-data upload; field: `file`.
#[handler]
async fn route_det_upload(req: &mut Request, res: &mut Response) -> anyhow::Result<()> {
    let form = req
        .form_data()
        .await
        .map_err(|e| anyhow::anyhow!("invalid multipart form: {e}"))?;

    let file = form
        .files
        .flat_iter()
        .next()
        .map(|(_, f)| f)
        .ok_or_else(|| anyhow::anyhow!("missing image file field"))?;

    let det_request = DETRequest {
        image: BASE64_STANDARD.encode(std::fs::read(file.path())?),
    };

    let mut sub_req = salvo::Request::new();
    sub_req.add_header("content-type", "application/json", true)?;
    sub_req.replace_body(ReqBody::Once(bytes::Bytes::from(serde_json::to_string(
        &det_request,
    )?)));

    let mut sub_res = salvo::Response::new();
    let mut sub_ctrl = salvo::FlowCtrl::default();

    route_det
        .handle(&mut sub_req, &mut Depot::new(), &mut sub_res, &mut sub_ctrl)
        .await;

    render_sub_response(res, &mut sub_res)
}

#[endpoint(responses((status_code = 200, body = APIResponse<SlideMatchResponse>)))]
async fn route_slide_match(
    req: JsonBody<SlideMatchRequest>,
    res: &mut Response,
) -> anyhow::Result<()> {
    let target_image = BASE64_STANDARD.decode(&req.target_image)?;
    let background_image = BASE64_STANDARD.decode(&req.background_image)?;

    let result = run_blocking_inference(move || {
        if req.simple_target.unwrap_or_default() {
            simple_slide_match(target_image, background_image)
        } else {
            slide_match(target_image, background_image)
        }
    })
    .await?;

    let response = SlideMatchResponse {
        target: vec![result.x1, result.y1, result.x2, result.y2],
        target_x: result.target_x,
        target_y: result.target_y,
    };

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(response),
    };

    debug!(
        "slide match response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));

    Ok(())
}

#[endpoint(responses((status_code = 200, body = APIResponse<SlideComparisonResponse>)))]
async fn route_slide_comparison(
    req: JsonBody<SlideComparisonRequest>,
    res: &mut Response,
) -> anyhow::Result<()> {
    let target_image = BASE64_STANDARD.decode(&req.target_image)?;
    let background_image = BASE64_STANDARD.decode(&req.background_image)?;
    let result =
        run_blocking_inference(|| slide_comparison(target_image, background_image)).await?;
    let response = SlideComparisonResponse {
        x: result.0,
        y: result.1,
    };

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(response),
    };

    debug!(
        "slide comparison response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));

    Ok(())
}

#[endpoint(responses((status_code = 200, body = APIResponse<SlideSolveResponse>)))]
async fn route_slide_solve(
    req: JsonBody<SlideSolveRequest>,
    res: &mut Response,
) -> anyhow::Result<()> {
    let target_image = BASE64_STANDARD.decode(&req.target_image)?;
    let background_image = BASE64_STANDARD.decode(&req.background_image)?;
    let simple_target = req.simple_target.unwrap_or_default();

    let result = run_blocking_inference(move || {
        if simple_target {
            simple_slide_match(target_image, background_image)
        } else {
            slide_match(target_image, background_image)
        }
    })
    .await?;

    let offset = req.offset.unwrap_or_default();
    let gap_x = result.x1 as f64 + offset;
    let distance = match req.handle_x {
        Some(handle_x) => (gap_x - handle_x).max(0.0),
        None => gap_x.max(0.0),
    };

    let duration_ms = req
        .duration_ms
        .unwrap_or_else(|| kaptchaocr::trajectory::default_duration_ms(distance))
        .clamp(80.0, 5000.0);

    let seed = req.seed.unwrap_or_else(random_seed);

    let trajectory = if req.trajectory.unwrap_or(true) {
        kaptchaocr::trajectory::human_trajectory(distance, duration_ms, seed)
            .into_iter()
            .map(SolveTrajectoryPoint::from)
            .collect()
    } else {
        Vec::new()
    };

    let response = SlideSolveResponse {
        target: vec![result.x1, result.y1, result.x2, result.y2],
        target_x: result.target_x,
        target_y: result.target_y,
        gap_x: result.x1,
        gap_y: result.y1,
        distance,
        duration_ms,
        trajectory,
    };

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(response),
    };

    debug!(
        "slide solve response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));

    Ok(())
}

/// Generates a random seed from the system time when `seed` is not specified.
fn random_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5EED)
}

#[endpoint(responses((status_code = 200, body = APIResponse<StatusResponse>)))]
async fn route_status(res: &mut Response) {
    let args = ARGS.get().unwrap();
    let mut enabled_features = Vec::new();

    if args.ocr || args.old {
        enabled_features.push("ocr".to_string());
    }

    if args.det {
        enabled_features.push("det".to_string());
    }

    if args.slide {
        enabled_features.push("slide".to_string());
    };

    if args.mcp {
        enabled_features.push("mcp".to_string());
    };

    if args.only_mcp {
        enabled_features.push("only_mcp".to_string());
    };

    let response = StatusResponse {
        service_status: "running".to_string(),
        enabled_features,
    };

    let response = APIResponse {
        code: 200,
        msg: "success".to_string(),
        data: Some(response),
    };

    debug!(
        "status response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    res.render(Json(response));
}

/// Liveness probe: returns 200 as long as the process is alive.
#[handler]
async fn route_healthz(res: &mut Response) {
    res.render(Text::Plain("ok"));
}

/// Readiness probe and Prometheus text metrics.
#[handler]
async fn route_metrics(res: &mut Response) {
    let req = REQ_COUNT.load(Ordering::SeqCst);
    let errs = ERR_COUNT.load(Ordering::SeqCst);
    let active = ACTIVE_TASKS.load(Ordering::SeqCst);

    let queue_depth = match INFERENCE_SEM.get() {
        Some(sem) => sem.available_permits(),
        None => 0,
    };

    let max_concurrency = INFERENCE_CAP.get().copied().unwrap_or(0);

    let body = format!(
        "# TYPE kaptchaocr_requests_total counter\n\
         kaptchaocr_requests_total {req}\n\
         # TYPE kaptchaocr_errors_total counter\n\
         kaptchaocr_errors_total {errs}\n\
         # TYPE kaptchaocr_active_tasks gauge\n\
         kaptchaocr_active_tasks {active}\n\
         # TYPE kaptchaocr_queue_depth gauge\n\
         kaptchaocr_queue_depth {queue_depth}\n\
         # TYPE kaptchaocr_max_concurrency gauge\n\
         kaptchaocr_max_concurrency {max_concurrency}\n"
    );

    res.render(Text::Plain(body));
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum JsonRpcRequestOrNotification {
    Request(JsonRpcRequest<ClientRequest>),
    Notification(JsonRpcNotification<ClientNotification>),
}

/// Handles a single JSON-RPC message; notifications return `Ok(None)`, everything else returns a response object.
async fn handle_single_rpc(
    req_body: JsonRpcRequestOrNotification,
    depot: &mut Depot,
) -> anyhow::Result<Option<Value>> {
    let req_body = if let JsonRpcRequestOrNotification::Request(v) = req_body {
        v
    } else {
        debug!("received notification");
        return Ok(None);
    };

    let response: Value = match &req_body.request {
        ClientRequest::InitializeRequest(_) => {
            debug!("initialize");

            static RESULT: std::sync::LazyLock<InitializeResult> = std::sync::LazyLock::new(|| {
                serde_json::from_str(include_str!("../initialize.json")).unwrap()
            });

            serde_json::to_value(JsonRpcResponse {
                jsonrpc: JsonRpcVersion2_0,
                id: req_body.id,
                result: RESULT.clone(),
            })?
        }
        ClientRequest::ListToolsRequest(_) => {
            debug!("list tools");

            static RESULT: std::sync::LazyLock<ListToolsResult> = std::sync::LazyLock::new(|| {
                let args = ARGS.get().expect("ARGS must be set before serving");

                let mut value: serde_json::Value =
                    serde_json::from_str(include_str!("../list.json"))
                        .expect("list.json must be valid JSON");

                if let Some(tools) = value["tools"].as_array_mut() {
                    tools.retain(|tool| {
                        let name = tool["name"].as_str().unwrap_or_default();
                        match name {
                            "ocr" => args.ocr || args.old,
                            "det" => args.det,
                            "slide_match" | "slide_comparison" => args.slide,
                            _ => false,
                        }
                    });
                }

                serde_json::from_value(value).expect("filtered list.json must deserialize")
            });

            serde_json::to_value(JsonRpcResponse {
                jsonrpc: JsonRpcVersion2_0,
                id: req_body.id,
                result: RESULT.clone(),
            })?
        }
        ClientRequest::CallToolRequest(v) => {
            debug!("call tool: {}", v.params.name);

            let id = req_body.id.clone();

            match v.params.name.as_ref() {
                "ocr" | "det" | "slide_match" | "slide_comparison" => {
                    let mut req = salvo::Request::new();

                    req.add_header("content-type", "application/json", true)?;

                    req.replace_body(ReqBody::Once(bytes::Bytes::from(
                        serde_json::to_value(v.params.arguments.clone())?.to_string(),
                    )));

                    let req = &mut req;

                    let args = ARGS.get().unwrap();

                    let mut sub_res = salvo::Response::new();
                    let mut sub_ctrl = salvo::FlowCtrl::default();

                    let handle_result: anyhow::Result<Option<Value>> = match v.params.name.as_ref()
                    {
                        "ocr" if args.ocr => {
                            route_ocr
                                .handle(req, depot, &mut sub_res, &mut sub_ctrl)
                                .await;
                            Ok(None)
                        }
                        "det" if args.det => {
                            route_det
                                .handle(req, depot, &mut sub_res, &mut sub_ctrl)
                                .await;
                            Ok(None)
                        }
                        "slide_match" if args.slide => {
                            route_slide_match
                                .handle(req, depot, &mut sub_res, &mut sub_ctrl)
                                .await;
                            Ok(None)
                        }
                        "slide_comparison" if args.slide => {
                            route_slide_comparison
                                .handle(req, depot, &mut sub_res, &mut sub_ctrl)
                                .await;
                            Ok(None)
                        }
                        v => Ok(Some(serde_json::to_value(JsonRpcError {
                            jsonrpc: JsonRpcVersion2_0,
                            id: id.clone(),
                            error: ErrorData::internal_error(
                                format!("the tool exists, but the server is not enabled: {}", v),
                                None,
                            ),
                        })?)),
                    };

                    match handle_result {
                        Ok(Some(response)) => response,
                        Ok(None) => match sub_res.take_body() {
                            ResBody::Once(v) => {
                                let result = serde_json::from_slice::<APIResponse<Value>>(&v)?;

                                if result.code == 200 {
                                    serde_json::to_value(JsonRpcResponse {
                                        jsonrpc: JsonRpcVersion2_0,
                                        id: id.clone(),
                                        result: CallToolResult::success(vec![Content::text(
                                            result.data.unwrap().to_string(),
                                        )]),
                                    })?
                                } else {
                                    serde_json::to_value(JsonRpcResponse {
                                        jsonrpc: JsonRpcVersion2_0,
                                        id: id.clone(),
                                        result: CallToolResult::error(vec![Content::text(
                                            result.msg,
                                        )]),
                                    })?
                                }
                            }
                            ResBody::Error(v) => {
                                debug!("tool call error: {}", v.to_string());

                                serde_json::to_value(JsonRpcResponse {
                                    jsonrpc: JsonRpcVersion2_0,
                                    id: id.clone(),
                                    result: CallToolResult::error(vec![Content::text(
                                        v.to_string(),
                                    )]),
                                })?
                            }
                            _ => anyhow::bail!("take_body"),
                        },
                        Err(e) => {
                            error!("tool call error: {e:#}");

                            serde_json::to_value(JsonRpcError {
                                jsonrpc: JsonRpcVersion2_0,
                                id: id.clone(),
                                error: ErrorData::internal_error(
                                    format!("internal error: {e}"),
                                    None,
                                ),
                            })?
                        }
                    }
                }
                v => {
                    debug!("tool does not exist: {}", v.to_string());

                    serde_json::to_value(JsonRpcError {
                        jsonrpc: JsonRpcVersion2_0,
                        id: id.clone(),
                        error: ErrorData::invalid_params(
                            format!("tool does not exist: {}", v),
                            None,
                        ),
                    })?
                }
            }
        }
        v => {
            debug!("method not found: {}", v.method());

            serde_json::to_value(JsonRpcError {
                jsonrpc: JsonRpcVersion2_0,
                id: req_body.id,
                error: ErrorData::new(ErrorCode::METHOD_NOT_FOUND, v.method().to_string(), None),
            })?
        }
    };

    Ok(Some(response))
}

#[handler]
async fn route_mcp(
    req_body: JsonBody<Value>,
    depot: &mut Depot,
    res: &mut Response,
) -> anyhow::Result<()> {
    let body = req_body.into_inner();

    if let Some(items) = body.as_array() {
        // JSON-RPC batch: process entries one by one, collecting all responses into an array
        let mut responses = Vec::new();

        for item in items {
            let Ok(entry) = serde_json::from_value::<JsonRpcRequestOrNotification>(item.clone())
            else {
                continue;
            };

            if let Some(response) = handle_single_rpc(entry, depot).await? {
                responses.push(response);
            }
        }

        res.render(Json(responses));
    } else {
        let entry = serde_json::from_value::<JsonRpcRequestOrNotification>(body)?;

        if let Some(response) = handle_single_rpc(entry, depot).await? {
            res.render(Json(response));
        }
    }

    Ok(())
}

#[handler]
fn default_error_handler(res: &mut Response) {
    if let ResBody::Error(v) = &res.body {
        // Log the full error, send only a generic message to the client to avoid leaking internal details
        error!("request failed: {}", v.to_string());

        res.render(Json(APIResponse {
            code: v.code.as_u16(),
            msg: "internal server error".to_string(),
            data: <Option<String>>::None,
        }));
    } else {
        res.render(Text::Plain(""));
    }
}

/// TOML config file structure; fields correspond to the CLI arguments available in `Args`.
#[derive(Debug, Default, Deserialize)]
struct Config {
    address: Option<String>,
    ocr: Option<bool>,
    old: Option<bool>,
    det: Option<bool>,
    slide: Option<bool>,
    ocr_charset_range: Option<String>,
    ocr_path: Option<String>,
    det_path: Option<String>,
    max_concurrency: Option<usize>,
    rate_limit: Option<usize>,
    api_key: Option<String>,
    num_threads: Option<usize>,
}

/// Collects long option names explicitly given on the command line (`--xxx` or `--xxx=value`).
fn provided_cli_flags() -> std::collections::HashSet<String> {
    std::env::args()
        .skip(1)
        .filter_map(|arg| {
            let arg = arg.strip_prefix("--")?;
            let name = arg.split('=').next()?.to_string();
            Some(name)
        })
        .collect()
}

/// Reads the TOML config file and merges fields not explicitly given on the command line into `Args`.
fn apply_config(mut args: Args) -> anyhow::Result<Args> {
    let Some(path) = args.config.clone() else {
        return Ok(args);
    };

    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("failed to read config {}: {e}", path))?;

    let mut cfg: Config = toml::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse config {}: {e}", path))?;

    let provided = provided_cli_flags();

    macro_rules! merge {
        ($field:ident) => {
            if cfg.$field.is_some() && !provided.contains(stringify!($field)) {
                args.$field = cfg.$field.take().unwrap();
            }
        };
        ($field:ident, Option) => {
            if cfg.$field.is_some() && !provided.contains(stringify!($field)) {
                args.$field = cfg.$field.take();
            }
        };
    }

    merge!(address);
    merge!(ocr);
    merge!(old);
    merge!(det);
    merge!(slide);
    merge!(ocr_charset_range, Option);
    merge!(ocr_path);
    merge!(det_path);
    merge!(max_concurrency);
    merge!(rate_limit);
    merge!(api_key, Option);
    merge!(num_threads);

    Ok(args)
}

#[tokio::main]
async fn main() {
    let mut args = match apply_config(Args::parse()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    tracing_subscriber::fmt()
        .with_ansi(enable_ansi_support().is_ok())
        .with_env_filter(
            // RUST_LOG = "info,kaptchaocr=debug"
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    if !(args.ocr || args.old) && !args.det && !args.slide {
        warn!("no enabled features, default enabled all features");

        args.ocr = true;
        args.old = true;
        args.det = true;
        args.slide = true;
    }

    ARGS.set(args.clone()).unwrap();

    // Concurrency semaphore: 0 uses the number of CPU cores (the default).
    let concurrency = if args.max_concurrency == 0 {
        std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
    } else {
        args.max_concurrency
    };

    INFERENCE_SEM
        .set(Arc::new(Semaphore::new(concurrency)))
        .unwrap();
    INFERENCE_CAP.set(concurrency).unwrap();

    if args.manifest.is_none()
        && (args.profile.is_some() || args.rec_model.is_some() || args.det_model.is_some())
    {
        warn!("--profile / --rec-model / --det-model require --manifest; ignoring them");
    }

    if let Err(e) = if let Some(manifest) = &args.manifest {
        init_from_manifest(manifest, &args)
    } else {
        init_ocr(&args)
    } {
        error!("failed to initialize models: {:#}", e);
        std::process::exit(1);
    }

    if args.slide {
        info!("slide enabled successfully");
    }

    if args.mcp {
        info!("mcp enabled successfully");
    }

    if args.only_mcp {
        info!("only mcp enabled successfully");
    }

    if args.rate_limit > 0 {
        info!("rate limit enabled: {} req / 10s per IP", args.rate_limit);
    }

    if args.api_key.is_some() {
        info!("api key auth enabled");
    }

    if args.num_threads > 0 {
        info!("ort intra-op threads: {}", args.num_threads);
    }

    let mut router = Router::new();

    router = router
        .hoop(salvo::prelude::Logger::new())
        .hoop(auth_and_rate_limit)
        .push(Router::with_path("/status").get(route_status))
        .push(Router::with_path("/healthz").get(route_healthz))
        .push(Router::with_path("/metrics").get(route_metrics));

    if (args.ocr || args.old) && !args.only_mcp {
        router = router
            .push(Router::with_path("/ocr").post(route_ocr))
            .push(Router::with_path("/ocr/upload").post(route_ocr_upload))
            .push(Router::with_path("/solve").post(route_solve));
    }

    if args.det && !args.only_mcp {
        router = router
            .push(Router::with_path("/det").post(route_det))
            .push(Router::with_path("/det/upload").post(route_det_upload));
    }

    // Click solving needs both a detector (boxes) and a recognizer (characters).
    if args.det && (args.ocr || args.old) && !args.only_mcp {
        router = router.push(Router::with_path("/solve/click").post(route_solve_click));
    }

    if args.slide && !args.only_mcp {
        router = router
            .push(Router::with_path("/slide_match").post(route_slide_match))
            .push(Router::with_path("/slide_comparison").post(route_slide_comparison))
            .push(Router::with_path("/slide/solve").post(route_slide_solve));
    }

    if args.mcp || args.only_mcp {
        router = router.push(Router::with_path("/mcp").post(route_mcp))
    }

    let docs = OpenApi::new("kaptchaocr api", "1.0.0").merge_router(&router);

    let router = router
        .unshift(docs.into_router("/docs.json"))
        .unshift(SwaggerUi::new("/docs.json").into_router("/docs"));

    let service = Service::new(router).catcher(Catcher::default().hoop(default_error_handler));

    request::set_global_secure_max_size(50 * 1024 * 1024);

    let acceptor = TcpListener::new(args.address);

    // Graceful shutdown: wait for SIGINT / SIGTERM
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
    let mut signals = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to install SIGTERM handler");

    let ctrl_c_tx = shutdown_tx.clone();
    let ctrl_c_task = tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                info!("received SIGINT, shutting down gracefully");
                let _ = ctrl_c_tx.send(());
            }
            Err(e) => {
                error!("failed to listen for ctrl-c: {e}");
            }
        }
    });

    let sigterm_tx = shutdown_tx;
    let sigterm_task = tokio::spawn(async move {
        loop {
            if signals.recv().await.is_some() {
                info!("received SIGTERM, shutting down gracefully");
                let _ = sigterm_tx.send(());
                break;
            }
        }
    });

    if let Some(v) = &args.acme {
        let server = Server::new(acceptor.acme().add_domain(v).bind().await);
        let handle = server.handle();
        let serve_task = tokio::spawn(server.serve(service));

        let _ = shutdown_rx.recv().await;
        handle.stop_graceful(std::time::Duration::from_secs(10));
        let _ = serve_task.await;
    } else {
        let server = Server::new(acceptor.bind().await);
        let handle = server.handle();
        let serve_task = tokio::spawn(server.serve(service));

        let _ = shutdown_rx.recv().await;
        handle.stop_graceful(std::time::Duration::from_secs(10));
        let _ = serve_task.await;
    }

    ctrl_c_task.abort();
    sigterm_task.abort();
    info!("server exited cleanly");
}

fn ocr_charset_range(args: &Args) -> anyhow::Result<Option<CharsetRange>> {
    args.ocr_charset_range
        .as_ref()
        .map(|v| match v.as_str() {
            "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" => {
                Ok(CharsetRange::from(v.parse::<i32>()?))
            }
            v => Ok(CharsetRange::from(v)),
        })
        .transpose()
}

/// Loads rec / det models from a model manifest (`--manifest`).
/// Defaults to the first rec / det entry in the manifest; `--rec-model <name>` / `--profile <name>`
/// can select the rec entry, and `--det-model <name>` the det entry. Manifest paths are relative to the working directory.
fn init_from_manifest(manifest_path: &str, args: &Args) -> anyhow::Result<()> {
    use kaptchaocr::manifest::ModelManifest;

    let manifest = ModelManifest::load(manifest_path)?;

    if args.rec_model.is_none() {
        if let Some(profile) = &args.profile {
            if !manifest.profiles.contains_key(profile) {
                anyhow::bail!(
                    "unknown profile {profile:?} in {manifest_path}; available profiles: {:?}",
                    manifest.profiles.keys().collect::<Vec<_>>()
                );
            }
        }
    }

    let rec_name = args.rec_model.clone().or_else(|| {
        args.profile
            .as_ref()
            .and_then(|p| manifest.profiles.get(p))
            .cloned()
    });

    if let Some(entry) = manifest
        .models
        .iter()
        .find(|m| m.task == "rec" && rec_name.as_deref().is_none_or(|name| m.name == name))
    {
        if args.ocr || args.old {
            let mut model = entry.build()?;

            if let Some(v) = ocr_charset_range(args)? {
                model.set_ranges(v);
            }

            OCR.set(model)
                .map_err(|_| anyhow::anyhow!("OCR already enabled"))?;

            info!("ocr loaded from manifest: {}", entry.name);
            set_model_name(entry.name.clone());
        }
    } else if let Some(name) = &rec_name {
        anyhow::bail!("no rec model named {name:?} in {manifest_path}");
    }

    let det_name = args.det_model.clone();

    if let Some(entry) = manifest
        .models
        .iter()
        .find(|m| m.task == "det" && det_name.as_deref().is_none_or(|name| m.name == name))
    {
        if args.det {
            let model = entry.build()?;

            DET.set(model)
                .map_err(|_| anyhow::anyhow!("detection already enabled"))?;

            info!("det loaded from manifest: {}", entry.name);
        }
    } else if let Some(name) = &det_name {
        anyhow::bail!("no det model named {name:?} in {manifest_path}");
    }

    Ok(())
}

#[cfg(feature = "inline-model")]
fn init_ocr(args: &Args) -> anyhow::Result<()> {
    let ocr_charset_range = ocr_charset_range(args)?;

    let num_threads = (args.num_threads > 0).then_some(args.num_threads);

    if args.ocr {
        let mut kaptchaocr = match num_threads {
            Some(n) => kaptchaocr_classification_with_num_threads(n)?,
            None => kaptchaocr_classification()?,
        };

        if let Some(v) = ocr_charset_range {
            kaptchaocr.set_ranges(v)
        };

        OCR.set(kaptchaocr)
            .map_err(|_| anyhow::anyhow!("OCR already enabled"))?;

        set_model_name("stock-common");
        info!("ocr enabled successfully");
    } else if args.old {
        let mut kaptchaocr = match num_threads {
            Some(n) => kaptchaocr_classification_old_with_num_threads(n)?,
            None => kaptchaocr_classification_old()?,
        };

        if let Some(v) = ocr_charset_range {
            kaptchaocr.set_ranges(v)
        };

        OCR.set(kaptchaocr)
            .map_err(|_| anyhow::anyhow!("OCR already enabled"))?;

        set_model_name("stock-old");
        info!("old enabled successfully");
    }

    if args.det {
        let kaptchaocr = match num_threads {
            Some(n) => kaptchaocr_detection_with_num_threads(n)?,
            None => kaptchaocr_detection()?,
        };

        DET.set(kaptchaocr)
            .map_err(|_| anyhow::anyhow!("detection already enabled"))?;

        info!("det enabled successfully");
    }

    Ok(())
}

#[cfg(not(feature = "inline-model"))]
fn init_ocr(args: &Args) -> anyhow::Result<()> {
    use std::fs::{read, read_to_string};
    use std::path::PathBuf;
    use std::str::FromStr;

    let ocr_charset_range = ocr_charset_range(args)?;

    if args.ocr || args.old {
        let mut path = PathBuf::from(args.ocr_path.clone());

        let model = read(&path)?;

        path.set_extension("json");

        let charset = read_to_string(path)?;

        let mut kaptchaocr = if args.num_threads > 0 {
            KaptchaOcr::new_with_num_threads(
                &model,
                Charset::from_str(&charset)?,
                args.num_threads,
            )?
        } else {
            KaptchaOcr::new(&model, Charset::from_str(&charset)?)?
        };

        if let Some(v) = ocr_charset_range {
            kaptchaocr.set_ranges(v)
        };

        OCR.set(kaptchaocr)
            .map_err(|_| anyhow::anyhow!("OCR already enabled"))?;

        set_model_name(
            std::path::Path::new(&args.ocr_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("custom"),
        );

        if args.ocr {
            info!("ocr enabled successfully");
        } else if args.old {
            info!("old enabled successfully");
        }
    }

    if args.det {
        let model = read(args.det_path.clone())?;

        let kaptchaocr = if args.num_threads > 0 {
            KaptchaOcr::new_model_with_num_threads(&model, args.num_threads)?
        } else {
            KaptchaOcr::new_model(&model)?
        };

        DET.set(kaptchaocr)
            .map_err(|_| anyhow::anyhow!("detection already enabled"))?;

        info!("det enabled successfully");
    }

    Ok(())
}
