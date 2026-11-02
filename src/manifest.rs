//! ONNX model manifest: configure any exported ONNX model (recognition or
//! detection) without code changes.
//!
//! See `model/models.json` at the repository root for examples.
//! Model and charset paths are relative to the process working directory.
use crate::{Charset, DbParams, DetType, KaptchaOcr};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Root of a model manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelManifest {
    pub models: Vec<ManifestModel>,

    /// Optional named profiles: a map from profile name to model name, e.g.
    /// `{"fast": "ft-tiny", "balanced": "ft-small", "accurate": "ft-medium"}`.
    /// The server selects one with `--profile <name>` (same as `--rec-model <name>`).
    #[serde(default)]
    pub profiles: BTreeMap<String, String>,
}

/// A single model entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestModel {
    /// Model identifier (free-form name).
    pub name: String,
    /// Task type: `rec` (content recognition) or `det` (object detection).
    pub task: String,
    /// Path to the ONNX model file.
    pub path: String,
    /// Charset JSON path for a recognition model (required when `task == "rec"`).
    #[serde(default)]
    pub charset: Option<String>,
    /// Decoding and post-processing config for a detection model (optional when `task == "det"`).
    #[serde(default)]
    pub det: Option<ManifestDetConfig>,
}

/// Detection model config (YOLO or DB).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ManifestDetConfig {
    /// `yolo` (anchor grid) or `db` (DB probability map).
    pub r#type: String,
    /// NMS IoU threshold.
    pub nms_thr: f32,
    /// Score threshold.
    pub score_thr: f32,
    // —— DB post-processing (used when `type == "db"`) ——
    /// Probability-map binarization threshold.
    pub thresh: f32,
    /// Minimum average score for a kept box.
    pub box_thresh: f32,
    /// Text-box expansion (unclip) ratio.
    pub unclip_ratio: f32,
    /// Maximum number of candidate boxes.
    pub max_candidates: usize,
}

impl Default for ManifestDetConfig {
    fn default() -> Self {
        Self {
            r#type: "yolo".to_string(),
            nms_thr: crate::NMS_THR,
            score_thr: crate::SCORE_THR,
            thresh: 0.2,
            box_thresh: 0.4,
            unclip_ratio: 1.4,
            max_candidates: 3000,
        }
    }
}

impl ModelManifest {
    /// Parse from a JSON string.
    pub fn from_json_str(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Read and parse a manifest file.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read manifest {}: {e}", path.display()))?;
        Self::from_json_str(&text)
            .map_err(|e| anyhow::anyhow!("failed to parse manifest {}: {e}", path.display()))
    }
}

impl ManifestModel {
    /// Build a model instance from this manifest entry.
    pub fn build(&self) -> anyhow::Result<KaptchaOcr<'static>> {
        let bytes = std::fs::read(&self.path).map_err(|e| {
            anyhow::anyhow!("failed to read model {} ({}): {e}", self.path, self.name)
        })?;

        match self.task.as_str() {
            "rec" => {
                let charset_path = self.charset.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("model {}: rec models require a charset path", self.name)
                })?;
                let charset_text = std::fs::read_to_string(charset_path).map_err(|e| {
                    anyhow::anyhow!(
                        "failed to read charset {} ({}): {e}",
                        charset_path,
                        self.name
                    )
                })?;
                let charset: Charset = serde_json::from_str(&charset_text)?;
                Ok(KaptchaOcr::new(bytes, charset)?)
            }
            "det" => {
                let mut model = KaptchaOcr::new_model(bytes)?;
                let cfg = self.det.clone().unwrap_or_default();
                model.set_det_type(match cfg.r#type.as_str() {
                    "db" => DetType::Db,
                    _ => DetType::Yolo,
                });
                if model.det_type() == DetType::Db {
                    model.set_db_params(DbParams {
                        thresh: cfg.thresh,
                        box_thresh: cfg.box_thresh,
                        unclip_ratio: cfg.unclip_ratio,
                        max_candidates: cfg.max_candidates,
                    });
                }
                Ok(model)
            }
            other => anyhow::bail!("model {}: unknown task {other:?} (rec|det)", self.name),
        }
    }
}
