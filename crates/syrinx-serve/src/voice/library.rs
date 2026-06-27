//! A directory-backed **voice library**: persist each [`Voice`] as a per-voice
//! safetensors (the two tensors) plus a sidecar JSON (name / transcript / tokens /
//! source), and `save` / `load` / `list` / `remove` them. A saved voice round-trips
//! byte-exactly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{safetensors, Device, Tensor};

use super::{Voice, VoiceMeta};

/// safetensors key for the CAM++ speaker embedding.
const KEY_EMBEDDING: &str = "speaker_embedding";
/// safetensors key for the prompt mel.
const KEY_PROMPT_FEAT: &str = "prompt_feat";

/// Errors raised by a [`VoiceLibrary`] operation.
#[derive(Debug)]
pub enum VoiceLibraryError {
    /// Filesystem I/O failed.
    Io(String),
    /// A Candle / safetensors op failed (tensor (de)serialization).
    Candle(String),
    /// The JSON sidecar failed to (de)serialize.
    Json(String),
    /// A requested voice (or one of its two files) was not found.
    NotFound(String),
    /// A voice name that cannot be a safe single file stem (empty, or path-bearing).
    BadName(String),
}

impl std::fmt::Display for VoiceLibraryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoiceLibraryError::Io(m) => write!(f, "voice-library io error: {m}"),
            VoiceLibraryError::Candle(m) => write!(f, "voice-library tensor error: {m}"),
            VoiceLibraryError::Json(m) => write!(f, "voice-library json error: {m}"),
            VoiceLibraryError::NotFound(m) => write!(f, "voice-library not found: {m}"),
            VoiceLibraryError::BadName(m) => write!(f, "voice-library bad name: {m}"),
        }
    }
}

impl std::error::Error for VoiceLibraryError {}

impl From<candle_core::Error> for VoiceLibraryError {
    fn from(e: candle_core::Error) -> Self {
        VoiceLibraryError::Candle(e.to_string())
    }
}

/// A directory holding persisted [`Voice`]s — two files per voice (`<name>.safetensors`
/// for the tensors, `<name>.voice.json` for the metadata sidecar).
pub struct VoiceLibrary {
    dir: PathBuf,
}

impl VoiceLibrary {
    /// Open (creating it if absent) a voice library rooted at `dir`.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, VoiceLibraryError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| VoiceLibraryError::Io(e.to_string()))?;
        Ok(Self { dir })
    }

    /// The library root directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The safetensors path for a voice `name` (`<dir>/<name>.safetensors`).
    pub fn path(&self, name: &str) -> Result<PathBuf, VoiceLibraryError> {
        Ok(self.dir.join(format!("{}.safetensors", safe_stem(name)?)))
    }

    /// The JSON sidecar path for a voice `name` (`<dir>/<name>.voice.json`).
    fn meta_path(&self, name: &str) -> Result<PathBuf, VoiceLibraryError> {
        Ok(self.dir.join(format!("{}.voice.json", safe_stem(name)?)))
    }

    /// Persist `voice`: write its two tensors to `<name>.safetensors` and its metadata to
    /// `<name>.voice.json`. Overwrites an existing voice of the same name.
    pub fn save(&self, voice: &Voice) -> Result<(), VoiceLibraryError> {
        let st_path = self.path(&voice.name)?;
        let meta_path = self.meta_path(&voice.name)?;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert(KEY_EMBEDDING.to_string(), voice.speaker_embedding.clone());
        tensors.insert(KEY_PROMPT_FEAT.to_string(), voice.prompt_feat.clone());
        safetensors::save(&tensors, &st_path)?;

        let json = serde_json::to_vec_pretty(&voice.meta())
            .map_err(|e| VoiceLibraryError::Json(e.to_string()))?;
        std::fs::write(&meta_path, json).map_err(|e| VoiceLibraryError::Io(e.to_string()))?;
        Ok(())
    }

    /// Load the voice `name` (its tensors onto the CPU device — the parity device).
    pub fn load(&self, name: &str) -> Result<Voice, VoiceLibraryError> {
        self.load_on_device(name, &Device::Cpu)
    }

    /// Load the voice `name` with its tensors placed on `dev`.
    pub fn load_on_device(&self, name: &str, dev: &Device) -> Result<Voice, VoiceLibraryError> {
        let st_path = self.path(name)?;
        let meta_path = self.meta_path(name)?;
        if !st_path.exists() || !meta_path.exists() {
            return Err(VoiceLibraryError::NotFound(name.to_string()));
        }

        let meta_bytes =
            std::fs::read(&meta_path).map_err(|e| VoiceLibraryError::Io(e.to_string()))?;
        let meta: VoiceMeta = serde_json::from_slice(&meta_bytes)
            .map_err(|e| VoiceLibraryError::Json(e.to_string()))?;

        let tensors = safetensors::load(&st_path, dev)?;
        let speaker_embedding = tensors
            .get(KEY_EMBEDDING)
            .ok_or_else(|| VoiceLibraryError::Candle(format!("`{KEY_EMBEDDING}` missing")))?
            .clone();
        let prompt_feat = tensors
            .get(KEY_PROMPT_FEAT)
            .ok_or_else(|| VoiceLibraryError::Candle(format!("`{KEY_PROMPT_FEAT}` missing")))?
            .clone();

        Ok(Voice {
            name: meta.name,
            speaker_embedding,
            prompt_feat,
            prompt_token: meta.prompt_token,
            prompt_text: meta.prompt_text,
            source: meta.source,
        })
    }

    /// List the names of every persisted voice (sorted), discovered by their JSON sidecars.
    pub fn list(&self) -> Result<Vec<String>, VoiceLibraryError> {
        let mut names = Vec::new();
        let entries =
            std::fs::read_dir(&self.dir).map_err(|e| VoiceLibraryError::Io(e.to_string()))?;
        for entry in entries {
            let entry = entry.map_err(|e| VoiceLibraryError::Io(e.to_string()))?;
            let file = entry.file_name();
            let file = file.to_string_lossy();
            if let Some(stem) = file.strip_suffix(".voice.json") {
                // Only count a voice whose tensors are also present.
                if self.dir.join(format!("{stem}.safetensors")).exists() {
                    names.push(stem.to_string());
                }
            }
        }
        names.sort();
        Ok(names)
    }

    /// Remove the voice `name` (both files). Errors with [`VoiceLibraryError::NotFound`] if
    /// neither file exists; a partially-present voice is cleaned up best-effort.
    pub fn remove(&self, name: &str) -> Result<(), VoiceLibraryError> {
        let st_path = self.path(name)?;
        let meta_path = self.meta_path(name)?;
        let had_any = st_path.exists() || meta_path.exists();
        if !had_any {
            return Err(VoiceLibraryError::NotFound(name.to_string()));
        }
        if st_path.exists() {
            std::fs::remove_file(&st_path).map_err(|e| VoiceLibraryError::Io(e.to_string()))?;
        }
        if meta_path.exists() {
            std::fs::remove_file(&meta_path).map_err(|e| VoiceLibraryError::Io(e.to_string()))?;
        }
        Ok(())
    }
}

/// Validate that `name` is a safe single-path-component file stem: non-empty and free of
/// path separators / parent refs, so it cannot escape the library directory.
fn safe_stem(name: &str) -> Result<&str, VoiceLibraryError> {
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name == "."
        || name == ".."
        || name.contains('\0')
    {
        return Err(VoiceLibraryError::BadName(name.to_string()));
    }
    Ok(name)
}
