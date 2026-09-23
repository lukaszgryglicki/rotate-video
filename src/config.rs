use crate::geometry::{parse_order, parse_output_size, Axis, OutputSize};
use crate::util::{env_args, env_bool, env_or, env_parse, env_str};
use crate::volume::{parse_interp, Interp};
use anyhow::{bail, Result};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioMode {
    /// Pitch-preserving time stretch (atempo chain).
    Tempo,
    /// Tape-style: resample (pitch changes with speed).
    Rate,
    /// Keep audio speed unchanged (only reverse/trim).
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Auto,
    Stream,
    Ram,
    Disk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Anchor {
    Start,
    #[default]
    Middle,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clip {
    None,
    Min(Anchor),
    Middle(Anchor),
    Nth(Option<usize>),
}

#[derive(Clone, Debug)]
pub struct Config {
    pub threads: usize,
    pub quiet: bool,
    pub dry_run: bool,
    pub keep_temp: bool,
    pub tmpdir: Option<PathBuf>,
    pub force: bool,
    pub mode: Mode,
    pub in_memory_pct: f64,
    pub chunk_mb: usize,
    pub clip: Clip,
    pub order: [Axis; 3],
    pub interp: Interp,
    pub output: OutputSize,
    pub fill_spec: String,
    pub even_dims: bool,
    pub audio_mode: AudioMode,
    pub audio_reverse: Option<bool>,

    pub ffmpeg: String,
    pub ffprobe: String,
    pub loglevel: String,
    pub global_args: Vec<String>,
    pub in_args: Vec<String>,
    pub vmap: String,
    pub amap: String,
    pub fps: Option<String>,
    pub pix_fmt: String,
    pub vf: Option<String>,
    pub decode_args: Vec<String>,
    pub temp_vcodec: Option<String>,
    pub temp_vcodec_args: Option<Vec<String>>,
    pub audio_codec: String,
    pub audio_ext: String,
    pub audio_args: Vec<String>,
    pub no_audio: bool,
    pub out_fps: Option<String>,
    pub vcodec: Option<String>,
    pub vcodec_args: Option<Vec<String>>,
    pub out_pix_fmt: String,
    pub acodec: Option<String>,
    pub acodec_args: Option<Vec<String>>,
    pub audio_filters: Option<String>,
    pub encode_args: Option<Vec<String>>,
    pub color_args: Option<Vec<String>>,
}

pub fn parse_anchor(v: &str) -> Result<Anchor> {
    Ok(match v.trim().to_ascii_lowercase().as_str() {
        "start" | "begin" | "beginning" | "first" | "head" => Anchor::Start,
        "middle" | "mid" | "center" | "centre" => Anchor::Middle,
        "end" | "last" | "tail" => Anchor::End,
        _ => bail!("ROT_CLIP_TIME must be start | middle | end, got {v:?}"),
    })
}

pub fn parse_clip(v: &str, nth_env: Option<usize>, anchor: Anchor) -> Result<Clip> {
    let s = v.trim().to_ascii_lowercase();
    let (kind, rest) = s.split_once([':', '=']).unwrap_or((s.as_str(), ""));
    let anchor = || -> Result<Anchor> {
        if rest.is_empty() {
            Ok(anchor)
        } else {
            parse_anchor(rest).map_err(|_| anyhow::anyhow!("ROT_CLIP={v}: time anchor must be start | middle | end"))
        }
    };
    Ok(match kind {
        "none" | "off" | "input" => Clip::None,
        "min" | "cube" => Clip::Min(anchor()?),
        "middle" | "mid" | "median" => Clip::Middle(anchor()?),
        "nth" => {
            let n = if rest.is_empty() {
                nth_env
            } else {
                Some(rest.parse::<usize>().map_err(|e| anyhow::anyhow!("ROT_CLIP: bad N in {v:?}: {e}"))?)
            };
            if n == Some(0) {
                bail!("ROT_CLIP=nth: N must be >= 1");
            }
            Clip::Nth(n)
        }
        _ => bail!("ROT_CLIP must be none | min[:start|:middle|:end] | middle[:start|:middle|:end] | nth[:N], got {v:?}"),
    })
}

impl Config {
    pub fn from_env() -> Result<Config> {
        let threads = match env_parse::<usize>("ROT_THREADS")? {
            Some(0) | None => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            Some(n) => n,
        };
        let audio_mode = match env_or("ROT_AUDIO_MODE", "tempo").to_ascii_lowercase().as_str() {
            "tempo" | "atempo" | "stretch" => AudioMode::Tempo,
            "rate" | "resample" | "tape" | "pitch" => AudioMode::Rate,
            "none" | "keep" | "off" => AudioMode::None,
            v => bail!("ROT_AUDIO_MODE must be tempo | rate | none, got {v:?}"),
        };
        let audio_reverse = match env_or("ROT_AUDIO_REVERSE", "auto").to_ascii_lowercase().as_str() {
            "auto" => None,
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            v => bail!("ROT_AUDIO_REVERSE must be auto | 1 | 0, got {v:?}"),
        };
        let mode = match env_or("ROT_MODE", "auto").to_ascii_lowercase().as_str() {
            "auto" => Mode::Auto,
            "stream" | "planar" => Mode::Stream,
            "ram" | "memory" | "mem" => Mode::Ram,
            "disk" | "mmap" => Mode::Disk,
            v => bail!("ROT_MODE must be auto | stream | ram | disk, got {v:?}"),
        };
        let in_memory_pct = env_parse::<f64>("ROT_IN_MEMORY")?.unwrap_or(80.0);
        if !(0.0..=100.0).contains(&in_memory_pct) {
            bail!("ROT_IN_MEMORY must be a percentage 0..100, got {in_memory_pct}");
        }
        let chunk_mb = env_parse::<usize>("ROT_CHUNK_MB")?.unwrap_or(512).max(1);
        let anchor = parse_anchor(&env_or("ROT_CLIP_TIME", "middle"))?;
        let clip = parse_clip(&env_or("ROT_CLIP", "none"), env_parse::<usize>("ROT_NTH")?, anchor)?;
        let opt_args = |name: &str| -> Result<Option<Vec<String>>> {
            match env_str(name) {
                None => Ok(None),
                Some(_) => Ok(Some(env_args(name, &[])?)),
            }
        };
        Ok(Config {
            threads,
            quiet: env_bool("ROT_QUIET", false)?,
            dry_run: env_bool("ROT_DRY_RUN", false)?,
            keep_temp: env_bool("ROT_KEEP_TEMP", false)?,
            tmpdir: env_str("ROT_TMPDIR").map(PathBuf::from),
            force: env_bool("ROT_FORCE", false)?,
            mode,
            in_memory_pct,
            chunk_mb,
            clip,
            order: parse_order(&env_or("ROT_ORDER", "xyz"))?,
            interp: parse_interp(&env_or("ROT_INTERP", "trilinear"))?,
            output: parse_output_size(&env_or("ROT_OUTPUT", "fit"))?,
            fill_spec: env_or("ROT_FILL", "0,0,0"),
            even_dims: env_bool("ROT_EVEN_DIMS", true)?,
            audio_mode,
            audio_reverse,

            ffmpeg: env_or("FF_FFMPEG", "ffmpeg"),
            ffprobe: env_or("FF_FFPROBE", "ffprobe"),
            loglevel: env_or("FF_LOGLEVEL", "error"),
            global_args: env_args("FF_GLOBAL_ARGS", &[])?,
            in_args: env_args("FF_IN_ARGS", &[])?,
            vmap: env_or("FF_VMAP", "0:v:0"),
            amap: env_or("FF_AMAP", "0:a:0"),
            fps: env_str("FF_FPS"),
            pix_fmt: env_or("FF_PIX_FMT", "auto"),
            vf: env_str("FF_VF").filter(|v| !v.eq_ignore_ascii_case("none")),
            decode_args: env_args("FF_DECODE_ARGS", &[])?,
            temp_vcodec: env_str("FF_TEMP_VCODEC"),
            temp_vcodec_args: opt_args("FF_TEMP_VCODEC_ARGS")?,
            audio_codec: env_or("FF_AUDIO_CODEC", "pcm_f32le"),
            audio_ext: env_or("FF_AUDIO_EXT", "wav"),
            audio_args: env_args("FF_AUDIO_ARGS", &[])?,
            no_audio: env_bool("FF_NO_AUDIO", false)?,
            out_fps: env_str("FF_OUT_FPS"),
            vcodec: env_str("FF_VCODEC"),
            vcodec_args: opt_args("FF_VCODEC_ARGS")?,
            out_pix_fmt: env_or("FF_OUT_PIX_FMT", "auto"),
            acodec: env_str("FF_ACODEC"),
            acodec_args: opt_args("FF_ACODEC_ARGS")?,
            audio_filters: env_str("FF_AUDIO_FILTERS"),
            encode_args: opt_args("FF_ENCODE_ARGS")?,
            color_args: opt_args("FF_COLOR_ARGS")?,
        })
    }
}

/// Per-container encoder defaults (used unless the matching FF_* variable is set).
pub struct EncodeDefaults {
    pub vcodec: &'static str,
    pub vcodec_args: &'static [&'static str],
    pub acodec: &'static str,
    pub acodec_args: &'static [&'static str],
    pub encode_args: &'static [&'static str],
}

pub const X265_ARGS: &[&str] = &["-crf", "16", "-preset", "medium", "-x265-params", "log-level=error"];

pub fn encode_defaults(ext: &str) -> EncodeDefaults {
    const AAC: &[&str] = &["-b:a", "192k"];
    match ext.to_ascii_lowercase().as_str() {
        "mp4" | "m4v" | "mov" => EncodeDefaults {
            vcodec: "libx265",
            vcodec_args: X265_ARGS,
            acodec: "aac",
            acodec_args: AAC,
            encode_args: &["-movflags", "+faststart"],
        },
        "webm" => EncodeDefaults {
            vcodec: "libvpx-vp9",
            vcodec_args: &["-crf", "30", "-b:v", "0", "-row-mt", "1"],
            acodec: "libopus",
            acodec_args: &["-b:a", "160k"],
            encode_args: &[],
        },
        "avi" => EncodeDefaults {
            vcodec: "mpeg4",
            vcodec_args: &["-q:v", "2"],
            acodec: "libmp3lame",
            acodec_args: AAC,
            encode_args: &[],
        },
        _ => EncodeDefaults {
            vcodec: "libx265",
            vcodec_args: X265_ARGS,
            acodec: "aac",
            acodec_args: AAC,
            encode_args: &[],
        },
    }
}

pub const USAGE: &str = "\
rotatevideo - rotate a video as a 3D volume (X = width, Y = height, Z = frames/time)

usage: rotatevideo INPUT OUTPUT XDEG YDEG ZDEG

Rotations follow the right-hand rule around each axis and are applied X, then Y, then Z
(ROT_ORDER). Everything else is configured through environment variables:

  ROT_CLIP=none          clip the INPUT volume before rotating:
                         none   - use the whole video
                         min    - cube: every side = min(W,H,frames), centered
                         middle - every side <= the middle value of (W,H,frames), centered
                         nth[:N] - keep every Nth frame (fps unchanged); N defaults to ceil(frames/max(W,H))
                         min and middle take an optional time anchor: min:start, middle:end, ...
  ROT_CLIP_TIME=middle   which part of the video min/middle keep: start | middle | end
  ROT_NTH=N              N for ROT_CLIP=nth
  ROT_MODE=auto          stream (rotation keeps time: no volume in memory) | ram | disk | auto
  ROT_IN_MEMORY=80       auto picks ram when the raw volume fits this % of physical RAM, else disk
  ROT_CHUNK_MB=512       RAM per chunk in the reversed streaming path (temp file per chunk)
  ROT_FORCE=0            proceed even when the temp disk seems too small
  ROT_THREADS=N          worker threads (default: all CPU cores)
  ROT_INTERP=trilinear   trilinear | nearest
  ROT_OUTPUT=fit         fit (bounding box) | crop (input size) | WxHxD (0 = fit, -1 = input)
  ROT_ORDER=xyz          order in which the three rotations are applied
  ROT_FILL=0,0,0         fill for empty space: R,G,B[,A] | #RRGGBB[AA] | gray value
  ROT_EVEN_DIMS=1        round output width/height up to even values
  ROT_AUDIO_MODE=tempo   tempo (pitch preserving) | rate (tape style) | none
  ROT_AUDIO_REVERSE=auto auto (reverse when the time axis flips) | 1 | 0
  ROT_TMPDIR=DIR         temp dir base (default: directory of OUTPUT)
  ROT_KEEP_TEMP=0        keep the temp dir (raw/intermediate frames, audio)
  ROT_DRY_RUN=0          only probe, print geometry, mode and commands
  ROT_QUIET=0            no progress output

  FF_FFMPEG=ffmpeg       FF_FFPROBE=ffprobe      FF_LOGLEVEL=error
  FF_GLOBAL_ARGS=        extra global ffmpeg args for every invocation
  FF_IN_ARGS=            extra input args (before -i INPUT), e.g. \"-ss 10 -t 20\" or \"-hwaccel auto\"
  FF_VMAP=0:v:0          FF_AMAP=0:a:0           stream selection
  FF_FPS=                decode frame rate (default: probed avg_frame_rate)
  FF_PIX_FMT=auto        intermediate pixel format: auto = the input's own (yuv420p, yuv420p10le, ...)
                         when supported, else rgb24; planar yuv/gray 8-16 bit and packed rgb/gray formats
  FF_VF=                 extra decode video filters, e.g. \"scale=960:-2\"
  FF_DECODE_ARGS=        extra output args for the frame decode
  FF_TEMP_VCODEC=        intermediate codec for the reversed streaming path (default libx264 crf 10;
                         ffv1 or \"libx264\" + FF_TEMP_VCODEC_ARGS=\"-qp 0\" for lossless)
  FF_TEMP_VCODEC_ARGS=   its args (default \"-crf 10 -preset veryfast\")
  FF_AUDIO_CODEC=pcm_f32le FF_AUDIO_EXT=wav      FF_AUDIO_ARGS=   lossless audio dump
  FF_NO_AUDIO=0          drop audio
  FF_OUT_FPS=            output frame rate (default: same as decode fps)
  FF_VCODEC=             default libx265 (libvpx-vp9 for webm, mpeg4 for avi); e.g. libx264
  FF_VCODEC_ARGS=        default \"-crf 16 -preset medium\" (x265) / \"-crf 30 -b:v 0 -row-mt 1\" (vp9)
  FF_OUT_PIX_FMT=auto    output pixel format (auto = intermediate yuv format or yuv420p; none = encoder)
  FF_ACODEC=             default aac (libopus for webm, libmp3lame for avi)
  FF_ACODEC_ARGS=        default \"-b:a 192k\"
  FF_AUDIO_FILTERS=      extra audio filters appended to the generated chain
  FF_COLOR_ARGS=         default: copy color_range/space/primaries/trc tags from the input
  FF_ENCODE_ARGS=        extra output args, default \"-movflags +faststart\" (+ \"-tag:v hvc1\") for mp4/mov

Any *_ARGS variable accepts shell-like quoting; the value \"none\" clears a non-empty default.
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_parsing() {
        let m = Anchor::Middle;
        assert_eq!(parse_clip("none", None, m).unwrap(), Clip::None);
        assert_eq!(parse_clip("MIN", None, m).unwrap(), Clip::Min(Anchor::Middle));
        assert_eq!(parse_clip("middle", None, m).unwrap(), Clip::Middle(Anchor::Middle));
        assert_eq!(parse_clip("min", None, Anchor::End).unwrap(), Clip::Min(Anchor::End));
        assert_eq!(parse_clip("min:start", None, Anchor::End).unwrap(), Clip::Min(Anchor::Start));
        assert_eq!(parse_clip("MIDDLE:End", None, m).unwrap(), Clip::Middle(Anchor::End));
        assert_eq!(parse_clip("cube=begin", None, m).unwrap(), Clip::Min(Anchor::Start));
        assert_eq!(parse_clip("nth", None, m).unwrap(), Clip::Nth(None));
        assert_eq!(parse_clip("nth", Some(7), m).unwrap(), Clip::Nth(Some(7)));
        assert_eq!(parse_clip("nth:50", Some(7), m).unwrap(), Clip::Nth(Some(50)));
        assert_eq!(parse_clip("nth=3", None, m).unwrap(), Clip::Nth(Some(3)));
        assert!(parse_clip("nth:0", None, m).is_err());
        assert!(parse_clip("nth:x", None, m).is_err());
        assert!(parse_clip("cube3", None, m).is_err());
        assert!(parse_clip("min:late", None, m).is_err());
        assert_eq!(parse_anchor("END").unwrap(), Anchor::End);
        assert!(parse_anchor("x").is_err());
    }

    #[test]
    fn x265_is_default() {
        for ext in ["mp4", "mkv", "mov", "MP4", "hevc", ""] {
            assert_eq!(encode_defaults(ext).vcodec, "libx265", "{ext}");
        }
        assert_eq!(encode_defaults("webm").vcodec, "libvpx-vp9");
        assert_eq!(encode_defaults("avi").vcodec, "mpeg4");
    }
}
