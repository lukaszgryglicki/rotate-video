use crate::config::{encode_defaults, AudioMode, Config};
use crate::pixfmt::{PixelFormat, PlaneKind};
use crate::util::shell_join;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Deserialize, Default)]
struct ProbeJson {
    #[serde(default)]
    streams: Vec<StreamJson>,
    format: Option<FormatJson>,
}

#[derive(Deserialize, Default)]
struct StreamJson {
    codec_type: Option<String>,
    codec_name: Option<String>,
    width: Option<u64>,
    height: Option<u64>,
    pix_fmt: Option<String>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    nb_frames: Option<String>,
    duration: Option<String>,
    sample_rate: Option<String>,
    channels: Option<u64>,
    color_range: Option<String>,
    color_space: Option<String>,
    color_transfer: Option<String>,
    color_primaries: Option<String>,
}

#[derive(Deserialize, Default)]
struct FormatJson {
    duration: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AudioInfo {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ColorInfo {
    pub range: Option<String>,
    pub space: Option<String>,
    pub transfer: Option<String>,
    pub primaries: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ProbeInfo {
    pub vcodec: String,
    pub width: usize,
    pub height: usize,
    pub pix_fmt: String,
    pub fps: String,
    pub fps_f: f64,
    pub nb_frames: Option<u64>,
    pub duration: Option<f64>,
    pub audio: Option<AudioInfo>,
    pub color: ColorInfo,
}

pub fn parse_rate(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Some((n, d)) = s.split_once('/') {
        let n: f64 = n.trim().parse().ok()?;
        let d: f64 = d.trim().parse().ok()?;
        (d != 0.0 && n > 0.0).then_some(n / d)
    } else {
        s.parse::<f64>().ok().filter(|v| *v > 0.0)
    }
}

fn run_probe_json(cfg: &Config, path: &Path) -> Result<ProbeJson> {
    let mut cmd = Command::new(&cfg.ffprobe);
    cmd.args(["-hide_banner", "-v", "error", "-print_format", "json", "-show_streams", "-show_format"])
        .arg(path)
        .stdin(Stdio::null());
    let out = cmd.output().with_context(|| format!("cannot run {}", cfg.ffprobe))?;
    if !out.status.success() {
        bail!("ffprobe failed on {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
    }
    serde_json::from_slice(&out.stdout).context("cannot parse ffprobe JSON")
}

pub fn probe_input(cfg: &Config, input: &Path) -> Result<ProbeInfo> {
    let p = run_probe_json(cfg, input)?;
    let v = p
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .ok_or_else(|| anyhow::anyhow!("no video stream in {}", input.display()))?;
    let a = p.streams.iter().find(|s| s.codec_type.as_deref() == Some("audio"));
    let (width, height) = match (v.width, v.height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => (w as usize, h as usize),
        _ => bail!("cannot determine video dimensions"),
    };
    let fps = match &cfg.fps {
        Some(f) => f.clone(),
        None => {
            let avg = v.avg_frame_rate.as_deref().filter(|r| parse_rate(r).is_some());
            let r = v.r_frame_rate.as_deref().filter(|r| parse_rate(r).is_some());
            avg.or(r).ok_or_else(|| anyhow::anyhow!("cannot determine frame rate; set FF_FPS"))?.to_string()
        }
    };
    let fps_f = parse_rate(&fps).ok_or_else(|| anyhow::anyhow!("bad frame rate {fps:?}"))?;
    let duration = v
        .duration
        .as_deref()
        .and_then(|d| d.parse::<f64>().ok())
        .or_else(|| p.format.as_ref().and_then(|f| f.duration.as_deref()).and_then(|d| d.parse().ok()));
    let known = |s: &Option<String>| s.clone().filter(|v| v != "unknown" && v != "unspecified" && !v.is_empty());
    Ok(ProbeInfo {
        vcodec: v.codec_name.clone().unwrap_or_default(),
        width,
        height,
        pix_fmt: v.pix_fmt.clone().unwrap_or_default(),
        fps,
        fps_f,
        nb_frames: v.nb_frames.as_deref().and_then(|n| n.parse().ok()).filter(|n| *n > 0),
        duration,
        audio: a.map(|s| AudioInfo {
            codec: s.codec_name.clone().unwrap_or_default(),
            sample_rate: s.sample_rate.as_deref().and_then(|r| r.parse().ok()).unwrap_or(48000),
            channels: s.channels.unwrap_or(0),
        }),
        color: ColorInfo {
            range: known(&v.color_range),
            space: known(&v.color_space),
            transfer: known(&v.color_transfer),
            primaries: known(&v.color_primaries),
        },
    })
}

/// How the input is turned into raw frames (shared by every decode variant and the audio dump).
#[derive(Clone, Debug, Default)]
pub struct Decode {
    pub pix_fmt: String,
    /// CFR rate of the produced frames (`-r`).
    pub fps: String,
    pub vf: String,
    pub frames: Option<usize>,
    pub video_in_args: Vec<String>,
    pub audio_in_args: Vec<String>,
}

fn base_cmd(cfg: &Config, stdin: bool) -> Vec<String> {
    let mut v = vec![cfg.ffmpeg.clone(), "-hide_banner".into(), "-loglevel".into(), cfg.loglevel.clone()];
    if !stdin {
        v.push("-nostdin".into());
    }
    v.extend(cfg.global_args.iter().cloned());
    v
}

fn in_head(cfg: &Config, extra: &[String], input: &Path) -> Vec<String> {
    let mut v = base_cmd(cfg, false);
    v.extend(cfg.in_args.iter().cloned());
    v.extend(extra.iter().cloned());
    v.extend(["-i".into(), input.to_string_lossy().into_owned()]);
    v
}

fn decode_head(cfg: &Config, dec: &Decode, input: &Path) -> Vec<String> {
    let mut v = in_head(cfg, &dec.video_in_args, input);
    v.extend(["-map".into(), cfg.vmap.clone(), "-an".into(), "-sn".into(), "-dn".into()]);
    v.extend(["-r".into(), dec.fps.clone()]);
    if !dec.vf.is_empty() {
        v.extend(["-vf".into(), dec.vf.clone()]);
    }
    v.extend(["-pix_fmt".into(), dec.pix_fmt.clone()]);
    if let Some(n) = dec.frames {
        v.extend(["-frames:v".into(), n.to_string()]);
    }
    v.extend(cfg.decode_args.iter().cloned());
    v
}

/// One decoded frame into a NUT container: tells us the real post-filter frame size.
pub fn probe_frame_cmd(cfg: &Config, dec: &Decode, input: &Path, out: &Path) -> Vec<String> {
    let mut v = decode_head(cfg, dec, input);
    v.extend(["-frames:v".into(), "1".into(), "-c:v".into(), "rawvideo".into(), "-f".into(), "nut".into()]);
    v.extend(["-y".into(), out.to_string_lossy().into_owned()]);
    v
}

pub fn decode_file_cmd(cfg: &Config, dec: &Decode, input: &Path, out: &Path) -> Vec<String> {
    let mut v = decode_head(cfg, dec, input);
    v.extend(["-f".into(), "rawvideo".into(), "-y".into(), out.to_string_lossy().into_owned()]);
    v
}

pub fn decode_pipe_cmd(cfg: &Config, dec: &Decode, input: &Path) -> Vec<String> {
    let mut v = decode_head(cfg, dec, input);
    v.extend(["-f".into(), "rawvideo".into(), "pipe:1".into()]);
    v
}

/// Intermediate codec for the reversed streaming path: x264 where it takes the format as is, else ffv1.
pub fn temp_codec_defaults(pf: &PixelFormat, w: usize, h: usize) -> (&'static str, &'static [&'static str]) {
    const X264: &[&str] = &["-crf", "10", "-preset", "veryfast"];
    const FFV1: &[&str] = &["-level", "3", "-coder", "1", "-context", "1", "-g", "1", "-slicecrc", "0"];
    let depth_ok = pf.depth == 8 || pf.depth == 10;
    let sub_ok = pf.is_yuv()
        && matches!((pf.planes[1].sx, pf.planes[1].sy), (0, 0) | (1, 0) | (1, 1))
        && w.is_multiple_of(1 << pf.planes[1].sx)
        && h.is_multiple_of(1 << pf.planes[1].sy);
    if depth_ok && !pf.has_alpha() && (sub_ok || pf.is_gray()) {
        ("libx264", X264)
    } else if matches!(pf.planes[0].kind, PlaneKind::Packed("rgb" | "bgr" | "bgrx")) && pf.bytes == 1 {
        ("libx264rgb", X264)
    } else {
        ("ffv1", FFV1)
    }
}

/// Transcode into consecutive chunk files of `chunk` frames each (keyframe forced at every chunk start).
#[allow(clippy::too_many_arguments)]
pub fn segment_cmd(cfg: &Config, dec: &Decode, input: &Path, pf: &PixelFormat, wh: [usize; 2], chunk: usize, fps_f: f64, pattern: &Path) -> Vec<String> {
    let (codec, args) = temp_codec_defaults(pf, wh[0], wh[1]);
    let mut v = decode_head(cfg, dec, input);
    v.extend(["-c:v".into(), cfg.temp_vcodec.clone().unwrap_or_else(|| codec.to_string())]);
    match &cfg.temp_vcodec_args {
        Some(a) => v.extend(a.iter().cloned()),
        None => v.extend(args.iter().map(|s| s.to_string())),
    }
    v.extend(["-force_key_frames".into(), format!("expr:gte(n,n_forced*{chunk})")]);
    v.extend(["-f".into(), "segment".into(), "-segment_time".into(), format!("{:.6}", (chunk as f64 - 0.5) / fps_f)]);
    v.extend(["-segment_format".into(), "matroska".into(), "-reset_timestamps".into(), "1".into()]);
    v.extend(["-y".into(), pattern.to_string_lossy().into_owned()]);
    v
}

pub fn chunk_cmd(cfg: &Config, pix_fmt: &str, chunk: &Path) -> Vec<String> {
    let mut v = base_cmd(cfg, false);
    v.extend(["-i".into(), chunk.to_string_lossy().into_owned()]);
    v.extend(["-map".into(), "0:v:0".into(), "-an".into(), "-sn".into(), "-dn".into()]);
    v.extend(["-fps_mode".into(), "passthrough".into(), "-pix_fmt".into(), pix_fmt.to_string()]);
    v.extend(["-f".into(), "rawvideo".into(), "pipe:1".into()]);
    v
}

pub fn audio_cmd(cfg: &Config, dec: &Decode, input: &Path, out: &Path) -> Vec<String> {
    let mut v = in_head(cfg, &dec.audio_in_args, input);
    v.extend(["-map".into(), cfg.amap.clone(), "-vn".into(), "-sn".into(), "-dn".into()]);
    v.extend(["-c:a".into(), cfg.audio_codec.clone()]);
    v.extend(cfg.audio_args.iter().cloned());
    v.extend(["-y".into(), out.to_string_lossy().into_owned()]);
    v
}

pub struct AudioPlan {
    pub reversed: bool,
    /// Speed factor: >1 plays faster (shorter), <1 slower (longer).
    pub speed: f64,
    pub new_duration: Option<f64>,
    pub filter: String,
}

/// `in_frames` = frames of the volume (after ROT_CLIP); `nth` frames of source time per volume frame.
/// With unknown counts (streaming) the frame count is unchanged, so only the rates matter.
#[allow(clippy::too_many_arguments)]
pub fn audio_plan(
    cfg: &Config,
    reversed: bool,
    in_frames: Option<usize>,
    in_fps: f64,
    nth: usize,
    out_frames: Option<usize>,
    out_fps: f64,
    sample_rate: u32,
) -> AudioPlan {
    let (speed, new_dur) = match (in_frames, out_frames) {
        (Some(i), Some(o)) => {
            let old = i as f64 * nth as f64 / in_fps;
            let new = o as f64 / out_fps;
            (if new > 0.0 { old / new } else { 1.0 }, Some(new))
        }
        _ => (nth as f64 * out_fps / in_fps, None),
    };
    let mut chain: Vec<String> = Vec::new();
    if reversed {
        chain.push("areverse".into());
    }
    if (speed - 1.0).abs() > 1e-9 {
        match cfg.audio_mode {
            AudioMode::Tempo => {
                let mut t = speed;
                while t > 2.0 + 1e-9 || (t - 2.0).abs() <= 1e-9 {
                    chain.push("atempo=2.0".into());
                    t /= 2.0;
                }
                while t < 0.5 - 1e-9 || (t - 0.5).abs() <= 1e-9 {
                    chain.push("atempo=0.5".into());
                    t *= 2.0;
                }
                if (t - 1.0).abs() > 1e-9 {
                    chain.push(format!("atempo={t:.9}"));
                }
            }
            AudioMode::Rate => {
                let sr = sample_rate.max(1);
                let new_sr = ((sr as f64) * speed).round().max(1.0) as u64;
                chain.push(format!("asetrate={new_sr}"));
                chain.push(format!("aresample={sr}"));
            }
            AudioMode::None => {}
        }
    }
    if let Some(d) = new_dur {
        chain.push(format!("atrim=duration={d:.6}"));
    }
    chain.push("asetpts=PTS-STARTPTS".into());
    if let Some(extra) = &cfg.audio_filters {
        chain.push(extra.clone());
    }
    AudioPlan { reversed, speed, new_duration: new_dur, filter: chain.join(",") }
}

pub struct EncodeParams<'a> {
    pub output: &'a Path,
    pub width: usize,
    pub height: usize,
    pub out_fps: &'a str,
    pub pix_fmt: &'a str,
    pub out_pix_fmt: &'a str,
    pub audio: Option<(&'a Path, &'a AudioPlan)>,
    pub color: &'a ColorInfo,
}

pub fn encode_cmd(cfg: &Config, p: &EncodeParams) -> Vec<String> {
    let ext = p.output.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let d = encode_defaults(&ext);
    let mut v = base_cmd(cfg, true);
    v.extend(["-thread_queue_size".into(), "1024".into()]);
    v.extend(["-f".into(), "rawvideo".into(), "-pix_fmt".into(), p.pix_fmt.to_string()]);
    v.extend(["-video_size".into(), format!("{}x{}", p.width, p.height)]);
    v.extend(["-framerate".into(), p.out_fps.to_string(), "-i".into(), "pipe:0".into()]);
    if let Some((apath, _)) = p.audio {
        v.extend(["-thread_queue_size".into(), "1024".into(), "-i".into(), apath.to_string_lossy().into_owned()]);
    }
    v.extend(["-map".into(), "0:v:0".into()]);
    if let Some((_, plan)) = p.audio {
        v.extend(["-map".into(), "1:a:0".into(), "-af".into(), plan.filter.clone()]);
        v.extend(["-c:a".into(), cfg.acodec.clone().unwrap_or_else(|| d.acodec.to_string())]);
        match &cfg.acodec_args {
            Some(a) => v.extend(a.iter().cloned()),
            None => v.extend(d.acodec_args.iter().map(|s| s.to_string())),
        }
    } else {
        v.push("-an".into());
    }
    let vcodec = cfg.vcodec.clone().unwrap_or_else(|| d.vcodec.to_string());
    v.extend(["-c:v".into(), vcodec.clone()]);
    match &cfg.vcodec_args {
        Some(a) => v.extend(a.iter().cloned()),
        None => v.extend(d.vcodec_args.iter().map(|s| s.to_string())),
    }
    if !p.out_pix_fmt.eq_ignore_ascii_case("none") {
        v.extend(["-pix_fmt".into(), p.out_pix_fmt.to_string()]);
    }
    match &cfg.color_args {
        Some(a) => v.extend(a.iter().cloned()),
        None => {
            let c = p.color;
            for (flag, val) in [
                ("-color_range", &c.range),
                ("-colorspace", &c.space),
                ("-color_primaries", &c.primaries),
                ("-color_trc", &c.transfer),
            ] {
                if let Some(x) = val {
                    v.extend([flag.into(), x.clone()]);
                }
            }
        }
    }
    match &cfg.encode_args {
        Some(a) => v.extend(a.iter().cloned()),
        None => {
            v.extend(d.encode_args.iter().map(|s| s.to_string()));
            let hevc = vcodec.contains("265") || vcodec.contains("hevc");
            if hevc && matches!(ext.as_str(), "mp4" | "m4v" | "mov") {
                v.extend(["-tag:v".into(), "hvc1".into()]);
            }
        }
    }
    v.extend(["-y".into(), p.output.to_string_lossy().into_owned()]);
    v
}

pub fn run(cmd: &[String]) -> Result<()> {
    let st = Command::new(&cmd[0])
        .args(&cmd[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("cannot run {}", cmd[0]))?;
    if !st.success() {
        bail!("command failed ({st}): {}", shell_join(cmd));
    }
    Ok(())
}

/// Width/height/pix_fmt of the single-frame NUT probe.
pub fn probe_frame_size(cfg: &Config, nut: &Path) -> Result<(usize, usize, String)> {
    let p = run_probe_json(cfg, nut)?;
    let v = p
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video"))
        .ok_or_else(|| anyhow::anyhow!("frame probe produced no video stream"))?;
    match (v.width, v.height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => Ok((w as usize, h as usize, v.pix_fmt.clone().unwrap_or_default())),
        _ => bail!("frame probe: cannot determine decoded frame size"),
    }
}

pub fn temp_base(cfg: &Config, output: &Path) -> PathBuf {
    if let Some(d) = &cfg.tmpdir {
        return d.clone();
    }
    match output.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates() {
        assert_eq!(parse_rate("30/1"), Some(30.0));
        assert!((parse_rate("30000/1001").unwrap() - 29.97).abs() < 0.01);
        assert_eq!(parse_rate("25"), Some(25.0));
        assert_eq!(parse_rate("0/0"), None);
        assert_eq!(parse_rate("x"), None);
    }

    fn cfg(mode: AudioMode) -> Config {
        let mut c = Config::from_env().unwrap();
        c.audio_mode = mode;
        c.audio_filters = None;
        c
    }

    #[test]
    fn tempo_chain() {
        // 3600 -> 1080 frames: 3.33x faster => 2.0, 1.6667
        let p = audio_plan(&cfg(AudioMode::Tempo), false, Some(3600), 30.0, 1, Some(1080), 30.0, 48000);
        assert!((p.speed - 3600.0 / 1080.0).abs() < 1e-9);
        assert!(p.filter.starts_with("atempo=2.0,atempo=1.666666667,atrim=duration=36.000000,asetpts"), "{}", p.filter);
        // 1080 -> 3600 frames: 0.3x => 0.5, 0.6
        let p = audio_plan(&cfg(AudioMode::Tempo), true, Some(1080), 30.0, 1, Some(3600), 30.0, 48000);
        assert!(p.filter.starts_with("areverse,atempo=0.5,atempo=0.600000000,atrim=duration=120.000000"), "{}", p.filter);
        // same length: no tempo stage
        let p = audio_plan(&cfg(AudioMode::Tempo), false, Some(100), 25.0, 1, Some(100), 25.0, 48000);
        assert_eq!(p.filter, "atrim=duration=4.000000,asetpts=PTS-STARTPTS");
        // every 50th frame at unchanged fps: 50x faster
        let p = audio_plan(&cfg(AudioMode::Tempo), false, Some(1200), 100.0, 50, Some(1200), 100.0, 48000);
        assert!((p.speed - 50.0).abs() < 1e-9);
        assert_eq!(p.new_duration, Some(12.0));
    }

    #[test]
    fn unknown_frame_count_uses_rates_only() {
        let p = audio_plan(&cfg(AudioMode::Tempo), true, None, 30.0, 1, None, 30.0, 48000);
        assert_eq!(p.filter, "areverse,asetpts=PTS-STARTPTS");
        assert_eq!(p.new_duration, None);
        let p = audio_plan(&cfg(AudioMode::Tempo), false, None, 30.0, 1, None, 60.0, 48000);
        assert!((p.speed - 2.0).abs() < 1e-9);
        assert_eq!(p.filter, "atempo=2.0,asetpts=PTS-STARTPTS");
        let p = audio_plan(&cfg(AudioMode::Tempo), false, None, 100.0, 4, None, 100.0, 48000);
        assert_eq!(p.filter, "atempo=2.0,atempo=2.0,asetpts=PTS-STARTPTS");
    }

    #[test]
    fn rate_chain() {
        let p = audio_plan(&cfg(AudioMode::Rate), false, Some(200), 30.0, 1, Some(100), 30.0, 44100);
        assert_eq!(p.filter, "asetrate=88200,aresample=44100,atrim=duration=3.333333,asetpts=PTS-STARTPTS");
        let p = audio_plan(&cfg(AudioMode::None), true, Some(200), 30.0, 1, Some(100), 30.0, 44100);
        assert_eq!(p.filter, "areverse,atrim=duration=3.333333,asetpts=PTS-STARTPTS");
    }

    #[test]
    fn encode_defaults_by_ext() {
        let mut c = Config::from_env().unwrap();
        c.vcodec = None;
        c.vcodec_args = None;
        c.encode_args = None;
        c.acodec = None;
        c.acodec_args = None;
        c.color_args = None;
        let color = ColorInfo { range: Some("tv".into()), space: Some("bt709".into()), ..Default::default() };
        let plan = audio_plan(&c, false, Some(10), 10.0, 1, Some(10), 10.0, 48000);
        let cmd = encode_cmd(
            &c,
            &EncodeParams {
                output: Path::new("out.mp4"),
                width: 640,
                height: 480,
                out_fps: "30000/1001",
                pix_fmt: "yuv420p",
                out_pix_fmt: "yuv420p",
                audio: Some((Path::new("/t/a.wav"), &plan)),
                color: &color,
            },
        );
        let s = shell_join(&cmd);
        assert!(s.contains("-f rawvideo -pix_fmt yuv420p -video_size 640x480 -framerate 30000/1001 -i pipe:0"), "{s}");
        assert!(
            s.contains("-c:v libx265 -crf 16 -preset medium -x265-params log-level=error -pix_fmt yuv420p -color_range tv -colorspace bt709 -movflags +faststart -tag:v hvc1 -y out.mp4"),
            "{s}"
        );
        assert!(s.contains("-map 1:a:0 -af "), "{s}");
        let webm = EncodeParams {
            output: Path::new("x/out.webm"),
            width: 2,
            height: 2,
            out_fps: "25",
            pix_fmt: "rgb24",
            out_pix_fmt: "yuv420p",
            audio: None,
            color: &ColorInfo::default(),
        };
        let s = shell_join(&encode_cmd(&c, &webm));
        assert!(s.contains("-an -c:v libvpx-vp9 -crf 30 -b:v 0 -row-mt 1 -pix_fmt yuv420p -y x/out.webm"), "{s}");
        // x264 only on request, and then no hvc1 tag
        c.vcodec = Some("libx264".into());
        c.vcodec_args = Some(vec!["-crf".into(), "18".into()]);
        let s = shell_join(&encode_cmd(&c, &EncodeParams { output: Path::new("o.mp4"), audio: None, ..webm }));
        assert!(s.contains("-c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart -y o.mp4"), "{s}");
        assert!(!s.contains("hvc1"));
        // mkv: x265 without mp4 flags
        c.vcodec = None;
        c.vcodec_args = None;
        let s = shell_join(&encode_cmd(&c, &EncodeParams { output: Path::new("o.mkv"), ..webm }));
        assert!(s.contains("-c:v libx265 -crf 16 -preset medium -x265-params log-level=error -pix_fmt yuv420p -y o.mkv"), "{s}");
    }

    #[test]
    fn decode_commands() {
        let mut c = Config::from_env().unwrap();
        c.in_args.clear();
        c.decode_args.clear();
        c.temp_vcodec = None;
        c.temp_vcodec_args = None;
        let dec = Decode {
            pix_fmt: "yuv420p".into(),
            fps: "100/50".into(),
            vf: "crop=1080:1080,framestep=50".into(),
            frames: Some(1080),
            video_in_args: vec!["-ss".into(), "294.595".into()],
            audio_in_args: vec!["-ss".into(), "294.6".into(), "-t".into(), "10.8".into()],
        };
        let s = shell_join(&decode_pipe_cmd(&c, &dec, Path::new("in.mp4")));
        assert!(s.contains("-ss 294.595 -i in.mp4 -map 0:v:0 -an -sn -dn -r 100/50 -vf crop=1080:1080,framestep=50 -pix_fmt yuv420p -frames:v 1080 -f rawvideo pipe:1"), "{s}");
        let s = shell_join(&probe_frame_cmd(&c, &dec, Path::new("in.mp4"), Path::new("p.nut")));
        assert!(s.ends_with("-frames:v 1080 -frames:v 1 -c:v rawvideo -f nut -y p.nut"), "{s}");
        let s = shell_join(&audio_cmd(&c, &dec, Path::new("in.mp4"), Path::new("a.wav")));
        assert!(s.contains("-ss 294.6 -t 10.8 -i in.mp4 -map 0:a:0 -vn -sn -dn -c:a pcm_f32le -y a.wav"), "{s}");
        let pf = PixelFormat::parse("yuv420p").unwrap();
        let s = shell_join(&segment_cmd(&c, &dec, Path::new("in.mp4"), &pf, [1080, 1080], 200, 100.0, Path::new("/t/chunk-%06d.mkv")));
        assert!(s.contains("-c:v libx264 -crf 10 -preset veryfast -force_key_frames 'expr:gte(n,n_forced*200)' -f segment -segment_time 1.995000 -segment_format matroska -reset_timestamps 1 -y /t/chunk-%06d.mkv"), "{s}");
        let s = shell_join(&chunk_cmd(&c, "yuv420p", Path::new("/t/chunk-000001.mkv")));
        assert!(s.ends_with("-i /t/chunk-000001.mkv -map 0:v:0 -an -sn -dn -fps_mode passthrough -pix_fmt yuv420p -f rawvideo pipe:1"), "{s}");
    }

    #[test]
    fn temp_codecs() {
        let f = |n: &str| temp_codec_defaults(&PixelFormat::parse(n).unwrap(), 320, 240).0;
        assert_eq!(f("yuv420p"), "libx264");
        assert_eq!(temp_codec_defaults(&PixelFormat::parse("yuv420p").unwrap(), 321, 239).0, "ffv1");
        assert_eq!(temp_codec_defaults(&PixelFormat::parse("yuv422p").unwrap(), 320, 239).0, "libx264");
        assert_eq!(temp_codec_defaults(&PixelFormat::parse("yuv444p").unwrap(), 321, 239).0, "libx264");
        assert_eq!(f("yuv444p10le"), "libx264");
        assert_eq!(f("gray"), "libx264");
        assert_eq!(f("rgb24"), "libx264rgb");
        assert_eq!(f("bgr0"), "libx264rgb");
        assert_eq!(f("yuv420p12le"), "ffv1");
        assert_eq!(f("yuva420p"), "ffv1");
        assert_eq!(f("rgba"), "ffv1");
        assert_eq!(f("rgb48le"), "ffv1");
    }
}
