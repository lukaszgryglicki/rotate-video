mod clip;
mod config;
mod ffmpeg;
mod geometry;
mod pipeline;
mod pixfmt;
mod sys;
mod util;
mod volume;

use anyhow::{bail, Context, Result};
use config::{Config, Mode, USAGE};
use ffmpeg::{AudioPlan, ColorInfo, Decode, EncodeParams};
use geometry::{depth_kept, fmt_mat, is_planar, output_dims, planar_xy, rotation_matrix, time_reversed, tracked_output, Mat3, Track};
use pixfmt::PixelFormat;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use util::{fmt_bytes, fmt_dur, rate_div, shell_join, Progress};
use volume::{Flat, Frames};

macro_rules! log {
    ($cfg:expr, $($arg:tt)*) => { if !$cfg.quiet { eprintln!($($arg)*); } };
}

struct TempDir {
    path: PathBuf,
    keep: bool,
}

impl TempDir {
    fn create(base: &Path, keep: bool) -> Result<TempDir> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
        let path = base.join(format!("rotatevideo-tmp-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).with_context(|| format!("cannot create temp dir {}", path.display()))?;
        Ok(TempDir { path, keep })
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if self.keep {
            eprintln!("kept temp dir {}", self.path.display());
        } else {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

fn main() {
    sys::install_signal_handlers();
    let args: Vec<String> = std::env::args().collect();
    if args.iter().skip(1).any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return;
    }
    if args.iter().skip(1).any(|a| a == "-V" || a == "--version") {
        println!("rotatevideo {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.len() != 6 {
        eprint!("{USAGE}");
        std::process::exit(2);
    }
    if let Err(e) = run(&args[1..]) {
        eprintln!("rotatevideo: error: {e:#}");
        std::process::exit(1);
    }
}

/// Everything the encoder side needs, independent of how the frames are produced.
struct Job<'a> {
    cfg: &'a Config,
    pf: &'a PixelFormat,
    rot: Mat3,
    fill: Vec<Vec<u8>>,
    output: &'a Path,
    out_fps: String,
    out_fps_f: f64,
    out_pix_fmt: String,
    color: &'a ColorInfo,
    audio: Option<(PathBuf, u32)>,
    audio_reverse: bool,
    in_fps_f: f64,
    nth: usize,
}

impl Job<'_> {
    fn audio_plan(&self, in_frames: Option<usize>, out_frames: Option<usize>) -> Option<AudioPlan> {
        self.audio.as_ref().map(|(_, sr)| {
            ffmpeg::audio_plan(self.cfg, self.audio_reverse, in_frames, self.in_fps_f, self.nth, out_frames, self.out_fps_f, *sr)
        })
    }

    fn encode_cmd(&self, w: usize, h: usize, plan: Option<&AudioPlan>) -> Vec<String> {
        if w > 8192 || h > 8192 {
            log!(self.cfg, "warning: output {w}x{h} exceeds 8192 px; the encoder may refuse it (ROT_CROP=1, ROT_OUTPUT=crop or FF_VF=scale=... shrink it)");
        }
        let audio = self.audio.as_ref().zip(plan).map(|((p, _), plan)| (p.as_path(), plan));
        ffmpeg::encode_cmd(
            self.cfg,
            &EncodeParams {
                output: self.output,
                width: w,
                height: h,
                out_fps: &self.out_fps,
                pix_fmt: &self.pf.name,
                out_pix_fmt: &self.out_pix_fmt,
                audio,
                color: self.color,
            },
        )
    }

    fn out_dims(&self, in_dims: [usize; 3]) -> ([usize; 3], Track) {
        if self.cfg.crop {
            tracked_output(&self.rot, in_dims, self.cfg.output, self.cfg.even_dims)
        } else {
            (output_dims(&self.rot, in_dims, self.cfg.output, self.cfg.even_dims), Track::default())
        }
    }

    fn describe_window(&self, in_dims: [usize; 3], out: [usize; 3], track: &Track) -> Option<String> {
        if !self.cfg.crop {
            return None;
        }
        let full = output_dims(&self.rot, in_dims, self.cfg.output, self.cfg.even_dims);
        if full[..2] == out[..2] {
            return None;
        }
        Some(if track.is_static() {
            format!("window: {}x{} centred (full box {}x{})", out[0], out[1], full[0], full[1])
        } else {
            format!("window: {}x{} follows content, centre moves ({:+.3}, {:+.3}) px/frame (full box {}x{})", out[0], out[1], track.vel[0], track.vel[1], full[0], full[1])
        })
    }

    fn run_volume<F: Frames>(&self, frames: &F, in_dims: [usize; 3]) -> Result<(u64, u64)> {
        let (out, track) = self.out_dims(in_dims);
        log!(self.cfg, "volume: {}x{}x{} -> {}x{}x{}", in_dims[0], in_dims[1], in_dims[2], out[0], out[1], out[2]);
        if let Some(w) = self.describe_window(in_dims, out, &track) {
            log!(self.cfg, "{w}");
        }
        let plan = self.audio_plan(Some(in_dims[2]), Some(out[2]));
        if let Some(p) = &plan {
            log!(self.cfg, "audio: {}", describe_audio(p));
        }
        let enc = self.encode_cmd(out[0], out[1], plan.as_ref());
        let n = pipeline::run_volume(self.cfg, self.pf, &self.rot, in_dims, out, track, self.fill.clone(), frames, &enc)?;
        Ok((in_dims[2] as u64, n))
    }
}

fn describe_audio(p: &AudioPlan) -> String {
    format!(
        "{}speed x{:.4}{}, filter: {}",
        if p.reversed { "reversed, " } else { "" },
        p.speed,
        p.new_duration.map(|d| format!(", {}", fmt_dur(d))).unwrap_or_default(),
        p.filter
    )
}

fn secs(x: f64) -> String {
    format!("{x:.6}")
}

fn run(a: &[String]) -> Result<()> {
    let t0 = Instant::now();
    let input = PathBuf::from(&a[0]);
    let output = PathBuf::from(&a[1]);
    let mut deg = [0f64; 3];
    for (d, s) in deg.iter_mut().zip(&a[2..5]) {
        *d = s.trim().parse().with_context(|| format!("bad angle {s:?}"))?;
    }
    let cfg = Config::from_env()?;
    rayon::ThreadPoolBuilder::new().num_threads(cfg.threads).build_global().ok();
    if !input.is_file() {
        bail!("input {} is not a file", input.display());
    }
    if let Some(p) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        if !p.is_dir() {
            bail!("output directory {} does not exist", p.display());
        }
    }

    let info = ffmpeg::probe_input(&cfg, &input)?;
    log!(
        cfg,
        "input: {} {}x{} {} {} fps{}{}, audio: {}",
        info.vcodec,
        info.width,
        info.height,
        info.pix_fmt,
        info.fps,
        info.nb_frames.map(|n| format!(", {n} frames")).unwrap_or_default(),
        info.duration.map(|d| format!(", {}", fmt_dur(d))).unwrap_or_default(),
        info.audio.as_ref().map(|a| format!("{} {} Hz {} ch", a.codec, a.sample_rate, a.channels)).unwrap_or_else(|| "none".into())
    );

    let rot = rotation_matrix(deg, cfg.order);
    let reversed = time_reversed(&rot);
    let planar = is_planar(&rot) && depth_kept(cfg.output);
    log!(
        cfg,
        "rotation: X {} Y {} Z {} deg (order {:?}) {} time {}",
        deg[0],
        deg[1],
        deg[2],
        cfg.order,
        fmt_mat(&rot),
        if reversed { "reversed" } else { "forward" }
    );

    let mut pix_fmt_name = if cfg.pix_fmt.eq_ignore_ascii_case("auto") {
        if PixelFormat::parse(&info.pix_fmt).is_ok() { info.pix_fmt.clone() } else { "rgb24".to_string() }
    } else {
        cfg.pix_fmt.clone()
    };
    // ffmpeg >= 8 decodes the deprecated yuvj* formats as yuv* + full range
    let mut color = info.color.clone();
    if let Some(rest) = pix_fmt_name.strip_prefix("yuvj") {
        pix_fmt_name = format!("yuv{rest}");
        color.range = Some("pc".into());
    }
    let pf = PixelFormat::parse(&pix_fmt_name)?;
    let range = color.range.as_deref();
    let full_range = range == Some("pc") || (pf.is_gray() && range != Some("tv"));
    let fill = pf.fill(&cfg.fill_spec, full_range)?;
    let out_pix_fmt = if cfg.out_pix_fmt.eq_ignore_ascii_case("auto") { pf.encoder_default() } else { cfg.out_pix_fmt.clone() };
    let out_fps = cfg.out_fps.clone().unwrap_or_else(|| info.fps.clone());
    let out_fps_f = ffmpeg::parse_rate(&out_fps).ok_or_else(|| anyhow::anyhow!("bad FF_OUT_FPS {out_fps:?}"))?;

    let tmp = TempDir::create(&ffmpeg::temp_base(&cfg, &output), cfg.keep_temp)?;

    // Real decoded frame size (after FF_VF) from a single frame.
    let mut dec = Decode { pix_fmt: pix_fmt_name.clone(), fps: info.fps.clone(), vf: cfg.vf.clone().unwrap_or_default(), ..Default::default() };
    let nut = tmp.path.join("probe.nut");
    let probe_cmd = ffmpeg::probe_frame_cmd(&cfg, &dec, &input, &nut);
    ffmpeg::run(&probe_cmd).context("decoding one frame to determine the frame size")?;
    let (w, h, got) = ffmpeg::probe_frame_size(&cfg, &nut)?;
    if got != pix_fmt_name {
        bail!("decoder produced {got} frames instead of {pix_fmt_name}");
    }
    let _ = std::fs::remove_file(&nut);

    let exact_count = cfg.in_args.is_empty() && cfg.fps.is_none() && cfg.vf.is_none() && info.nb_frames.is_some();
    let total: Option<usize> = if cfg.fps.is_none() { info.nb_frames.map(|n| n as usize) } else { None }
        .or_else(|| info.duration.map(|d| (d * info.fps_f).round() as usize))
        .filter(|n| *n > 0);
    let plan = clip::plan(cfg.clip, w, h, total, cfg.even_dims)?;
    log!(
        cfg,
        "decoded frames: {}x{} {}, {}{}; clip: {}",
        w,
        h,
        pix_fmt_name,
        total.map(|t| t.to_string()).unwrap_or_else(|| "unknown number of".into()),
        if exact_count { "" } else { " (estimated)" },
        plan.desc
    );

    let mut vf: Vec<String> = cfg.vf.iter().cloned().collect();
    if let Some((cw, ch)) = plan.crop {
        vf.push(format!("crop={cw}:{ch}"));
    }
    if plan.nth > 1 {
        vf.push(format!("framestep={}", plan.nth));
        dec.fps = rate_div(&info.fps, plan.nth);
    }
    dec.vf = vf.join(",");
    dec.frames = plan.take;
    if plan.start > 0 {
        // a quarter frame early: the wanted frame is the first one at/after the seek point
        dec.video_in_args = vec!["-ss".into(), secs((plan.start as f64 - 0.25) / info.fps_f)];
        dec.audio_in_args = vec!["-ss".into(), secs(plan.start as f64 / info.fps_f)];
    }
    if let Some(t) = plan.take {
        dec.audio_in_args.extend(["-t".into(), secs(t as f64 / info.fps_f)]);
    }
    let (cw, ch) = (plan.w, plan.h);
    let frame_bytes = pf.frame_bytes(cw, ch);
    let dec_fps_f = info.fps_f / plan.nth as f64;
    let in_frames_known = plan.take.or(if exact_count { plan.frames } else { None });

    let raw_est = plan.frames.map(|n| n as u64 * frame_bytes as u64);
    let phys = sys::phys_mem();
    let budget = phys.map(|p| (p as f64 * cfg.in_memory_pct / 100.0) as u64);
    let mode = match cfg.mode {
        Mode::Auto if planar => Mode::Stream,
        Mode::Auto => match (raw_est, budget) {
            (Some(r), Some(b)) if r <= b => Mode::Ram,
            _ => Mode::Disk,
        },
        Mode::Stream if !planar => bail!(
            "ROT_MODE=stream needs a rotation that keeps the time axis (X/Y multiples of 180 with any Z) and ROT_OUTPUT without a fixed depth"
        ),
        m => m,
    };
    let hint = "ROT_CLIP=min|middle|nth, FF_VF=scale=..., or ROT_FORCE=1";
    match mode {
        Mode::Ram => {
            if let (Some(r), Some(b)) = (raw_est, budget) {
                if r > b && !cfg.force {
                    bail!("raw volume {} exceeds the RAM budget {} ({}% of {}); use ROT_MODE=disk, {hint}", fmt_bytes(r), fmt_bytes(b), cfg.in_memory_pct, fmt_bytes(phys.unwrap_or(0)));
                }
            }
        }
        Mode::Disk => {
            if let (Some(r), Some(f)) = (raw_est, sys::free_space(&tmp.path)) {
                if r + r / 20 > f && !cfg.force {
                    bail!("raw volume {} does not fit the {} free in {}; use ROT_TMPDIR=..., {hint}", fmt_bytes(r), fmt_bytes(f), tmp.path.display());
                }
            }
        }
        _ => {}
    }
    log!(
        cfg,
        "mode: {}{}{}",
        format!("{mode:?}").to_ascii_lowercase(),
        raw_est.map(|r| format!(", raw volume {} ({} per frame)", fmt_bytes(r), fmt_bytes(frame_bytes as u64))).unwrap_or_default(),
        match mode {
            Mode::Ram => budget.map(|b| format!(", RAM budget {}", fmt_bytes(b))).unwrap_or_default(),
            Mode::Disk => format!(", temp dir {}", tmp.path.display()),
            _ => String::new(),
        }
    );

    let has_audio = info.audio.is_some() && !cfg.no_audio;
    let audio_path = tmp.path.join(format!("audio.{}", cfg.audio_ext));
    let audio_cmd = has_audio.then(|| ffmpeg::audio_cmd(&cfg, &dec, &input, &audio_path));
    let job = Job {
        cfg: &cfg,
        pf: &pf,
        rot,
        fill,
        output: &output,
        out_fps: out_fps.clone(),
        out_fps_f,
        out_pix_fmt,
        color: &color,
        audio: has_audio.then(|| (audio_path.clone(), info.audio.as_ref().map(|a| a.sample_rate).unwrap_or(48000))),
        audio_reverse: cfg.audio_reverse.unwrap_or(reversed),
        in_fps_f: info.fps_f,
        nth: plan.nth,
    };
    let chunk = ((cfg.chunk_mb as u64 * 1048576) / frame_bytes as u64).clamp(1, 1000) as usize;
    let raw_path = tmp.path.join(format!("frames.{pix_fmt_name}"));
    let chunk_pattern = tmp.path.join("chunk-%06d.mkv");

    if cfg.dry_run {
        println!("# frame size probe\n{}", shell_join(&probe_cmd));
        match mode {
            Mode::Stream if reversed => println!("# transcode into chunks of {chunk} frames\n{}", shell_join(&ffmpeg::segment_cmd(&cfg, &dec, &input, &pf, [cw, ch], chunk, dec_fps_f, &chunk_pattern))),
            Mode::Stream => println!("# decode (streamed)\n{}", shell_join(&ffmpeg::decode_pipe_cmd(&cfg, &dec, &input))),
            Mode::Ram => println!("# decode (into RAM)\n{}", shell_join(&ffmpeg::decode_pipe_cmd(&cfg, &dec, &input))),
            Mode::Disk | Mode::Auto => println!("# decode (to disk)\n{}", shell_join(&ffmpeg::decode_file_cmd(&cfg, &dec, &input, &raw_path))),
        }
        if let Some(c) = &audio_cmd {
            println!("# audio dump\n{}", shell_join(c));
        }
        let d_est = plan.frames.unwrap_or(1);
        let (out, in_frames, out_frames, window) = if mode == Mode::Stream {
            let rxy = planar_xy(&rot);
            (output_dims(&rxy, [cw, ch, 1], cfg.output, cfg.even_dims), in_frames_known, in_frames_known, None)
        } else {
            let (o, track) = job.out_dims([cw, ch, d_est]);
            let window = job.describe_window([cw, ch, d_est], o, &track);
            (o, Some(d_est), Some(o[2]), window)
        };
        let aplan = job.audio_plan(in_frames, out_frames);
        println!("# encode {}x{}{}", out[0], out[1], if mode == Mode::Stream { String::new() } else { format!("x{} (from {}x{}x{})", out[2], cw, ch, d_est) });
        if let Some(w) = window {
            println!("# {w}");
        }
        if let Some(p) = &aplan {
            println!("# audio: {}", describe_audio(p));
        }
        println!("{}", shell_join(&job.encode_cmd(out[0], out[1], aplan.as_ref())));
        return Ok(());
    }

    if let Some(c) = &audio_cmd {
        log!(cfg, "audio: dumping to {}", audio_path.display());
        ffmpeg::run(c).context("dumping audio")?;
    }

    let (n_in, n_out) = match mode {
        Mode::Stream => {
            let rxy = planar_xy(&rot);
            let out = output_dims(&rxy, [cw, ch, 1], cfg.output, cfg.even_dims);
            log!(cfg, "stream: {}x{} -> {}x{} per frame{}", cw, ch, out[0], out[1], if reversed { format!(", reversed via chunks of {chunk} frames") } else { String::new() });
            let aplan = job.audio_plan(in_frames_known, in_frames_known);
            if let Some(p) = &aplan {
                log!(cfg, "audio: {}", describe_audio(p));
            }
            let enc = job.encode_cmd(out[0], out[1], aplan.as_ref());
            let total = plan.frames.map(|n| n as u64);
            if reversed {
                let seg = ffmpeg::segment_cmd(&cfg, &dec, &input, &pf, [cw, ch], chunk, dec_fps_f, &chunk_pattern);
                pipeline::run_stream_reversed(&cfg, &pf, &rxy, [cw, ch], [out[0], out[1]], job.fill.clone(), &seg, &tmp.path, chunk, &enc, total)?
            } else {
                let dcmd = ffmpeg::decode_pipe_cmd(&cfg, &dec, &input);
                pipeline::run_stream_forward(&cfg, &pf, &rxy, [cw, ch], [out[0], out[1]], job.fill.clone(), &dcmd, &enc, total)?
            }
        }
        Mode::Ram => {
            let dcmd = ffmpeg::decode_pipe_cmd(&cfg, &dec, &input);
            let limit = if cfg.force { None } else { budget.map(|b| b + b / 10) };
            let frames = pipeline::read_all_frames(&dcmd, frame_bytes, limit, Progress::new("decode", plan.frames.map(|n| n as u64), cfg.quiet))?;
            if frames.is_empty() {
                bail!("no frames decoded");
            }
            let d = frames.len();
            job.run_volume(&frames, [cw, ch, d])?
        }
        Mode::Disk | Mode::Auto => {
            let dcmd = ffmpeg::decode_file_cmd(&cfg, &dec, &input, &raw_path);
            let poll_path = raw_path.clone();
            pipeline::run_polling(&dcmd, Progress::new("decode", plan.frames.map(|n| n as u64), cfg.quiet), move || {
                std::fs::metadata(&poll_path).map(|m| m.len() / frame_bytes as u64).unwrap_or(0)
            })?;
            let file = std::fs::File::open(&raw_path)?;
            let len = file.metadata()?.len() as usize;
            if len == 0 {
                bail!("no frames decoded");
            }
            if !len.is_multiple_of(frame_bytes) {
                bail!("raw file size {len} is not a multiple of the frame size {frame_bytes}");
            }
            let mmap = unsafe { memmap2::Mmap::map(&file) }.context("mmap raw frames")?;
            let _ = mmap.advise(memmap2::Advice::WillNeed);
            let d = len / frame_bytes;
            job.run_volume(&Flat { data: &mmap, frame_bytes }, [cw, ch, d])?
        }
    };

    let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    log!(cfg, "done: {n_in} frames in, {n_out} frames out, {} written to {} in {}", fmt_bytes(size), output.display(), fmt_dur(t0.elapsed().as_secs_f64()));
    Ok(())
}
