use crate::config::Config;
use crate::ffmpeg;
use crate::geometry::{Mat3, Track};
use crate::pixfmt::PixelFormat;
use crate::sys::check_interrupted;
use crate::util::{fmt_bytes, shell_join, Progress};
use crate::volume::{Flat, Frames, Renderer};
use anyhow::{bail, Context, Result};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{sync_channel, Receiver, Sender, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const QUEUE: usize = 3;

fn spawn(cmd: &[String], stdin: Stdio, stdout: Stdio) -> Result<Child> {
    Command::new(&cmd[0])
        .args(&cmd[1..])
        .stdin(stdin)
        .stdout(stdout)
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("cannot run {}", cmd[0]))
}

fn status_err(what: &str, st: ExitStatus, cmd: &[String]) -> anyhow::Error {
    anyhow::anyhow!("{what} failed ({st}): {}", shell_join(cmd))
}

/// Ok(true) = full frame, Ok(false) = clean EOF before the frame, Err = truncated frame.
pub fn read_frame<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => {
                if got == 0 {
                    return Ok(false);
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("decoder stopped mid-frame ({got} of {} bytes)", buf.len()),
                ));
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

/// Encoder process fed from a bounded queue by a writer thread.
pub struct Sink {
    cmd: Vec<String>,
    child: Option<Child>,
    full_tx: Option<SyncSender<Vec<u8>>>,
    empty_rx: Receiver<Vec<u8>>,
    writer: Option<JoinHandle<io::Result<u64>>>,
    frame_bytes: usize,
}

impl Sink {
    pub fn spawn(cmd: &[String], frame_bytes: usize) -> Result<Sink> {
        let mut child = spawn(cmd, Stdio::piped(), Stdio::inherit())?;
        let mut stdin = child.stdin.take().context("encoder stdin")?;
        let (full_tx, full_rx) = sync_channel::<Vec<u8>>(QUEUE);
        let (empty_tx, empty_rx) = sync_channel::<Vec<u8>>(QUEUE);
        for _ in 0..QUEUE {
            empty_tx.send(vec![0u8; frame_bytes]).unwrap();
        }
        let writer = thread::spawn(move || -> io::Result<u64> {
            let mut n = 0u64;
            for buf in full_rx {
                stdin.write_all(&buf)?;
                n += 1;
                if empty_tx.send(buf).is_err() {
                    break;
                }
            }
            stdin.flush()?;
            drop(stdin);
            Ok(n)
        });
        Ok(Sink { cmd: cmd.to_vec(), child: Some(child), full_tx: Some(full_tx), empty_rx, writer: Some(writer), frame_bytes })
    }

    /// A free output buffer (blocks while the queue is full).
    pub fn buffer(&mut self) -> Result<Vec<u8>> {
        match self.empty_rx.recv() {
            Ok(mut b) => {
                b.resize(self.frame_bytes, 0);
                Ok(b)
            }
            Err(_) => Err(self.fail()),
        }
    }

    pub fn push(&mut self, buf: Vec<u8>) -> Result<()> {
        match self.full_tx.as_ref().expect("sink finished").send(buf) {
            Ok(()) => Ok(()),
            Err(_) => Err(self.fail()),
        }
    }

    fn fail(&mut self) -> anyhow::Error {
        self.full_tx.take();
        let werr = self.writer.take().and_then(|w| w.join().ok()).and_then(|r| r.err());
        let st = self.child.take().and_then(|mut c| c.wait().ok());
        match (st, werr) {
            (Some(st), _) if !st.success() => status_err("encoder", st, &self.cmd),
            (_, Some(e)) => anyhow::anyhow!("writing to encoder: {e}"),
            _ => anyhow::anyhow!("encoder stopped unexpectedly"),
        }
    }

    /// Closes the pipe, waits for the encoder, returns the number of frames written.
    pub fn finish(mut self) -> Result<u64> {
        self.full_tx.take();
        let n = match self.writer.take().unwrap().join() {
            Ok(Ok(n)) => n,
            Ok(Err(e)) => {
                let st = self.child.take().map(|mut c| c.wait());
                if let Some(Ok(st)) = st {
                    if !st.success() {
                        return Err(status_err("encoder", st, &self.cmd));
                    }
                }
                bail!("writing to encoder: {e}");
            }
            Err(_) => bail!("encoder writer thread panicked"),
        };
        let st = self.child.take().unwrap().wait().context("waiting for encoder")?;
        if !st.success() {
            return Err(status_err("encoder", st, &self.cmd));
        }
        Ok(n)
    }
}

impl Drop for Sink {
    fn drop(&mut self) {
        self.full_tx.take();
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Decoder process read by a reader thread into a bounded queue of frames.
pub struct Source {
    cmd: Vec<String>,
    child: Option<Child>,
    full_rx: Receiver<Vec<u8>>,
    empty_tx: Option<Sender<Vec<u8>>>,
    reader: Option<JoinHandle<io::Result<u64>>>,
}

impl Source {
    pub fn spawn(cmd: &[String], frame_bytes: usize) -> Result<Source> {
        let mut child = spawn(cmd, Stdio::null(), Stdio::piped())?;
        let mut stdout = child.stdout.take().context("decoder stdout")?;
        let (full_tx, full_rx) = sync_channel::<Vec<u8>>(QUEUE);
        let (empty_tx, empty_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        for _ in 0..QUEUE {
            empty_tx.send(vec![0u8; frame_bytes]).unwrap();
        }
        let reader = thread::spawn(move || -> io::Result<u64> {
            let mut n = 0u64;
            loop {
                let mut buf = match empty_rx.recv() {
                    Ok(b) => b,
                    Err(_) => return Ok(n),
                };
                buf.resize(frame_bytes, 0);
                if !read_frame(&mut stdout, &mut buf)? {
                    return Ok(n);
                }
                n += 1;
                if full_tx.send(buf).is_err() {
                    return Ok(n);
                }
            }
        });
        Ok(Source { cmd: cmd.to_vec(), child: Some(child), full_rx, empty_tx: Some(empty_tx), reader: Some(reader) })
    }

    /// Next decoded frame, None at end of stream (call `finish` to learn whether it was clean).
    pub fn next(&mut self) -> Option<Vec<u8>> {
        self.full_rx.recv().ok()
    }

    pub fn recycle(&mut self, buf: Vec<u8>) {
        if let Some(tx) = &self.empty_tx {
            let _ = tx.send(buf);
        }
    }

    /// Frames delivered; fails when the decoder exited abnormally or mid-frame.
    pub fn finish(mut self) -> Result<u64> {
        self.empty_tx.take();
        let res = self.reader.take().unwrap().join();
        let mut child = self.child.take().unwrap();
        match res {
            Ok(Ok(n)) => {
                let st = child.wait().context("waiting for decoder")?;
                if !st.success() {
                    return Err(status_err("decoder", st, &self.cmd));
                }
                Ok(n)
            }
            Ok(Err(e)) => {
                let _ = child.kill();
                let _ = child.wait();
                bail!("reading from decoder: {e}")
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                bail!("decoder reader thread panicked")
            }
        }
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.empty_tx.take();
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Runs a command to completion while showing progress from `poll` (frames done so far).
pub fn run_polling(cmd: &[String], mut progress: Progress, poll: impl Fn() -> u64) -> Result<()> {
    let mut child = spawn(cmd, Stdio::null(), Stdio::inherit())?;
    let st = loop {
        if let Some(st) = child.try_wait().context("waiting for ffmpeg")? {
            break st;
        }
        if let Err(e) = check_interrupted() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }
        progress.update(poll());
        thread::sleep(Duration::from_millis(250));
    };
    if !st.success() {
        return Err(status_err("ffmpeg", st, cmd));
    }
    let done = poll();
    progress.finish(if done == 0 { progress.total().unwrap_or(0) } else { done });
    Ok(())
}

/// Decodes everything into RAM. Aborts (killing the decoder) when the data would exceed `budget`.
pub fn read_all_frames(cmd: &[String], frame_bytes: usize, budget: Option<u64>, mut progress: Progress) -> Result<Vec<Vec<u8>>> {
    let mut child = spawn(cmd, Stdio::null(), Stdio::piped())?;
    let mut stdout = child.stdout.take().context("decoder stdout")?;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    loop {
        if let Err(e) = check_interrupted() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }
        if let Some(b) = budget {
            if (frames.len() as u64 + 1) * frame_bytes as u64 > b {
                let _ = child.kill();
                let _ = child.wait();
                bail!(
                    "input does not fit into the RAM budget of {} ({} frames read); use ROT_MODE=disk, ROT_CLIP=..., FF_VF=scale=..., or ROT_IN_MEMORY / ROT_FORCE=1",
                    fmt_bytes(b),
                    frames.len()
                );
            }
        }
        let mut buf = vec![0u8; frame_bytes];
        match read_frame(&mut stdout, &mut buf) {
            Ok(true) => {
                frames.push(buf);
                progress.update(frames.len() as u64);
            }
            Ok(false) => break,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                bail!("reading from decoder: {e}");
            }
        }
    }
    drop(stdout);
    let st = child.wait().context("waiting for decoder")?;
    if !st.success() {
        return Err(status_err("decoder", st, cmd));
    }
    progress.finish(frames.len() as u64);
    Ok(frames)
}

/// Common volume path: rotate `frames` (in_dims) into out_dims and feed the encoder.
#[allow(clippy::too_many_arguments)]
pub fn run_volume<F: Frames>(
    cfg: &Config,
    pf: &PixelFormat,
    rot: &Mat3,
    in_dims: [usize; 3],
    out_dims: [usize; 3],
    track: Track,
    fill: Vec<Vec<u8>>,
    frames: &F,
    enc_cmd: &[String],
) -> Result<u64> {
    let r = Renderer::new(pf, rot, in_dims, out_dims, track, cfg.interp, fill);
    let mut sink = Sink::spawn(enc_cmd, r.out_frame_bytes())?;
    let mut progress = Progress::new("encode", Some(out_dims[2] as u64), cfg.quiet);
    for z in 0..out_dims[2] {
        check_interrupted()?;
        let mut buf = sink.buffer()?;
        r.render_frame(frames, z, &mut buf);
        sink.push(buf)?;
        progress.update(z as u64 + 1);
    }
    let n = sink.finish()?;
    progress.finish(n);
    Ok(n)
}

/// Streaming path (time kept): decoder → per-frame XY rotation → encoder. Returns (in, out) frames.
#[allow(clippy::too_many_arguments)]
pub fn run_stream_forward(
    cfg: &Config,
    pf: &PixelFormat,
    rot_xy: &Mat3,
    in_wh: [usize; 2],
    out_wh: [usize; 2],
    fill: Vec<Vec<u8>>,
    dec_cmd: &[String],
    enc_cmd: &[String],
    total: Option<u64>,
) -> Result<(u64, u64)> {
    let r = Renderer::new(pf, rot_xy, [in_wh[0], in_wh[1], 1], [out_wh[0], out_wh[1], 1], Track::default(), cfg.interp, fill);
    let in_bytes = r.in_frame_bytes();
    let mut src = Source::spawn(dec_cmd, in_bytes)?;
    let mut sink = Sink::spawn(enc_cmd, r.out_frame_bytes())?;
    let mut progress = Progress::new("stream", total, cfg.quiet);
    let mut n = 0u64;
    while let Some(frame) = src.next() {
        check_interrupted()?;
        let mut out = sink.buffer()?;
        r.render_frame(&Flat { data: &frame, frame_bytes: in_bytes }, 0, &mut out);
        sink.push(out)?;
        src.recycle(frame);
        n += 1;
        progress.update(n);
    }
    let n_in = src.finish()?;
    let n_out = sink.finish()?;
    progress.finish(n_out);
    Ok((n_in, n_out))
}

fn decode_chunk(cfg: &Config, pix_fmt: &str, path: &Path, frame_bytes: usize) -> Result<Vec<Vec<u8>>> {
    let cmd = ffmpeg::chunk_cmd(cfg, pix_fmt, path);
    let out = Command::new(&cmd[0])
        .args(&cmd[1..])
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .with_context(|| format!("cannot run {}", cmd[0]))?;
    if !out.status.success() {
        return Err(status_err("chunk decoder", out.status, &cmd));
    }
    if !out.stdout.len().is_multiple_of(frame_bytes) {
        bail!("chunk {} decoded to {} bytes, not a multiple of the frame size {}", path.display(), out.stdout.len(), frame_bytes);
    }
    Ok(out.stdout.chunks_exact(frame_bytes).map(|c| c.to_vec()).collect())
}

pub fn list_chunks(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.starts_with("chunk-") && n.ends_with(".mkv")
        })
        .collect();
    v.sort();
    Ok(v)
}

/// Streaming path with reversed time: chunked intermediate, chunks consumed last to first.
#[allow(clippy::too_many_arguments)]
pub fn run_stream_reversed(
    cfg: &Config,
    pf: &PixelFormat,
    rot_xy: &Mat3,
    in_wh: [usize; 2],
    out_wh: [usize; 2],
    fill: Vec<Vec<u8>>,
    seg_cmd: &[String],
    tmp: &Path,
    chunk_frames: usize,
    enc_cmd: &[String],
    total: Option<u64>,
) -> Result<(u64, u64)> {
    let r = Renderer::new(pf, rot_xy, [in_wh[0], in_wh[1], 1], [out_wh[0], out_wh[1], 1], Track::default(), cfg.interp, fill);
    let in_bytes = r.in_frame_bytes();
    let tmp_owned = tmp.to_path_buf();
    run_polling(seg_cmd, Progress::new("chunks", total, cfg.quiet), move || {
        list_chunks(&tmp_owned).map(|v| v.len().saturating_sub(1) as u64 * chunk_frames as u64).unwrap_or(0)
    })?;
    let chunks = list_chunks(tmp)?;
    if chunks.is_empty() {
        bail!("no chunks produced (empty input?)");
    }
    let pix_fmt = pf.name.clone();
    let decode = |p: PathBuf| -> JoinHandle<Result<Vec<Vec<u8>>>> {
        let cfg = cfg.clone();
        let pix_fmt = pix_fmt.clone();
        thread::spawn(move || decode_chunk(&cfg, &pix_fmt, &p, in_bytes))
    };
    let mut sink = Sink::spawn(enc_cmd, r.out_frame_bytes())?;
    let mut progress = Progress::new("stream", total, cfg.quiet);
    let mut n_in = 0u64;
    let mut pending = Some(decode(chunks[chunks.len() - 1].clone()));
    for i in (0..chunks.len()).rev() {
        let frames = pending.take().unwrap().join().map_err(|_| anyhow::anyhow!("chunk decoder thread panicked"))??;
        if i > 0 {
            pending = Some(decode(chunks[i - 1].clone()));
        }
        for frame in frames.iter().rev() {
            check_interrupted()?;
            let mut out = sink.buffer()?;
            r.render_frame(&Flat { data: frame, frame_bytes: in_bytes }, 0, &mut out);
            sink.push(out)?;
            n_in += 1;
            progress.update(n_in);
        }
        drop(frames);
        if !cfg.keep_temp {
            let _ = std::fs::remove_file(&chunks[i]);
        }
    }
    let n_out = sink.finish()?;
    progress.finish(n_out);
    Ok((n_in, n_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_frame_eof_handling() {
        let data = vec![1u8; 10];
        let mut cur = io::Cursor::new(data);
        let mut buf = [0u8; 4];
        assert!(read_frame(&mut cur, &mut buf).unwrap());
        assert!(read_frame(&mut cur, &mut buf).unwrap());
        assert!(read_frame(&mut cur, &mut buf).is_err());
        let mut cur = io::Cursor::new(vec![1u8; 8]);
        assert!(read_frame(&mut cur, &mut buf).unwrap());
        assert!(read_frame(&mut cur, &mut buf).unwrap());
        assert!(!read_frame(&mut cur, &mut buf).unwrap());
    }

    #[test]
    fn chunk_listing_sorted() {
        let d = std::env::temp_dir().join(format!("rotatevideo-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for n in ["chunk-000002.mkv", "chunk-000000.mkv", "other.txt", "chunk-000001.mkv"] {
            std::fs::write(d.join(n), b"").unwrap();
        }
        let names: Vec<String> = list_chunks(&d).unwrap().iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["chunk-000000.mkv", "chunk-000001.mkv", "chunk-000002.mkv"]);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
