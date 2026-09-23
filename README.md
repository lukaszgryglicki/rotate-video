# rotate-video

`rotatevideo` treats a video as a 3D volume — **X = width, Y = height, Z = frames (time)** — and rotates
that volume around all three axes. A 2‑minute FHD clip at 30 fps is a 1920×1080×3600 block of voxels;
rotate it 90° around X and you get a 1920×3600 video that is 1080 frames long, where the former time axis
now runs top to bottom.

```
rotatevideo input.mp4 output.mp4 30 40 60      # X=30°, Y=40°, Z=60°
```

Pipeline: `ffprobe` → `ffmpeg` decodes raw frames (the input's own yuv format, so the decode itself is
lossless) and dumps the audio to WAV → the volume is resampled into the rotated frame (all CPU cores,
trilinear or nearest) → frames are piped into `ffmpeg`, encoded as **x265 CRF 16** by default, with the
audio time‑scaled to the new frame count and reversed when the time axis flips. Temp files are removed at
the end (Ctrl‑C included).

## Install

Requires Rust (cargo) and `ffmpeg`/`ffprobe` (with libx265) in `PATH`.

```
make                            # target/release/rotatevideo
make install                    # stripped binary in /data/scripts (INSTALL_DIR=... to change)
make test                       # unit tests + clippy
make clean                      # drop build intermediates, keep the release/debug binaries
```

## Usage

```
rotatevideo INPUT OUTPUT XDEG YDEG ZDEG
```

Angles are in degrees, right‑hand rule, axes X = right, Y = down, Z = forward in time. Rotations are applied
X, then Y, then Z (`ROT_ORDER`).

| command | result |
|---|---|
| `rotatevideo in.mp4 out.mp4 0 0 90` | 90° clockwise (W×H swapped) |
| `rotatevideo in.mp4 out.mp4 180 0 0` | vertically flipped and played backwards, audio reversed |
| `rotatevideo in.mp4 out.mp4 0 180 0` | mirrored and played backwards, audio reversed |
| `rotatevideo in.mp4 out.mp4 90 0 0` | time becomes height: W×D video, H frames long |
| `rotatevideo in.mp4 out.mp4 0 90 0` | time becomes width: D×H video, W frames long |
| `rotatevideo in.mp4 out.mp4 30 40 60` | arbitrary 3D rotation, output is the bounding box |

The output volume is the **bounding box** of the rotated input (nothing clipped, empty space black —
`ROT_FILL`). `ROT_OUTPUT=crop` keeps the input size, `ROT_OUTPUT=WxHxD` sets it explicitly
(`0` = fit, `-1` = input, per dimension).

`ROT_DRY_RUN=1` prints geometry, mode, sizes and the exact ffmpeg commands without processing anything.

## Modes

The tool picks how to hold the data (`ROT_MODE=auto`):

* **stream** — the rotation keeps the time axis (any Z angle, X/Y multiples of 180°: 90° turns, flips,
  reversal). Frames are rotated one by one, nothing is stored: any length works.
  Reversed time (`180 0 0`, `0 180 0`, …) is streamed from a compact intermediate transcode
  (libx264 CRF 10, chunks of `ROT_CHUNK_MB`) that is read from the end; `FF_TEMP_VCODEC=ffv1` or
  `FF_TEMP_VCODEC_ARGS="-qp 0"` make it lossless at the cost of temp space.
* **ram** — true 3D rotations, when the raw volume fits `ROT_IN_MEMORY` % (80) of physical RAM.
* **disk** — otherwise: raw frames go to a temp file next to the output (`ROT_TMPDIR`) and are mmapped.
  Works for anything the disk holds, but mixing time with a spatial axis becomes disk‑bound. The tool
  refuses when the free space is too small (`ROT_FORCE=1` overrides).

Raw size is `W × H × bytes/pixel × frames` — 1.5 bytes for yuv420p, 3 for yuv444p/rgb24, 2× for 10‑bit.
1920×1080 at 100 fps is ≈ 311 MB per second of video, so 7 minutes ≈ 129 GB. Shrink the volume with
`ROT_CLIP`, `FF_VF=scale=960:-2`, `FF_IN_ARGS="-ss 60 -t 20"` or `FF_FPS=25`.

## Clipping the input volume (`ROT_CLIP`)

Long videos are corridors (1920×1080×60000), and rotating a corridor mostly produces black. `ROT_CLIP`
cuts a centred part of the input before rotating:

| `ROT_CLIP` | 1920×1080×60000 becomes | meaning |
|---|---|---|
| `none` (default) | 1920×1080×60000 | whole video |
| `min` | 1080×1080×1080 | cube, side = min(W, H, frames); centred, 1080 frames around the middle (10.8 s at 100 fps) |
| `middle` | 1920×1080×1920 | every side ≤ the middle value of (W, H, frames); the middle 19.2 s |
| `nth:50` | 1920×1080×1200 | every 50th frame, fps unchanged (10 min → 12 s) |
| `nth` | 1920×1080×1875 | N = ceil(frames / max(W, H)) = 32 |

`min` and `middle` take the time window from the middle of the video by default; `min:start` / `middle:start`
take it from the beginning and `min:end` / `middle:end` from the end (a 10‑minute video clipped to 2 minutes
gives 4:00–6:00, 0:00–2:00 or 8:00–10:00). `ROT_CLIP_TIME=start|middle|end` sets the default when no suffix is given.
`ROT_NTH=N` is an alternative to `nth:N`. Clipping is done by ffmpeg (`crop`, `framestep`, `-ss`,
`-frames:v`) so only the clipped part is ever decoded; the audio is cut to the same window. With `nth` the
audio is sped up to match (10 min of sound in 12 s), like any other frame‑count change.

## Audio

Audio is dumped losslessly, then stretched/compressed so its duration equals `frames / fps` of the output
(`atempo`, pitch preserving; `ROT_AUDIO_MODE=rate` for tape‑style speed, `none` to leave it) and reversed
when the rotated time axis points backwards (`ROT_AUDIO_REVERSE` to force). When time maps onto a
spatial axis (e.g. `90 0 0`) the audio is simply rescaled to the new length.

## Configuration (environment variables)

`ROT_*`:

| variable | default | meaning |
|---|---|---|
| `ROT_CLIP` / `ROT_NTH` | `none` | `none`, `min[:start\|:middle\|:end]`, `middle[:…]`, `nth[:N]`, see above |
| `ROT_CLIP_TIME` | `middle` | default time anchor for `min`/`middle`: `start`, `middle`, `end` |
| `ROT_MODE` | `auto` | `stream`, `ram`, `disk`, `auto` |
| `ROT_IN_MEMORY` | `80` | % of physical RAM the raw volume may use in `ram` mode |
| `ROT_CHUNK_MB` | `512` | decoded frames held per chunk in the reversed streaming path |
| `ROT_FORCE` | `0` | ignore RAM/disk size checks |
| `ROT_THREADS` | all cores | worker threads |
| `ROT_INTERP` | `trilinear` | `trilinear` or `nearest` |
| `ROT_OUTPUT` | `fit` | `fit`, `crop`, or `WxHxD` (`0` = fit, `-1` = input) |
| `ROT_ORDER` | `xyz` | order in which the rotations are applied |
| `ROT_FILL` | `0,0,0` | fill colour: `R,G,B[,A]`, `#RRGGBB[AA]` or a gray value (converted to the yuv format) |
| `ROT_EVEN_DIMS` | `1` | round output W/H up to even numbers (x265/x264 need them for 4:2:0) |
| `ROT_AUDIO_MODE` | `tempo` | `tempo`, `rate` or `none` |
| `ROT_AUDIO_REVERSE` | `auto` | `auto`, `1`, `0` |
| `ROT_TMPDIR` | output's directory | where the temp dir is created (`/tmp` is often too small) |
| `ROT_KEEP_TEMP` | `0` | keep the temp dir |
| `ROT_DRY_RUN` | `0` | probe and print the plan only |
| `ROT_QUIET` | `0` | no progress output |

`FF_*` (every `*_ARGS` accepts shell‑like quoting, `none` clears a default):

| variable | default | meaning |
|---|---|---|
| `FF_FFMPEG` / `FF_FFPROBE` | `ffmpeg` / `ffprobe` | binaries |
| `FF_LOGLEVEL` | `error` | ffmpeg `-loglevel` |
| `FF_GLOBAL_ARGS` | | extra global args for every invocation |
| `FF_IN_ARGS` | | input args before `-i INPUT` for decode and audio (`-ss 10 -t 20`, `-hwaccel auto`) |
| `FF_VMAP` / `FF_AMAP` | `0:v:0` / `0:a:0` | stream selection |
| `FF_FPS` | probed | decode frame rate (CFR) |
| `FF_PIX_FMT` | `auto` | intermediate format: the input's own when supported, else `rgb24`. Supported: `yuv{444,422,420,440,411,410}p[{9..16}le]`, `yuva…`, `yuvj…`, `gray[{9..16}le]`, `rgb24 bgr24 rgba bgra argb abgr rgb0 bgr0 0rgb 0bgr ya8` and their 16‑bit variants |
| `FF_VF` | | extra decode filters, e.g. `scale=960:-2` (frame size is re‑probed) |
| `FF_DECODE_ARGS` | | extra output args for the frame decode |
| `FF_TEMP_VCODEC` / `FF_TEMP_VCODEC_ARGS` | `libx264` `-crf 10 -preset veryfast` (`ffv1` where x264 can't take the format) | intermediate codec of the reversed streaming path |
| `FF_AUDIO_CODEC` / `FF_AUDIO_EXT` / `FF_AUDIO_ARGS` | `pcm_f32le` / `wav` | lossless audio dump |
| `FF_NO_AUDIO` | `0` | drop audio |
| `FF_OUT_FPS` | decode fps | output frame rate |
| `FF_VCODEC` | `libx265` | `libvpx-vp9` for webm, `mpeg4` for avi; e.g. `libx264` |
| `FF_VCODEC_ARGS` | `-crf 16 -preset medium` | x265; vp9: `-crf 30 -b:v 0 -row-mt 1`; lossless: `FF_VCODEC=ffv1 FF_VCODEC_ARGS=none` |
| `FF_OUT_PIX_FMT` | `auto` | intermediate yuv format (10‑bit stays 10‑bit) or `yuv420p`; `none` = encoder's choice |
| `FF_ACODEC` / `FF_ACODEC_ARGS` | `aac -b:a 192k` | `libopus` for webm, `libmp3lame` for avi |
| `FF_AUDIO_FILTERS` | | extra audio filters appended to the generated chain |
| `FF_COLOR_ARGS` | input tags | `-color_range/-colorspace/-color_primaries/-color_trc` copied from the input |
| `FF_ENCODE_ARGS` | `-movflags +faststart -tag:v hvc1` (mp4/mov) | extra output args |

## Examples

```
rotatevideo in.mp4 out.mp4 0 180 0                        # mirrored + reversed, streamed, any length
FF_IN_ARGS="-ss 00:06:00" rotatevideo in.mp4 out.mp4 0 180 0        # same, but only from 6:00 to the end
FF_IN_ARGS="-ss 00:06:00 -to 00:08:30" rotatevideo in.mp4 o.mp4 0 180 0   # 6:00-8:30 only
ROT_CLIP=min rotatevideo in.mp4 cube.mp4 30 40 60          # 3D rotation of the central cube
ROT_CLIP=min:end rotatevideo in.mp4 cube.mp4 30 40 60      # same cube, but from the last seconds of the video
ROT_CLIP=nth:50 rotatevideo in.mp4 fast.mp4 45 0 0         # every 50th frame, then tilt time into height
ROT_CLIP=middle ROT_MODE=disk rotatevideo in.mp4 o.mp4 0 90 0
FF_VCODEC=libx264 FF_VCODEC_ARGS="-crf 18" rotatevideo in.mp4 o.mp4 0 0 90
FF_TEMP_VCODEC=ffv1 FF_VCODEC=ffv1 FF_VCODEC_ARGS=none rotatevideo in.mp4 lossless.mkv 180 0 0
```

`example_vids/` (not committed) holds outputs of the test runs, each with a `.sh` containing the
command that produced it.

## License

Apache-2.0, see [LICENSE](LICENSE).
