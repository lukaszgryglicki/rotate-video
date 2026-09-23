//! ROT_CLIP: shrink the input volume before rotating (center crop in X/Y, time window at start/middle/end, or every Nth frame).

use crate::config::{Anchor, Clip};
use anyhow::{bail, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipPlan {
    pub w: usize,
    pub h: usize,
    /// Estimated frame count after clipping.
    pub frames: Option<usize>,
    pub nth: usize,
    pub crop: Option<(usize, usize)>,
    /// First input frame to keep.
    pub start: usize,
    /// Exact frame count when time is clipped.
    pub take: Option<usize>,
    pub desc: String,
}

pub fn plan(clip: Clip, w: usize, h: usize, total: Option<usize>, even: bool) -> Result<ClipPlan> {
    match clip {
        Clip::None => Ok(ClipPlan { w, h, frames: total, nth: 1, crop: None, start: 0, take: None, desc: "none".into() }),
        Clip::Min(anchor) | Clip::Middle(anchor) => {
            let is_min = matches!(clip, Clip::Min(_));
            let name = if is_min { "min" } else { "middle" };
            let Some(d) = total else {
                bail!("ROT_CLIP={name} needs a known frame count (the input has no duration); use FF_FPS, ROT_CLIP=none or ROT_CLIP=nth:N")
            };
            let mut s = [w, h, d];
            s.sort_unstable();
            let side = if is_min { s[0] } else { s[1] };
            let ev = |v: usize| if even && v % 2 == 1 && v > 1 { v - 1 } else { v };
            let (cw, ch, cd) = (ev(w.min(side)), ev(h.min(side)), d.min(side));
            let (start, at) = match anchor {
                Anchor::Start => (0, "start"),
                Anchor::Middle => ((d - cd) / 2, "middle"),
                Anchor::End => (d - cd, "end"),
            };
            Ok(ClipPlan {
                w: cw,
                h: ch,
                frames: Some(cd),
                nth: 1,
                crop: (cw != w || ch != h).then_some((cw, ch)),
                start,
                take: (cd < d).then_some(cd),
                desc: format!("{name}:{at} (side {side}): {w}x{h}x{d} -> {cw}x{ch}x{cd}, frames {start}..{}", start + cd),
            })
        }
        Clip::Nth(n) => {
            let n = match (n, total) {
                (Some(n), _) => n,
                (None, Some(d)) => d.div_ceil(w.max(h)).max(1),
                (None, None) => bail!("ROT_CLIP=nth needs ROT_NTH=N (or ROT_CLIP=nth:N): the input frame count is unknown"),
            };
            Ok(ClipPlan {
                w,
                h,
                frames: total.map(|d| d.div_ceil(n)),
                nth: n,
                crop: None,
                start: 0,
                take: None,
                desc: format!("nth (every {n}. frame)"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_examples() {
        let (w, h, d) = (1920, 1080, 60000);
        let p = plan(Clip::Min(Anchor::Middle), w, h, Some(d), true).unwrap();
        assert_eq!((p.w, p.h, p.frames), (1080, 1080, Some(1080)));
        assert_eq!(p.crop, Some((1080, 1080)));
        assert_eq!((p.start, p.take), ((60000 - 1080) / 2, Some(1080)));
        let p = plan(Clip::Middle(Anchor::Middle), w, h, Some(d), true).unwrap();
        assert_eq!((p.w, p.h, p.frames), (1920, 1080, Some(1920)));
        assert_eq!(p.crop, None);
        assert_eq!((p.start, p.take), ((60000 - 1920) / 2, Some(1920)));
        let p = plan(Clip::None, w, h, Some(d), true).unwrap();
        assert_eq!((p.w, p.h, p.frames, p.nth), (1920, 1080, Some(60000), 1));
        let p = plan(Clip::Nth(Some(50)), w, h, Some(d), true).unwrap();
        assert_eq!((p.frames, p.nth, p.take), (Some(1200), 50, None));
        let p = plan(Clip::Nth(None), w, h, Some(d), true).unwrap();
        assert_eq!((p.nth, p.frames), (32, Some(1875)));
    }

    #[test]
    fn time_anchor() {
        let (w, h, d) = (1920, 1080, 60000);
        let p = plan(Clip::Min(Anchor::Start), w, h, Some(d), true).unwrap();
        assert_eq!((p.start, p.take), (0, Some(1080)));
        let p = plan(Clip::Min(Anchor::End), w, h, Some(d), true).unwrap();
        assert_eq!((p.start, p.take), (60000 - 1080, Some(1080)));
        let p = plan(Clip::Middle(Anchor::End), w, h, Some(d), true).unwrap();
        assert_eq!((p.start, p.take, p.frames), (60000 - 1920, Some(1920), Some(1920)));
        assert!(p.desc.starts_with("middle:end"));
        let p = plan(Clip::Middle(Anchor::Start), 1920, 1080, Some(600), true).unwrap();
        assert_eq!((p.start, p.take), (0, None));
        let p = plan(Clip::Min(Anchor::End), 1920, 1080, Some(600), true).unwrap();
        assert_eq!((p.start, p.take), (0, None));
    }

    #[test]
    fn short_video_is_not_cut_in_time() {
        let p = plan(Clip::Min(Anchor::Middle), 1920, 1080, Some(600), true).unwrap();
        assert_eq!((p.w, p.h, p.frames, p.take), (600, 600, Some(600), None));
        let p = plan(Clip::Middle(Anchor::Middle), 1920, 1080, Some(600), true).unwrap();
        assert_eq!((p.w, p.h, p.frames, p.take), (1080, 1080, Some(600), None));
        assert_eq!(p.crop, Some((1080, 1080)));
        let p = plan(Clip::Min(Anchor::Middle), 1919, 1080, Some(1081), true).unwrap();
        assert_eq!((p.w, p.h, p.frames), (1080, 1080, Some(1080)));
        let p = plan(Clip::Min(Anchor::Middle), 5, 5, Some(5), false).unwrap();
        assert_eq!(p.crop, None);
    }

    #[test]
    fn unknown_frame_count() {
        assert!(plan(Clip::Min(Anchor::Middle), 10, 10, None, true).is_err());
        assert!(plan(Clip::Nth(None), 10, 10, None, true).is_err());
        assert_eq!(plan(Clip::Nth(Some(4)), 10, 10, None, true).unwrap().frames, None);
        assert_eq!(plan(Clip::None, 10, 10, None, true).unwrap().frames, None);
    }
}
