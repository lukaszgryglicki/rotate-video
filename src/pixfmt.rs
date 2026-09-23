//! Raw intermediate pixel formats: packed (one interleaved plane) or planar YUV/gray, 8-16 bit.

use anyhow::{bail, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneKind {
    /// One char per channel: r g b a y (gray) x (padding).
    Packed(&'static str),
    Y,
    Cb,
    Cr,
    A,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    pub kind: PlaneKind,
    pub channels: usize,
    /// log2 horizontal / vertical subsampling.
    pub sx: u8,
    pub sy: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PixelFormat {
    pub name: String,
    pub bytes: usize,
    pub depth: u32,
    pub planes: Vec<Plane>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub offset: usize,
    pub w: usize,
    pub h: usize,
    pub bytes: usize,
}

const PACKED: &[(&str, &str, usize)] = &[
    ("rgb24", "rgb", 1),
    ("bgr24", "bgr", 1),
    ("rgba", "rgba", 1),
    ("bgra", "bgra", 1),
    ("argb", "argb", 1),
    ("abgr", "abgr", 1),
    ("rgb0", "rgbx", 1),
    ("bgr0", "bgrx", 1),
    ("0rgb", "xrgb", 1),
    ("0bgr", "xbgr", 1),
    ("ya8", "ya", 1),
    ("rgb48le", "rgb", 2),
    ("bgr48le", "bgr", 2),
    ("rgba64le", "rgba", 2),
    ("bgra64le", "bgra", 2),
    ("ya16le", "ya", 2),
];

pub const SUPPORTED: &str = "rgb24 bgr24 rgba bgra argb abgr rgb0 bgr0 0rgb 0bgr ya8 rgb48le bgr48le rgba64le bgra64le ya16le, \
gray gray{9,10,12,14,16}le, yuv{444,422,420,440,411,410}p[{9,10,12,14,16}le], yuva{444,422,420}p[{9,10,12,14,16}le], \
yuvj{444,422,420}p";

fn depth_suffix(s: &str) -> Option<(u32, usize)> {
    if s.is_empty() {
        return Some((8, 1));
    }
    let d: u32 = s.strip_suffix("le")?.parse().ok()?;
    (9..=16).contains(&d).then_some((d, 2))
}

impl PixelFormat {
    pub fn parse(name: &str) -> Result<PixelFormat> {
        let n = name.trim().to_ascii_lowercase();
        if let Some((_, layout, bytes)) = PACKED.iter().find(|(pn, _, _)| *pn == n) {
            let plane = Plane { kind: PlaneKind::Packed(layout), channels: layout.len(), sx: 0, sy: 0 };
            return Ok(PixelFormat { name: n, bytes: *bytes, depth: 8 * *bytes as u32, planes: vec![plane] });
        }
        if let Some((depth, bytes)) = n.strip_prefix("gray").and_then(depth_suffix) {
            let plane = Plane { kind: PlaneKind::Y, channels: 1, sx: 0, sy: 0 };
            return Ok(PixelFormat { name: n, bytes, depth, planes: vec![plane] });
        }
        if let Some(f) = Self::parse_yuv(&n) {
            return Ok(f);
        }
        bail!("unsupported pixel format {name:?}; supported: {SUPPORTED}")
    }

    fn parse_yuv(n: &str) -> Option<PixelFormat> {
        let (rest, alpha) = if let Some(r) = n.strip_prefix("yuva") {
            (r, true)
        } else if let Some(r) = n.strip_prefix("yuvj") {
            (r, false)
        } else {
            (n.strip_prefix("yuv")?, false)
        };
        let (sx, sy) = match rest.get(..3)? {
            "444" => (0, 0),
            "422" => (1, 0),
            "420" => (1, 1),
            "440" => (0, 1),
            "411" => (2, 0),
            "410" => (2, 1),
            _ => return None,
        };
        let (depth, bytes) = depth_suffix(rest.get(3..)?.strip_prefix('p')?)?;
        let mut planes = vec![
            Plane { kind: PlaneKind::Y, channels: 1, sx: 0, sy: 0 },
            Plane { kind: PlaneKind::Cb, channels: 1, sx, sy },
            Plane { kind: PlaneKind::Cr, channels: 1, sx, sy },
        ];
        if alpha {
            planes.push(Plane { kind: PlaneKind::A, channels: 1, sx: 0, sy: 0 });
        }
        Some(PixelFormat { name: n.to_string(), bytes, depth, planes })
    }

    pub fn is_yuv(&self) -> bool {
        self.planes.get(1).is_some_and(|p| p.kind == PlaneKind::Cb)
    }

    pub fn is_gray(&self) -> bool {
        self.planes.len() == 1 && self.planes[0].kind == PlaneKind::Y
    }

    pub fn has_alpha(&self) -> bool {
        self.planes.iter().any(|p| p.kind == PlaneKind::A)
    }

    #[cfg(test)]
    pub fn same_layout(&self, other: &PixelFormat) -> bool {
        self.bytes == other.bytes && self.planes == other.planes
    }

    /// Bytes per pixel of one plane.
    pub fn bpp(&self, plane: usize) -> usize {
        self.planes[plane].channels * self.bytes
    }

    pub fn plane_dims(&self, plane: usize, w: usize, h: usize) -> (usize, usize) {
        let p = &self.planes[plane];
        ((w + (1 << p.sx) - 1) >> p.sx, (h + (1 << p.sy) - 1) >> p.sy)
    }

    pub fn layout(&self, w: usize, h: usize) -> Vec<Region> {
        let mut offset = 0;
        (0..self.planes.len())
            .map(|p| {
                let (pw, ph) = self.plane_dims(p, w, h);
                let bytes = pw * ph * self.bpp(p);
                let r = Region { offset, w: pw, h: ph, bytes };
                offset += bytes;
                r
            })
            .collect()
    }

    pub fn frame_bytes(&self, w: usize, h: usize) -> usize {
        self.layout(w, h).iter().map(|r| r.bytes).sum()
    }

    /// Encoder pixel format used for FF_OUT_PIX_FMT=auto.
    pub fn encoder_default(&self) -> String {
        if self.is_yuv() && !self.has_alpha() {
            self.name.clone()
        } else {
            "yuv420p".into()
        }
    }

    /// Per-plane fill voxel from `#RRGGBB[AA]`, `R,G,B[,A]` or a single gray value (8-bit scale).
    pub fn fill(&self, spec: &str, full_range: bool) -> Result<Vec<Vec<u8>>> {
        let s = spec.trim();
        let comps: Vec<u32> = if let Some(hex) = s.strip_prefix('#') {
            if hex.len() != 6 && hex.len() != 8 {
                bail!("ROT_FILL: expected #RRGGBB or #RRGGBBAA, got {spec:?}");
            }
            (0..hex.len() / 2)
                .map(|i| u32::from_str_radix(&hex[2 * i..2 * i + 2], 16))
                .collect::<Result<_, _>>()
                .map_err(|e| anyhow::anyhow!("ROT_FILL: {spec:?}: {e}"))?
        } else {
            s.split(',')
                .map(|p| p.trim().parse::<u32>())
                .collect::<Result<_, _>>()
                .map_err(|e| anyhow::anyhow!("ROT_FILL: {spec:?}: {e}"))?
        };
        if comps.is_empty() || comps.len() > 4 || comps.iter().any(|&c| c > 255) {
            bail!("ROT_FILL: expected 1-4 components in 0..255, got {spec:?}");
        }
        let (r, g, b) = match comps.len() {
            1 | 2 => (comps[0], comps[0], comps[0]),
            _ => (comps[0], comps[1], comps[2]),
        };
        let a = match comps.len() {
            2 => comps[1],
            4 => comps[3],
            _ => 0,
        };
        let maxv = ((1u64 << self.depth) - 1) as f64;
        let sc = (1u64 << (self.depth - 8)) as f64;
        let (rf, gf, bf) = (r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0);
        let yf = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf;
        let (cbf, crf) = ((bf - yf) / 1.8556, (rf - yf) / 1.5748);
        let (yv, cbv, crv) = if full_range {
            let mid = (1u64 << (self.depth - 1)) as f64;
            (yf * maxv, mid + cbf * maxv, mid + crf * maxv)
        } else {
            ((16.0 + 219.0 * yf) * sc, (128.0 + 224.0 * cbf) * sc, (128.0 + 224.0 * crf) * sc)
        };
        let av = a as f64 / 255.0 * maxv;
        let push = |out: &mut Vec<u8>, v: f64| {
            let v = v.round().clamp(0.0, maxv) as u32;
            match self.bytes {
                1 => out.push(v as u8),
                _ => out.extend_from_slice(&(v as u16).to_le_bytes()),
            }
        };
        Ok(self
            .planes
            .iter()
            .map(|p| {
                let mut out = Vec::with_capacity(self.bpp(0));
                match p.kind {
                    PlaneKind::Packed(layout) => {
                        let y8 = ((299 * r + 587 * g + 114 * b) + 500) / 1000;
                        for ch in layout.chars() {
                            let v8 = match ch {
                                'r' => r,
                                'g' => g,
                                'b' => b,
                                'a' => a,
                                'y' => y8,
                                _ => 0,
                            };
                            push(&mut out, v8 as f64 * (maxv / 255.0));
                        }
                    }
                    PlaneKind::Y => push(&mut out, yv),
                    PlaneKind::Cb => push(&mut out, cbv),
                    PlaneKind::Cr => push(&mut out, crv),
                    PlaneKind::A => push(&mut out, av),
                }
                out
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_formats() {
        let f = PixelFormat::parse("yuv420p").unwrap();
        assert_eq!((f.bytes, f.depth, f.planes.len()), (1, 8, 3));
        assert_eq!((f.planes[1].sx, f.planes[1].sy), (1, 1));
        assert!(f.is_yuv() && !f.has_alpha());
        let f = PixelFormat::parse("yuv422p10le").unwrap();
        assert_eq!((f.bytes, f.depth), (2, 10));
        assert_eq!((f.planes[1].sx, f.planes[1].sy), (1, 0));
        let f = PixelFormat::parse("yuva444p").unwrap();
        assert_eq!(f.planes.len(), 4);
        assert!(f.has_alpha());
        assert_eq!(f.encoder_default(), "yuv420p");
        let f = PixelFormat::parse("gray16le").unwrap();
        assert!(f.is_gray() && f.bytes == 2 && f.depth == 16);
        let f = PixelFormat::parse("rgb24").unwrap();
        assert_eq!(f.planes[0].channels, 3);
        assert_eq!(f.bpp(0), 3);
        assert_eq!(f.encoder_default(), "yuv420p");
        assert_eq!(PixelFormat::parse("yuvj420p").unwrap().encoder_default(), "yuvj420p");
        assert!(PixelFormat::parse("nv12").is_err());
        assert!(PixelFormat::parse("yuv420p17le").is_err());
        assert!(PixelFormat::parse("yuv421p").is_err());
    }

    #[test]
    fn layouts() {
        let f = PixelFormat::parse("yuv420p").unwrap();
        assert_eq!(f.frame_bytes(1920, 1080), 1920 * 1080 * 3 / 2);
        let l = f.layout(5, 3);
        assert_eq!(l[0], Region { offset: 0, w: 5, h: 3, bytes: 15 });
        assert_eq!(l[1], Region { offset: 15, w: 3, h: 2, bytes: 6 });
        assert_eq!(l[2], Region { offset: 21, w: 3, h: 2, bytes: 6 });
        assert_eq!(f.frame_bytes(5, 3), 27);
        assert_eq!(PixelFormat::parse("yuv444p16le").unwrap().frame_bytes(4, 4), 4 * 4 * 3 * 2);
        assert_eq!(PixelFormat::parse("rgba").unwrap().frame_bytes(4, 4), 64);
        assert!(PixelFormat::parse("yuv420p").unwrap().same_layout(&PixelFormat::parse("yuvj420p").unwrap()));
    }

    #[test]
    fn fills() {
        let rgb = PixelFormat::parse("rgb24").unwrap();
        assert_eq!(rgb.fill("#ff8000", false).unwrap(), vec![vec![255, 128, 0]]);
        assert_eq!(rgb.fill("1,2,3", false).unwrap(), vec![vec![1, 2, 3]]);
        assert_eq!(rgb.fill("7", false).unwrap(), vec![vec![7, 7, 7]]);
        let bgra = PixelFormat::parse("bgra").unwrap();
        assert_eq!(bgra.fill("#01020304", false).unwrap(), vec![vec![3, 2, 1, 4]]);
        assert_eq!(PixelFormat::parse("gray16le").unwrap().fill("255", true).unwrap(), vec![vec![255, 255]]);
        assert_eq!(PixelFormat::parse("rgb48le").unwrap().fill("255,0,0", false).unwrap(), vec![vec![255, 255, 0, 0, 0, 0]]);
        let yuv = PixelFormat::parse("yuv420p").unwrap();
        assert_eq!(yuv.fill("0,0,0", false).unwrap(), vec![vec![16], vec![128], vec![128]]);
        assert_eq!(yuv.fill("0,0,0", true).unwrap(), vec![vec![0], vec![128], vec![128]]);
        assert_eq!(yuv.fill("255,255,255", false).unwrap(), vec![vec![235], vec![128], vec![128]]);
        assert_eq!(yuv.fill("255,255,255", true).unwrap(), vec![vec![255], vec![128], vec![128]]);
        let red = yuv.fill("255,0,0", false).unwrap();
        assert_eq!(red[0], vec![63]);
        assert!(red[1][0] < 128 && red[2][0] > 200);
        let y10 = PixelFormat::parse("yuv420p10le").unwrap();
        assert_eq!(y10.fill("0", false).unwrap(), vec![vec![64, 0], vec![0, 2], vec![0, 2]]);
        let ya = PixelFormat::parse("yuva420p").unwrap();
        assert_eq!(ya.fill("0,0,0,255", false).unwrap(), vec![vec![16], vec![128], vec![128], vec![255]]);
        assert_eq!(PixelFormat::parse("gray").unwrap().fill("0", true).unwrap(), vec![vec![0]]);
        assert!(rgb.fill("#12", false).is_err());
        assert!(rgb.fill("1,2,3,4,5", false).is_err());
        assert!(rgb.fill("300", false).is_err());
    }
}
