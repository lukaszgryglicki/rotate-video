//! Volume sampling: every output voxel is inverse-mapped into the input volume, plane by plane.

use crate::geometry::{Geometry, Mat3};
use crate::pixfmt::{PixelFormat, Region};
use anyhow::{bail, Result};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Nearest,
    Trilinear,
}

pub fn parse_interp(s: &str) -> Result<Interp> {
    match s.trim().to_ascii_lowercase().as_str() {
        "nearest" | "nn" | "point" => Ok(Interp::Nearest),
        "trilinear" | "linear" => Ok(Interp::Trilinear),
        _ => bail!("ROT_INTERP must be nearest or trilinear, got {s:?}"),
    }
}

pub trait Sample: Copy + Send + Sync {
    const BYTES: usize;
    const MAX: f32;
    fn load(b: &[u8]) -> f32;
    fn store(v: f32, b: &mut [u8]);
}

#[derive(Clone, Copy)]
pub struct U8;
#[derive(Clone, Copy)]
pub struct U16Le;

impl Sample for U8 {
    const BYTES: usize = 1;
    const MAX: f32 = 255.0;
    #[inline(always)]
    fn load(b: &[u8]) -> f32 {
        b[0] as f32
    }
    #[inline(always)]
    fn store(v: f32, b: &mut [u8]) {
        b[0] = (v + 0.5).clamp(0.0, Self::MAX) as u8;
    }
}

impl Sample for U16Le {
    const BYTES: usize = 2;
    const MAX: f32 = 65535.0;
    #[inline(always)]
    fn load(b: &[u8]) -> f32 {
        u16::from_le_bytes([b[0], b[1]]) as f32
    }
    #[inline(always)]
    fn store(v: f32, b: &mut [u8]) {
        let x = (v + 0.5).clamp(0.0, Self::MAX) as u16;
        b[0..2].copy_from_slice(&x.to_le_bytes());
    }
}

/// Random access to the raw input frames.
pub trait Frames: Sync {
    fn frame(&self, z: usize) -> &[u8];
}

/// Tightly packed consecutive frames (mmap, RAM copy, or a single frame).
pub struct Flat<'a> {
    pub data: &'a [u8],
    pub frame_bytes: usize,
}

impl Frames for Flat<'_> {
    #[inline(always)]
    fn frame(&self, z: usize) -> &[u8] {
        &self.data[z * self.frame_bytes..(z + 1) * self.frame_bytes]
    }
}

impl Frames for Vec<Vec<u8>> {
    #[inline(always)]
    fn frame(&self, z: usize) -> &[u8] {
        &self[z]
    }
}

pub struct Renderer<'a> {
    pf: &'a PixelFormat,
    geoms: Vec<Geometry>,
    in_planes: Vec<Region>,
    out_planes: Vec<Region>,
    interp: Interp,
    fill: Vec<Vec<u8>>,
}

impl<'a> Renderer<'a> {
    pub fn new(
        pf: &'a PixelFormat,
        rot: &Mat3,
        in_dims: [usize; 3],
        out_dims: [usize; 3],
        interp: Interp,
        fill: Vec<Vec<u8>>,
    ) -> Self {
        assert_eq!(fill.len(), pf.planes.len());
        Renderer {
            pf,
            geoms: pf.planes.iter().map(|p| Geometry::plane(rot, in_dims, out_dims, p.sx, p.sy)).collect(),
            in_planes: pf.layout(in_dims[0], in_dims[1]),
            out_planes: pf.layout(out_dims[0], out_dims[1]),
            interp,
            fill,
        }
    }

    pub fn in_frame_bytes(&self) -> usize {
        self.in_planes.iter().map(|r| r.bytes).sum()
    }

    pub fn out_frame_bytes(&self) -> usize {
        self.out_planes.iter().map(|r| r.bytes).sum()
    }

    /// Render output frame `z` into `out` (exactly one output frame). Rows run in parallel.
    pub fn render_frame<F: Frames>(&self, frames: &F, z: usize, out: &mut [u8]) {
        assert_eq!(out.len(), self.out_frame_bytes());
        for (p, reg) in self.out_planes.iter().enumerate() {
            let bpp = self.pf.bpp(p);
            out[reg.offset..reg.offset + reg.bytes]
                .par_chunks_mut(reg.w * bpp)
                .enumerate()
                .for_each(|(y, row)| self.render_row(frames, p, y, z, row));
        }
    }

    fn render_row<F: Frames>(&self, frames: &F, p: usize, y: usize, z: usize, row: &mut [u8]) {
        let bpp = self.pf.bpp(p);
        for px in row.chunks_exact_mut(bpp) {
            px.copy_from_slice(&self.fill[p]);
        }
        let g = &self.geoms[p];
        let q0 = g.map(0.0, y as f64, z as f64);
        let dq = g.dx();
        let margin = match self.interp {
            Interp::Nearest => 0.0,
            Interp::Trilinear => 0.5,
        };
        let Some((x0, x1)) = row_range(g, q0, dq, margin) else {
            return;
        };
        match self.interp {
            Interp::Nearest => self.nearest_row(frames, p, q0, dq, row, x0, x1),
            Interp::Trilinear => match (self.pf.bytes, self.pf.planes[p].channels) {
                (1, 1) => self.trilinear_row::<F, U8, 1>(frames, p, q0, dq, row, x0, x1),
                (1, 2) => self.trilinear_row::<F, U8, 2>(frames, p, q0, dq, row, x0, x1),
                (1, 3) => self.trilinear_row::<F, U8, 3>(frames, p, q0, dq, row, x0, x1),
                (1, 4) => self.trilinear_row::<F, U8, 4>(frames, p, q0, dq, row, x0, x1),
                (2, 1) => self.trilinear_row::<F, U16Le, 1>(frames, p, q0, dq, row, x0, x1),
                (2, 2) => self.trilinear_row::<F, U16Le, 2>(frames, p, q0, dq, row, x0, x1),
                (2, 3) => self.trilinear_row::<F, U16Le, 3>(frames, p, q0, dq, row, x0, x1),
                (2, 4) => self.trilinear_row::<F, U16Le, 4>(frames, p, q0, dq, row, x0, x1),
                _ => unreachable!("pixel formats have 1-4 channels of 1-2 bytes"),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn nearest_row<F: Frames>(
        &self,
        frames: &F,
        p: usize,
        q0: [f64; 3],
        dq: [f64; 3],
        row: &mut [u8],
        x0: usize,
        x1: usize,
    ) {
        let bpp = self.pf.bpp(p);
        let reg = self.in_planes[p];
        let [w, h, d] = self.geoms[p].in_dims;
        let (wi, hi, di) = (w as i64, h as i64, d as i64);
        for x in x0..x1 {
            let t = x as f64;
            let qx = (q0[0] + t * dq[0]).floor() as i64;
            let qy = (q0[1] + t * dq[1]).floor() as i64;
            let qz = (q0[2] + t * dq[2]).floor() as i64;
            if qx < 0 || qy < 0 || qz < 0 || qx >= wi || qy >= hi || qz >= di {
                continue;
            }
            let src = reg.offset + (qy as usize * w + qx as usize) * bpp;
            row[x * bpp..(x + 1) * bpp].copy_from_slice(&frames.frame(qz as usize)[src..src + bpp]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn trilinear_row<F: Frames, S: Sample, const C: usize>(
        &self,
        frames: &F,
        p: usize,
        q0: [f64; 3],
        dq: [f64; 3],
        row: &mut [u8],
        x0: usize,
        x1: usize,
    ) {
        let bpp = C * S::BYTES;
        let reg = self.in_planes[p];
        let [w, h, d] = self.geoms[p].in_dims;
        let (wi, hi, di) = (w as i64, h as i64, d as i64);
        let fillf: [f32; C] = std::array::from_fn(|c| S::load(&self.fill[p][c * S::BYTES..]));
        for x in x0..x1 {
            let t = x as f64;
            let qx = q0[0] + t * dq[0] - 0.5;
            let qy = q0[1] + t * dq[1] - 0.5;
            let qz = q0[2] + t * dq[2] - 0.5;
            let (fx, fy, fz) = (qx.floor(), qy.floor(), qz.floor());
            let (tx, ty, tz) = ((qx - fx) as f32, (qy - fy) as f32, (qz - fz) as f32);
            let (ix, iy, iz) = (fx as i64, fy as i64, fz as i64);
            let wx = [1.0 - tx, tx];
            let wy = [1.0 - ty, ty];
            let wz = [1.0 - tz, tz];
            let mut acc = [0f32; C];
            for (dz, &wz1) in wz.iter().enumerate() {
                if wz1 == 0.0 {
                    continue;
                }
                let sz = iz + dz as i64;
                if sz < 0 || sz >= di {
                    for c in 0..C {
                        acc[c] += wz1 * fillf[c];
                    }
                    continue;
                }
                let plane = &frames.frame(sz as usize)[reg.offset..reg.offset + reg.bytes];
                for (dy, &wy1) in wy.iter().enumerate() {
                    let wzy = wz1 * wy1;
                    if wzy == 0.0 {
                        continue;
                    }
                    let sy = iy + dy as i64;
                    for (dx, &wx1) in wx.iter().enumerate() {
                        let wgt = wzy * wx1;
                        if wgt == 0.0 {
                            continue;
                        }
                        let sx = ix + dx as i64;
                        if sx >= 0 && sy >= 0 && sx < wi && sy < hi {
                            let px = &plane[(sy as usize * w + sx as usize) * bpp..];
                            for c in 0..C {
                                acc[c] += wgt * S::load(&px[c * S::BYTES..]);
                            }
                        } else {
                            for c in 0..C {
                                acc[c] += wgt * fillf[c];
                            }
                        }
                    }
                }
            }
            let dst = &mut row[x * bpp..(x + 1) * bpp];
            for c in 0..C {
                S::store(acc[c], &mut dst[c * S::BYTES..]);
            }
        }
    }
}

/// Output x range whose sample points can touch the (margin-expanded) input box.
fn row_range(g: &Geometry, q0: [f64; 3], dq: [f64; 3], margin: f64) -> Option<(usize, usize)> {
    let w = g.out_dims[0] as f64;
    let mut tmin = 0.0f64;
    let mut tmax = w;
    for a in 0..3 {
        let lo = -margin;
        let hi = g.in_dims[a] as f64 + margin;
        if dq[a].abs() < 1e-12 {
            if q0[a] < lo || q0[a] > hi {
                return None;
            }
        } else {
            let t1 = (lo - q0[a]) / dq[a];
            let t2 = (hi - q0[a]) / dq[a];
            tmin = tmin.max(t1.min(t2));
            tmax = tmax.min(t1.max(t2));
        }
    }
    if tmin >= tmax {
        return None;
    }
    let x0 = (tmin.floor() - 1.0).max(0.0) as usize;
    let x1 = (tmax.ceil() + 1.0).min(w) as usize;
    (x0 < x1).then_some((x0, x1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{output_dims, planar_xy, rotation_matrix, Axis, OutputSize};

    const XYZ: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
    const W: usize = 4;
    const H: usize = 3;
    const D: usize = 2;

    fn volume() -> Vec<u8> {
        let mut v = Vec::new();
        for z in 0..D {
            for y in 0..H {
                for x in 0..W {
                    v.extend_from_slice(&[(x * 10) as u8, (y * 10) as u8, (z * 10 + 100) as u8]);
                }
            }
        }
        v
    }

    fn render_all(deg: [f64; 3], interp: Interp, size: OutputSize) -> (Vec<u8>, [usize; 3]) {
        let pf = PixelFormat::parse("rgb24").unwrap();
        let r = rotation_matrix(deg, XYZ);
        let out_dims = output_dims(&r, [W, H, D], size, false);
        let rd = Renderer::new(&pf, &r, [W, H, D], out_dims, interp, pf.fill("0", false).unwrap());
        let vol = volume();
        let frames = Flat { data: &vol, frame_bytes: W * H * 3 };
        let fsz = rd.out_frame_bytes();
        let mut out = vec![0u8; fsz * out_dims[2]];
        for z in 0..out_dims[2] {
            rd.render_frame(&frames, z, &mut out[z * fsz..(z + 1) * fsz]);
        }
        (out, out_dims)
    }

    fn px(buf: &[u8], dims: [usize; 3], x: usize, y: usize, z: usize) -> [u8; 3] {
        let i = ((z * dims[1] + y) * dims[0] + x) * 3;
        [buf[i], buf[i + 1], buf[i + 2]]
    }

    #[test]
    fn identity_is_exact() {
        for interp in [Interp::Nearest, Interp::Trilinear] {
            let (out, dims) = render_all([0.0, 0.0, 0.0], interp, OutputSize::Fit);
            assert_eq!(dims, [W, H, D]);
            assert_eq!(out, volume(), "{interp:?}");
        }
    }

    #[test]
    fn z90_is_clockwise_on_screen() {
        for interp in [Interp::Nearest, Interp::Trilinear] {
            let (out, dims) = render_all([0.0, 0.0, 90.0], interp, OutputSize::Fit);
            assert_eq!(dims, [H, W, D]);
            let vol = volume();
            for z in 0..D {
                for y in 0..W {
                    for x in 0..H {
                        // out(x', y') = in(x = y', y = H-1-x')
                        assert_eq!(px(&out, dims, x, y, z), px(&vol, [W, H, D], y, H - 1 - x, z), "{interp:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn x180_flips_height_and_time() {
        for interp in [Interp::Nearest, Interp::Trilinear] {
            let (out, dims) = render_all([180.0, 0.0, 0.0], interp, OutputSize::Fit);
            assert_eq!(dims, [W, H, D]);
            let vol = volume();
            for z in 0..D {
                for y in 0..H {
                    for x in 0..W {
                        assert_eq!(px(&out, dims, x, y, z), px(&vol, [W, H, D], x, H - 1 - y, D - 1 - z), "{interp:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn x90_turns_time_into_height() {
        let (out, dims) = render_all([90.0, 0.0, 0.0], Interp::Nearest, OutputSize::Fit);
        assert_eq!(dims, [W, D, H]);
        let vol = volume();
        // Rx(90): Y -> +Z, Z -> -Y, so out(x, y', z') = in(x, y = z', z = D-1-y')
        for z in 0..H {
            for y in 0..D {
                for x in 0..W {
                    assert_eq!(px(&out, dims, x, y, z), px(&vol, [W, H, D], x, z, D - 1 - y));
                }
            }
        }
    }

    #[test]
    fn fit_rotation_has_black_corners_and_keeps_content() {
        let (out, dims) = render_all([0.0, 0.0, 45.0], Interp::Trilinear, OutputSize::Fit);
        assert_eq!(dims[2], D);
        assert!(dims[0] >= 4 && dims[1] >= 4);
        assert_eq!(px(&out, dims, 0, 0, 0), [0, 0, 0]);
        let nonzero = out.chunks(3).filter(|p| p != &[0, 0, 0]).count();
        assert!(nonzero > 0);
    }

    #[test]
    fn crop_keeps_input_dims() {
        let (_, dims) = render_all([30.0, 40.0, 60.0], Interp::Trilinear, OutputSize::Crop);
        assert_eq!(dims, [W, H, D]);
    }

    // yuv420p volume 4x4x2: Y = 10x + y + 100z, Cb = 50 + 2x + y + 10z, Cr = 200 - ...
    const YW: usize = 4;
    const YH: usize = 4;

    fn yuv_frame(z: usize) -> Vec<u8> {
        let mut f = Vec::new();
        for y in 0..YH {
            for x in 0..YW {
                f.push((10 * x + y + 100 * z) as u8);
            }
        }
        for y in 0..2 {
            for x in 0..2 {
                f.push((50 + 2 * x + y + 10 * z) as u8);
            }
        }
        for y in 0..2 {
            for x in 0..2 {
                f.push((200 - 5 * x - 3 * y - 10 * z) as u8);
            }
        }
        f
    }

    fn yuv_render(deg: [f64; 3], interp: Interp) -> (Vec<Vec<u8>>, [usize; 3]) {
        let pf = PixelFormat::parse("yuv420p").unwrap();
        let r = rotation_matrix(deg, XYZ);
        let out_dims = output_dims(&r, [YW, YH, D], OutputSize::Fit, true);
        let rd = Renderer::new(&pf, &r, [YW, YH, D], out_dims, interp, pf.fill("0", false).unwrap());
        let frames: Vec<Vec<u8>> = (0..D).map(yuv_frame).collect();
        let out = (0..out_dims[2])
            .map(|z| {
                let mut o = vec![0u8; rd.out_frame_bytes()];
                rd.render_frame(&frames, z, &mut o);
                o
            })
            .collect();
        (out, out_dims)
    }

    #[test]
    fn yuv420p_identity_and_flips_are_exact() {
        for interp in [Interp::Nearest, Interp::Trilinear] {
            let (out, dims) = yuv_render([0.0, 0.0, 0.0], interp);
            assert_eq!(dims, [YW, YH, D]);
            assert_eq!(out, (0..D).map(yuv_frame).collect::<Vec<_>>(), "{interp:?}");

            // Ry(180) = hflip + time reversal, on every plane.
            let (out, dims) = yuv_render([0.0, 180.0, 0.0], interp);
            assert_eq!(dims, [YW, YH, D]);
            for z in 0..D {
                let src = yuv_frame(D - 1 - z);
                let (o, s) = (&out[z], &src);
                for y in 0..YH {
                    for x in 0..YW {
                        assert_eq!(o[y * YW + x], s[y * YW + (YW - 1 - x)], "{interp:?} luma");
                    }
                }
                for y in 0..2 {
                    for x in 0..2 {
                        assert_eq!(o[16 + y * 2 + x], s[16 + y * 2 + (1 - x)], "{interp:?} cb");
                        assert_eq!(o[20 + y * 2 + x], s[20 + y * 2 + (1 - x)], "{interp:?} cr");
                    }
                }
            }
        }
    }

    #[test]
    fn yuv420p_z90_rotates_chroma_consistently() {
        let (out, dims) = yuv_render([0.0, 0.0, 90.0], Interp::Trilinear);
        assert_eq!(dims, [YH, YW, D]);
        for z in 0..D {
            let s = yuv_frame(z);
            for y in 0..YW {
                for x in 0..YH {
                    assert_eq!(out[z][y * YH + x], s[(YH - 1 - x) * YW + y]);
                }
            }
            for y in 0..2 {
                for x in 0..2 {
                    assert_eq!(out[z][16 + y * 2 + x], s[16 + (1 - x) * 2 + y]);
                }
            }
        }
    }

    #[test]
    fn planar_rotation_equals_per_frame_rendering() {
        // Ry(180) via the full volume must equal the streaming path: 2D part per frame, reversed order.
        let pf = PixelFormat::parse("yuv420p").unwrap();
        let r = rotation_matrix([0.0, 180.0, 0.0], XYZ);
        let (full, dims) = yuv_render([0.0, 180.0, 0.0], Interp::Trilinear);
        let r2 = planar_xy(&r);
        let rd = Renderer::new(&pf, &r2, [YW, YH, 1], [dims[0], dims[1], 1], Interp::Trilinear, pf.fill("0", false).unwrap());
        for z in 0..D {
            let src = yuv_frame(D - 1 - z);
            let mut o = vec![0u8; rd.out_frame_bytes()];
            rd.render_frame(&Flat { data: &src, frame_bytes: src.len() }, 0, &mut o);
            assert_eq!(o, full[z]);
        }
    }

    #[test]
    fn yuv_fill_is_black_not_green() {
        let pf = PixelFormat::parse("yuv420p").unwrap();
        let r = rotation_matrix([0.0, 0.0, 45.0], XYZ);
        let out_dims = output_dims(&r, [YW, YH, D], OutputSize::Fit, true);
        let rd = Renderer::new(&pf, &r, [YW, YH, D], out_dims, Interp::Nearest, pf.fill("0", false).unwrap());
        let frames: Vec<Vec<u8>> = (0..D).map(yuv_frame).collect();
        let mut o = vec![0u8; rd.out_frame_bytes()];
        rd.render_frame(&frames, 0, &mut o);
        let l = pf.layout(out_dims[0], out_dims[1]);
        assert_eq!(o[0], 16);
        assert_eq!(o[l[1].offset], 128);
        assert_eq!(o[l[2].offset], 128);
    }
}
