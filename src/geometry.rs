//! Rotation geometry. Axes: X = width (right), Y = height (down), Z = frames (time forward).
//! Positive angles follow the right-hand rule around each axis.

use anyhow::{bail, Result};

pub type Mat3 = [[f64; 3]; 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

pub fn parse_order(s: &str) -> Result<[Axis; 3]> {
    let s = s.trim().to_ascii_lowercase();
    let mut seen = [false; 3];
    let mut out = [Axis::X; 3];
    if s.len() != 3 {
        bail!("ROT_ORDER must be a permutation of xyz, got {s:?}");
    }
    for (i, c) in s.chars().enumerate() {
        let a = match c {
            'x' => Axis::X,
            'y' => Axis::Y,
            'z' => Axis::Z,
            _ => bail!("ROT_ORDER must be a permutation of xyz, got {s:?}"),
        };
        if seen[a as usize] {
            bail!("ROT_ORDER must be a permutation of xyz, got {s:?}");
        }
        seen[a as usize] = true;
        out[i] = a;
    }
    Ok(out)
}

pub fn axis_matrix(axis: Axis, deg: f64) -> Mat3 {
    let r = deg.to_radians();
    let (s, c) = r.sin_cos();
    match axis {
        Axis::X => [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        Axis::Y => [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        Axis::Z => [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

pub fn transpose(a: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[j][i];
        }
    }
    m
}

pub fn mat_vec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    let mut r = [0.0; 3];
    for i in 0..3 {
        r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
    }
    r
}

/// Full rotation: rotations are applied in `order` (first element applied first). Entries within
/// 1e-12 of 0 or ±1 are snapped so exact 90° multiples stay exact.
pub fn rotation_matrix(deg: [f64; 3], order: [Axis; 3]) -> Mat3 {
    let mut m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for axis in order {
        let a = axis_matrix(axis, deg[axis as usize]);
        m = mat_mul(&a, &m);
    }
    for v in m.iter_mut().flatten() {
        if v.abs() < 1e-12 {
            *v = 0.0;
        } else if (v.abs() - 1.0).abs() < 1e-12 {
            *v = v.signum();
        }
    }
    m
}

/// True when time maps onto itself (R33 = ±1): every output frame comes from exactly one input frame.
pub fn is_planar(r: &Mat3) -> bool {
    r[2][2].abs() == 1.0
}

/// The in-plane (XY) part of a planar rotation, with time running forward.
pub fn planar_xy(r: &Mat3) -> Mat3 {
    let mut m = *r;
    m[0][2] = 0.0;
    m[1][2] = 0.0;
    m[2] = [0.0, 0.0, 1.0];
    m
}

/// Bounding box of the rotated input box: dim'_i = sum_j |R_ij| * dim_j.
pub fn fit_dims(r: &Mat3, dims: [usize; 3]) -> [usize; 3] {
    let mut out = [0usize; 3];
    for i in 0..3 {
        let e: f64 = (0..3).map(|j| r[i][j].abs() * dims[j] as f64).sum();
        out[i] = (e - 1e-6).ceil().max(1.0) as usize;
    }
    out
}

/// True when the rotated time axis points against the original time axis.
pub fn time_reversed(r: &Mat3) -> bool {
    r[2][2] < -1e-9
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputSize {
    Fit,
    Crop,
    Custom([DimSpec; 3]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimSpec {
    Fit,
    Input,
    Exact(usize),
}

pub fn parse_output_size(s: &str) -> Result<OutputSize> {
    let t = s.trim().to_ascii_lowercase();
    match t.as_str() {
        "fit" | "bbox" | "auto" => return Ok(OutputSize::Fit),
        "crop" | "input" | "same" => return Ok(OutputSize::Crop),
        _ => {}
    }
    let parts: Vec<&str> = t.split('x').collect();
    if parts.len() != 3 {
        bail!("ROT_OUTPUT must be fit | crop | WxHxD (each 0=fit, -1=input, or a positive size), got {s:?}");
    }
    let mut spec = [DimSpec::Fit; 3];
    for (i, p) in parts.iter().enumerate() {
        spec[i] = match p.trim() {
            "0" | "fit" => DimSpec::Fit,
            "-1" | "in" | "input" | "same" => DimSpec::Input,
            n => match n.parse::<usize>() {
                Ok(v) if v > 0 => DimSpec::Exact(v),
                _ => bail!("ROT_OUTPUT: bad dimension {p:?} in {s:?}"),
            },
        };
    }
    Ok(OutputSize::Custom(spec))
}

/// True when the spec leaves the frame count equal to the input's (needed for streaming).
pub fn depth_kept(spec: OutputSize) -> bool {
    match spec {
        OutputSize::Fit | OutputSize::Crop => true,
        OutputSize::Custom(sp) => !matches!(sp[2], DimSpec::Exact(_)),
    }
}

pub fn output_dims(r: &Mat3, in_dims: [usize; 3], spec: OutputSize, even: bool) -> [usize; 3] {
    let fit = fit_dims(r, in_dims);
    let mut out = match spec {
        OutputSize::Fit => fit,
        OutputSize::Crop => in_dims,
        OutputSize::Custom(sp) => {
            let mut d = [0; 3];
            for i in 0..3 {
                d[i] = match sp[i] {
                    DimSpec::Fit => fit[i],
                    DimSpec::Input => in_dims[i],
                    DimSpec::Exact(v) => v,
                };
            }
            d
        }
    };
    if even {
        for d in out.iter_mut().take(2) {
            *d += *d % 2;
        }
    }
    out
}

/// Output window that follows the content: its centre (in luma pixels, relative to the centre of the
/// rotated volume) at output frame z is `pos + vel * (z + 0.5 - zc)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Track {
    pub pos: [f64; 2],
    pub vel: [f64; 2],
    pub zc: f64,
}

impl Track {
    pub fn center(&self, z: f64) -> [f64; 2] {
        let t = z + 0.5 - self.zc;
        [self.pos[0] + self.vel[0] * t, self.pos[1] + self.vel[1] * t]
    }

    pub fn is_static(&self) -> bool {
        self.vel == [0.0, 0.0]
    }
}

/// XY bounding box of the slice of the rotated input box at output depth `z` (coordinates relative
/// to the rotated volume's centre), or None when the plane misses the box.
fn slice_bbox(r: &Mat3, in_dims: [usize; 3], z: f64) -> Option<[f64; 4]> {
    let half = [in_dims[0] as f64 / 2.0, in_dims[1] as f64 / 2.0, in_dims[2] as f64 / 2.0];
    let corner = |i: usize| {
        let v = [
            if i & 1 == 0 { -half[0] } else { half[0] },
            if i & 2 == 0 { -half[1] } else { half[1] },
            if i & 4 == 0 { -half[2] } else { half[2] },
        ];
        mat_vec(r, v)
    };
    let mut bb = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
    let mut add = |p: [f64; 3]| {
        bb[0] = bb[0].min(p[0]);
        bb[1] = bb[1].max(p[0]);
        bb[2] = bb[2].min(p[1]);
        bb[3] = bb[3].max(p[1]);
    };
    for i in 0..8 {
        for bit in [1, 2, 4] {
            if i & bit != 0 {
                continue;
            }
            let (a, b) = (corner(i), corner(i | bit));
            let (da, db) = (a[2] - z, b[2] - z);
            if da * db > 0.0 {
                continue;
            }
            if da == db {
                add(a);
                add(b);
            } else {
                let t = da / (da - db);
                add([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), z]);
            }
        }
    }
    bb[0].is_finite().then_some(bb)
}

/// Output size and window path when the window follows the content instead of covering the whole
/// bounding box. Per axis the window centre moves linearly along the candidate line (static, the
/// projected input axes — the corridor axis for a tilted video — or the least-squares fit through the
/// slice centres) that gives the smallest extent containing every slice; the full box is used where
/// no candidate is smaller.
pub fn tracked_output(r: &Mat3, in_dims: [usize; 3], spec: OutputSize, even: bool) -> ([usize; 3], Track) {
    let base = output_dims(r, in_dims, spec, even);
    let fit = fit_dims(r, in_dims);
    let depth = base[2];
    let zc = depth as f64 / 2.0;
    let slices: Vec<(f64, [f64; 4])> =
        (0..depth).filter_map(|k| slice_bbox(r, in_dims, k as f64 + 0.5 - zc).map(|bb| (k as f64 + 0.5 - zc, bb))).collect();
    let mut track = Track { zc, ..Default::default() };
    let mut size = [fit[0] as f64, fit[1] as f64];
    if !slices.is_empty() {
        let n = slices.len() as f64;
        let tm = slices.iter().map(|(t, _)| t).sum::<f64>() / n;
        let stt: f64 = slices.iter().map(|(t, _)| (t - tm) * (t - tm)).sum();
        let snap = |v: f64| if v.abs() < 1e-9 { 0.0 } else { v };
        for a in 0..2 {
            let c = |bb: &[f64; 4]| (bb[2 * a] + bb[2 * a + 1]) / 2.0;
            let cm = slices.iter().map(|(_, bb)| c(bb)).sum::<f64>() / n;
            let ls_vel = if stt > 0.0 { slices.iter().map(|(t, bb)| (t - tm) * (c(bb) - cm)).sum::<f64>() / stt } else { 0.0 };
            // (pos, vel) candidates: static, input axes through the centre, least squares
            let mut lines = vec![(0.0, 0.0)];
            lines.extend((0..3).filter(|&j| r[2][j].abs() > 1e-9).map(|j| (0.0, r[a][j] / r[2][j])));
            lines.push((cm - ls_vel * tm, ls_vel));
            let mut best: Option<(f64, f64, f64)> = None;
            for (pos, vel) in lines {
                let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                for (t, bb) in &slices {
                    let center = pos + vel * t;
                    lo = lo.min(bb[2 * a] - center);
                    hi = hi.max(bb[2 * a + 1] - center);
                }
                let extent = (hi - lo - 1e-6).ceil().max(1.0);
                if best.is_none_or(|b| extent < b.0) {
                    best = Some((extent, pos + (hi + lo) / 2.0, vel));
                }
            }
            if let Some((extent, pos, vel)) = best {
                if extent < fit[a] as f64 {
                    size[a] = extent;
                    track.pos[a] = snap(pos);
                    track.vel[a] = snap(vel);
                }
            }
        }
    }
    let mut out = base;
    let tracked_dim = |a: usize| {
        let mut d = size[a] as usize;
        if even {
            d += d % 2;
        }
        d
    };
    match spec {
        OutputSize::Fit => {
            out[0] = tracked_dim(0);
            out[1] = tracked_dim(1);
        }
        OutputSize::Crop => {}
        OutputSize::Custom(sp) => {
            for a in 0..2 {
                if sp[a] == DimSpec::Fit {
                    out[a] = tracked_dim(a);
                }
            }
        }
    }
    (out, track)
}

/// Maps output voxel centers of one plane to continuous input coordinates of the same plane.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub in_dims: [usize; 3],
    pub out_dims: [usize; 3],
    /// Inverse rotation in plane units: S⁻¹ Rᵀ S with S = diag(2^sx, 2^sy, 1).
    pub inv: Mat3,
    pub in_center: [f64; 3],
    pub out_center: [f64; 3],
    /// Window path in plane units.
    pub track: Track,
}

impl Geometry {
    #[cfg(test)]
    pub fn new(rot: Mat3, in_dims: [usize; 3], out_dims: [usize; 3]) -> Self {
        Self::plane(&rot, in_dims, out_dims, Track::default(), 0, 0)
    }

    /// Geometry of a plane subsampled by 2^sx x 2^sy; dims and track are full-resolution (luma) values.
    pub fn plane(rot: &Mat3, in_dims: [usize; 3], out_dims: [usize; 3], track: Track, sx: u8, sy: u8) -> Self {
        let s = [(1u32 << sx) as f64, (1u32 << sy) as f64, 1.0];
        let rt = transpose(rot);
        let mut inv = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                inv[i][j] = rt[i][j] * s[j] / s[i];
            }
        }
        let pd = |d: [usize; 3]| [(d[0] + (1 << sx) - 1) >> sx, (d[1] + (1 << sy) - 1) >> sy, d[2]];
        let c = |d: [usize; 3]| [d[0] as f64 / 2.0 / s[0], d[1] as f64 / 2.0 / s[1], d[2] as f64 / 2.0];
        let track = Track {
            pos: [track.pos[0] / s[0], track.pos[1] / s[1]],
            vel: [track.vel[0] / s[0], track.vel[1] / s[1]],
            zc: track.zc,
        };
        Geometry { in_dims: pd(in_dims), out_dims: pd(out_dims), inv, in_center: c(in_dims), out_center: c(out_dims), track }
    }

    /// Continuous input coordinate of the center of output voxel (x, y, z).
    pub fn map(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let w = self.track.center(z);
        let p = [
            x + 0.5 - self.out_center[0] + w[0],
            y + 0.5 - self.out_center[1] + w[1],
            z + 0.5 - self.out_center[2],
        ];
        let q = mat_vec(&self.inv, p);
        [
            q[0] + self.in_center[0],
            q[1] + self.in_center[1],
            q[2] + self.in_center[2],
        ]
    }

    /// Input-space step when moving one output voxel along output X.
    pub fn dx(&self) -> [f64; 3] {
        [self.inv[0][0], self.inv[1][0], self.inv[2][0]]
    }
}

pub fn fmt_mat(m: &Mat3) -> String {
    m.iter()
        .map(|r| format!("[{:+.4} {:+.4} {:+.4}]", r[0], r[1], r[2]))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    const XYZ: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    fn approx(a: [f64; 3], b: [f64; 3]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn identity() {
        let r = rotation_matrix([0.0, 0.0, 0.0], XYZ);
        assert!(approx(mat_vec(&r, [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]));
        assert_eq!(fit_dims(&r, [1920, 1080, 3600]), [1920, 1080, 3600]);
        assert!(!time_reversed(&r));
    }

    #[test]
    fn x90_swaps_height_and_time() {
        let r = rotation_matrix([90.0, 0.0, 0.0], XYZ);
        assert!(approx(mat_vec(&r, [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]));
        assert_eq!(fit_dims(&r, [1920, 1080, 3600]), [1920, 3600, 1080]);
        assert!(!time_reversed(&r));
    }

    #[test]
    fn y90_swaps_width_and_time() {
        let r = rotation_matrix([0.0, 90.0, 0.0], XYZ);
        assert_eq!(fit_dims(&r, [1920, 1080, 3600]), [3600, 1080, 1920]);
    }

    #[test]
    fn z90_swaps_width_and_height() {
        let r = rotation_matrix([0.0, 0.0, 90.0], XYZ);
        assert!(approx(mat_vec(&r, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        assert_eq!(fit_dims(&r, [1920, 1080, 3600]), [1080, 1920, 3600]);
    }

    #[test]
    fn x180_reverses_time() {
        assert!(time_reversed(&rotation_matrix([180.0, 0.0, 0.0], XYZ)));
        assert!(time_reversed(&rotation_matrix([0.0, 180.0, 0.0], XYZ)));
        assert!(!time_reversed(&rotation_matrix([0.0, 0.0, 180.0], XYZ)));
        assert!(time_reversed(&rotation_matrix([100.0, 0.0, 0.0], XYZ)));
        assert!(!time_reversed(&rotation_matrix([80.0, 0.0, 0.0], XYZ)));
    }

    #[test]
    fn order_matters() {
        let a = rotation_matrix([90.0, 90.0, 0.0], XYZ);
        let b = rotation_matrix([90.0, 90.0, 0.0], parse_order("yxz").unwrap());
        assert!(!approx(mat_vec(&a, [1.0, 2.0, 3.0]), mat_vec(&b, [1.0, 2.0, 3.0])));
        assert!(parse_order("xxz").is_err());
        assert!(parse_order("xy").is_err());
    }

    #[test]
    fn inverse_roundtrip() {
        let r = rotation_matrix([30.0, 40.0, 60.0], XYZ);
        let g = Geometry::new(r, [10, 20, 30], [10, 20, 30]);
        let v = [1.0, 2.0, 3.0];
        let back = mat_vec(&g.inv, mat_vec(&r, v));
        assert!(approx(back, v));
        assert!(approx(g.map(4.5, 9.5, 14.5), [5.0, 10.0, 15.0]));
    }

    #[test]
    fn output_sizes() {
        let r = rotation_matrix([0.0, 0.0, 90.0], XYZ);
        assert_eq!(output_dims(&r, [11, 5, 7], OutputSize::Fit, false), [5, 11, 7]);
        assert_eq!(output_dims(&r, [11, 5, 7], OutputSize::Fit, true), [6, 12, 7]);
        assert_eq!(output_dims(&r, [11, 5, 7], OutputSize::Crop, false), [11, 5, 7]);
        let c = parse_output_size("100x-1x0").unwrap();
        assert_eq!(output_dims(&r, [11, 5, 7], c, false), [100, 5, 7]);
        assert!(depth_kept(c));
        assert!(depth_kept(OutputSize::Fit));
        assert!(!depth_kept(parse_output_size("0x0x50").unwrap()));
        assert!(parse_output_size("10x10").is_err());
        assert!(parse_output_size("axbxc").is_err());
        assert_eq!(parse_output_size("crop").unwrap(), OutputSize::Crop);
    }

    #[test]
    fn snapping_and_planar() {
        let r = rotation_matrix([0.0, 180.0, 0.0], XYZ);
        assert_eq!(r, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
        assert!(is_planar(&r));
        assert_eq!(planar_xy(&r), [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!(is_planar(&rotation_matrix([0.0, 0.0, 33.0], XYZ)));
        assert!(is_planar(&rotation_matrix([180.0, 0.0, 45.0], XYZ)));
        assert!(!is_planar(&rotation_matrix([90.0, 0.0, 0.0], XYZ)));
        assert!(!is_planar(&rotation_matrix([1.0, 0.0, 0.0], XYZ)));
        let z90 = rotation_matrix([0.0, 0.0, 90.0], XYZ);
        assert_eq!(z90, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn subsampled_plane_geometry() {
        let r = rotation_matrix([0.0, 0.0, 90.0], XYZ);
        let luma = Geometry::new(r, [8, 6, 2], [6, 8, 2]);
        let chroma = Geometry::plane(&r, [8, 6, 2], [6, 8, 2], Track::default(), 1, 1);
        assert_eq!(chroma.in_dims, [4, 3, 2]);
        assert_eq!(chroma.out_dims, [3, 4, 2]);
        for (x, y, z) in [(0usize, 0usize, 0usize), (2, 3, 1), (1, 2, 0)] {
            let q = luma.map(2.0 * x as f64 + 0.5, 2.0 * y as f64 + 0.5, z as f64);
            let c = chroma.map(x as f64, y as f64, z as f64);
            assert!(approx(c, [q[0] / 2.0, q[1] / 2.0, q[2]]), "{c:?} vs {q:?}");
        }
        let odd = Geometry::plane(&r, [5, 3, 1], [3, 5, 1], Track::default(), 1, 1);
        assert_eq!(odd.in_dims, [3, 2, 1]);
        assert!(approx(odd.in_center, [1.25, 0.75, 0.5]));
    }

    /// True when every output voxel that maps inside the input lies inside the tracked window.
    fn window_contains_content(r: &Mat3, in_dims: [usize; 3], out: [usize; 3], track: Track) -> bool {
        let full = fit_dims(r, in_dims);
        let g = Geometry::plane(r, in_dims, full, Track::default(), 0, 0);
        let w = Geometry::plane(r, in_dims, out, track, 0, 0);
        let inside = |p: [f64; 3]| (0..3).all(|i| p[i] >= 0.0 && p[i] <= in_dims[i] as f64);
        for z in 0..full[2] {
            let c = track.center(z as f64);
            let (ox, oy) = (full[0] as f64 / 2.0 - out[0] as f64 / 2.0 + c[0], full[1] as f64 / 2.0 - out[1] as f64 / 2.0 + c[1]);
            for y in 0..full[1] {
                for x in 0..full[0] {
                    if !inside(g.map(x as f64, y as f64, z as f64)) {
                        continue;
                    }
                    let (wx, wy) = (x as f64 - ox, y as f64 - oy);
                    if wx < -0.5 || wy < -0.5 || wx > out[0] as f64 - 0.5 || wy > out[1] as f64 - 0.5 {
                        return false;
                    }
                    if !approx(w.map(wx, wy, z as f64), g.map(x as f64, y as f64, z as f64)) {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn tracked_identity_and_planar_equal_fit() {
        for deg in [[0.0, 0.0, 0.0], [0.0, 0.0, 90.0], [180.0, 0.0, 30.0]] {
            let r = rotation_matrix(deg, XYZ);
            let (out, track) = tracked_output(&r, [16, 10, 6], OutputSize::Fit, false);
            assert_eq!(out, output_dims(&r, [16, 10, 6], OutputSize::Fit, false), "{deg:?}");
            assert_eq!(track, Track { zc: out[2] as f64 / 2.0, ..Default::default() }, "{deg:?}");
        }
    }

    #[test]
    fn tracked_corridor_shrinks_and_moves() {
        let r = rotation_matrix([10.0, 0.0, 0.0], XYZ);
        let dims = [4, 6, 60];
        let full = output_dims(&r, dims, OutputSize::Fit, false);
        let (out, track) = tracked_output(&r, dims, OutputSize::Fit, false);
        assert_eq!(out[0], full[0]);
        assert_eq!(out[2], full[2]);
        assert!(out[1] < full[1], "{out:?} vs {full:?}");
        assert!(out[1] <= 8, "{out:?}");
        assert_eq!(track.vel[0], 0.0);
        assert!((track.vel[1].abs() - (10f64).to_radians().tan()).abs() < 0.02, "{track:?}");
        assert!(window_contains_content(&r, dims, out, track));
        let (even, _) = tracked_output(&r, dims, OutputSize::Fit, true);
        assert_eq!(even[1] % 2, 0);
    }

    #[test]
    fn tracked_symmetric_cube_is_static_fit() {
        let r = rotation_matrix([45.0, 0.0, 0.0], XYZ);
        let dims = [8, 8, 8];
        let (out, track) = tracked_output(&r, dims, OutputSize::Fit, false);
        let full = output_dims(&r, dims, OutputSize::Fit, false);
        assert_eq!([out[0], out[2]], [full[0], full[2]]);
        assert!(out[1] + 1 >= full[1] && out[1] <= full[1], "{out:?} vs {full:?}");
        assert!(track.is_static());
        assert_eq!(track.pos, [0.0, 0.0]);
        assert!(window_contains_content(&r, dims, out, track));
        let r = rotation_matrix([30.0, 40.0, 60.0], XYZ);
        let (out, track) = tracked_output(&r, [7, 9, 11], OutputSize::Fit, false);
        assert!(window_contains_content(&r, [7, 9, 11], out, track));
    }

    #[test]
    fn tracked_short_volume_follows_axis() {
        let r = rotation_matrix([10.0, -5.0, 0.0], XYZ);
        let dims = [160, 90, 60];
        let full = output_dims(&r, dims, OutputSize::Fit, true);
        let (out, track) = tracked_output(&r, dims, OutputSize::Fit, true);
        assert!(out[1] < full[1], "{out:?} vs {full:?}");
        assert!(out[1] <= 94, "{out:?}");
        assert!((track.vel[1] - r[1][2] / r[2][2]).abs() < 1e-9, "{track:?}");
        assert!(window_contains_content(&r, dims, out, track));
    }

    #[test]
    fn tracked_respects_output_spec() {
        let r = rotation_matrix([10.0, 0.0, 0.0], XYZ);
        let dims = [4, 6, 60];
        assert_eq!(tracked_output(&r, dims, OutputSize::Crop, false).0, dims);
        let c = parse_output_size("0x20x0").unwrap();
        let (out, _) = tracked_output(&r, dims, c, false);
        assert_eq!(out[1], 20);
        assert_eq!(out[0], tracked_output(&r, dims, OutputSize::Fit, false).0[0]);
    }

    #[test]
    fn tracked_plane_scales_offsets() {
        let r = rotation_matrix([10.0, 0.0, 0.0], XYZ);
        let dims = [8, 6, 40];
        let (out, track) = tracked_output(&r, dims, OutputSize::Fit, true);
        let luma = Geometry::plane(&r, dims, out, track, 0, 0);
        let chroma = Geometry::plane(&r, dims, out, track, 1, 1);
        for (x, y, z) in [(0usize, 0usize, 0usize), (1, 1, 7), (2, 1, 30)] {
            let q = luma.map(2.0 * x as f64 + 0.5, 2.0 * y as f64 + 0.5, z as f64);
            let c = chroma.map(x as f64, y as f64, z as f64);
            assert!(approx(c, [q[0] / 2.0, q[1] / 2.0, q[2]]), "{c:?} vs {q:?}");
        }
    }
}
