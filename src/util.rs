use anyhow::{bail, Result};
use std::io::Write;
use std::str::FromStr;
use std::time::{Duration, Instant};

pub fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

pub fn env_or(name: &str, default: &str) -> String {
    env_str(name).unwrap_or_else(|| default.to_string())
}

pub fn env_bool(name: &str, default: bool) -> Result<bool> {
    match env_str(name) {
        None => Ok(default),
        Some(v) => match v.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" | "y" => Ok(true),
            "0" | "false" | "no" | "off" | "n" => Ok(false),
            _ => bail!("{name}: expected boolean, got {v:?}"),
        },
    }
}

pub fn env_parse<T: FromStr>(name: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    match env_str(name) {
        None => Ok(None),
        Some(v) => match v.parse::<T>() {
            Ok(x) => Ok(Some(x)),
            Err(e) => bail!("{name}: cannot parse {v:?}: {e}"),
        },
    }
}

/// Env var holding a shell-like argument list ("" -> default, "none" -> empty).
pub fn env_args(name: &str, default: &[&str]) -> Result<Vec<String>> {
    match env_str(name) {
        None => Ok(default.iter().map(|s| s.to_string()).collect()),
        Some(v) if v.eq_ignore_ascii_case("none") => Ok(vec![]),
        Some(v) => split_args(&v).map_err(|e| anyhow::anyhow!("{name}: {e}")),
    }
}

/// Split a string into arguments the way a POSIX shell would (quotes, backslashes).
pub fn split_args(s: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut have = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                have = true;
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(x) => cur.push(x),
                        None => bail!("unterminated single quote in {s:?}"),
                    }
                }
            }
            '"' => {
                have = true;
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(x @ ('"' | '\\' | '$' | '`')) => cur.push(x),
                            Some(x) => {
                                cur.push('\\');
                                cur.push(x)
                            }
                            None => bail!("trailing backslash in {s:?}"),
                        },
                        Some(x) => cur.push(x),
                        None => bail!("unterminated double quote in {s:?}"),
                    }
                }
            }
            '\\' => match chars.next() {
                Some(x) => {
                    have = true;
                    cur.push(x)
                }
                None => bail!("trailing backslash in {s:?}"),
            },
            c if c.is_whitespace() => {
                if have {
                    out.push(std::mem::take(&mut cur));
                    have = false;
                }
            }
            c => {
                have = true;
                cur.push(c)
            }
        }
    }
    if have {
        out.push(cur);
    }
    Ok(out)
}

pub fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|a| {
            if a.is_empty() {
                "''".to_string()
            } else if a
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_./:=+,@%".contains(c))
            {
                a.clone()
            } else {
                format!("'{}'", a.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Divide a frame rate given as "num/den" or a decimal by `n`.
pub fn rate_div(rate: &str, n: usize) -> String {
    if n <= 1 {
        return rate.to_string();
    }
    if let Some((num, den)) = rate.split_once('/') {
        if let (Ok(num), Ok(den)) = (num.trim().parse::<u64>(), den.trim().parse::<u64>()) {
            return format!("{num}/{}", den * n as u64);
        }
    }
    match rate.trim().parse::<f64>() {
        Ok(f) => format!("{}", f / n as f64),
        Err(_) => rate.to_string(),
    }
}

pub struct Progress {
    total: Option<u64>,
    start: Instant,
    last: Option<Instant>,
    every: Duration,
    quiet: bool,
    label: &'static str,
    unit: &'static str,
}

impl Progress {
    pub fn new(label: &'static str, total: Option<u64>, quiet: bool) -> Self {
        Progress {
            total,
            start: Instant::now(),
            last: None,
            every: Duration::from_secs(2),
            quiet,
            label,
            unit: "frames",
        }
    }

    pub fn update(&mut self, done: u64) {
        if self.quiet {
            return;
        }
        let now = Instant::now();
        let finished = self.total.is_some_and(|t| done >= t);
        if !finished && self.last.is_some_and(|l| now.duration_since(l) < self.every) {
            return;
        }
        self.last = Some(now);
        let el = now.duration_since(self.start).as_secs_f64();
        let rate = if el > 0.0 { format!(", {:.1} {}/s", done as f64 / el, self.unit) } else { String::new() };
        match self.total {
            Some(total) => {
                let pct = if total > 0 { 100.0 * done as f64 / total as f64 } else { 100.0 };
                let eta = if done > 0 && done < total {
                    format!(", ETA {}", fmt_dur(el * (total - done) as f64 / done as f64))
                } else {
                    String::new()
                };
                eprint!("\r{}: {}/{} ({:.1}%), {} elapsed{}{}   ", self.label, done, total, pct, fmt_dur(el), rate, eta);
            }
            None => eprint!("\r{}: {} {}, {} elapsed{}   ", self.label, done, self.unit, fmt_dur(el), rate),
        }
        let _ = std::io::stderr().flush();
    }

    pub fn total(&self) -> Option<u64> {
        self.total
    }

    pub fn finish(&mut self, done: u64) {
        if self.quiet {
            return;
        }
        self.last = None;
        self.total = Some(done);
        self.update(done);
        eprintln!();
    }
}

pub fn fmt_dur(secs: f64) -> String {
    let s = secs.round() as u64;
    if s >= 3600 {
        format!("{}h{:02}m{:02}s", s / 3600, (s % 3600) / 60, s % 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

pub fn fmt_bytes(b: u64) -> String {
    const U: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.2} {}", U[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split() {
        assert_eq!(split_args("").unwrap(), Vec::<String>::new());
        assert_eq!(split_args("  -vf  scale=960:-2 ").unwrap(), vec!["-vf", "scale=960:-2"]);
        assert_eq!(
            split_args(r#"-vf "scale=960:-2, fps=15" -x 'a b' c\ d"#).unwrap(),
            vec!["-vf", "scale=960:-2, fps=15", "-x", "a b", "c d"]
        );
        assert_eq!(split_args(r#""a\"b""#).unwrap(), vec!["a\"b"]);
        assert_eq!(split_args("''").unwrap(), vec![""]);
        assert!(split_args("'abc").is_err());
    }

    #[test]
    fn rate_division() {
        assert_eq!(rate_div("30000/1001", 1), "30000/1001");
        assert_eq!(rate_div("30000/1001", 3), "30000/3003");
        assert_eq!(rate_div("100", 50), "2");
        assert_eq!(rate_div("29.97", 2), "14.985");
    }

    #[test]
    fn join() {
        let a: Vec<String> = vec!["ffmpeg".into(), "-vf".into(), "a b".into(), "".into()];
        assert_eq!(shell_join(&a), "ffmpeg -vf 'a b' ''");
    }
}
