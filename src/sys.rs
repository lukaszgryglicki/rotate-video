use anyhow::{bail, Result};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// SIGINT/SIGTERM only raise a flag, so the temp dir is still removed on the way out.
pub fn install_signal_handlers() {
    let handler: extern "C" fn(libc::c_int) = on_signal;
    unsafe {
        libc::signal(libc::SIGINT, handler as usize as libc::sighandler_t);
        libc::signal(libc::SIGTERM, handler as usize as libc::sighandler_t);
    }
}

pub fn check_interrupted() -> Result<()> {
    if INTERRUPTED.load(Ordering::SeqCst) {
        bail!("interrupted");
    }
    Ok(())
}

pub fn phys_mem() -> Option<u64> {
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    (pages > 0 && size > 0).then(|| pages as u64 * size as u64)
}

#[allow(clippy::unnecessary_cast)]
pub fn free_space(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut st) };
    (rc == 0).then(|| st.f_bavail as u64 * st.f_frsize as u64)
}
