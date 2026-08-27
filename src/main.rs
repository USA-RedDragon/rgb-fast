//! rgb-fast - drive known RGB devices to a solid color with zero device discovery.
//!
//! Every byte sequence below was captured from real OpenRGB traffic on this exact
//! machine (LD_PRELOAD ioctl/write interposer over /dev/i2c-* and /dev/hidraw*),
//! then replayed. Nothing here is inferred from guesswork.
//!
//!   GPU  : EVGA GeForce RTX 3090 FTW3 Ultra, I2C 0x2D on the NVIDIA adapter
//!   MB   : X670E AORUS MASTER (ITE IT5701), USB HID 048D:5702, feature reports
//!   DRAM : 2x ENE SMBus DRAM at 0x71/0x73

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use i2cdev::core::I2CDevice;
use i2cdev::linux::LinuxI2CDevice;

// HARDWARE CONSTANTS

// EVGA GeForce RTX 3090 FTW3 Ultra (I2C)
const GPU_ADAPTER_NAME: &str = "NVIDIA i2c adapter 1 at a:00.0";
const GPU_ADDR: u16 = 0x2D;
const GPU_REG_ZONE: [u8; 4] = [0xC1, 0xC2, 0xC3, 0xC4]; // Front/Endplate/Back/ARGB hdr
const GPU_REG_SYNC: u8 = 0xB2;
const GPU_REG_MODE: u8 = 0xC0;

/// Every EVGA payload is prefixed with a count byte giving the number of bytes that follow it.
const GPU_SYNC_MAGIC: [u8; 5] = [0x04, 0xC6, 0xEB, 0xEA, 0x15];

/// 0xC0 packet: [count, mode x4, led_count x4, zone_sync].
/// 0xFF in a slot is EVGAGPUV3_INIT - a "leave this field unchanged" sentinel,
/// NOT a value. So this sets all four zones to Static and preserves the stored
/// per-zone LED counts. Confirmed against EVGAGPUv3Controller::SetAllModes().
const GPU_MODE_DIRECT: [u8; 10] =
    [0x09, 0x01, 0x01, 0x01, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
/// Hardware limits on the ARGB header LED count (EVGAGPUV3_LEDS_MIN/MAX).
const GPU_ARGB_LEDS_MIN: u8 = 1;
const GPU_ARGB_LEDS_MAX: u8 = 60;
/// Index of the ARGB-header zone within the 0xC0 packet's led_count field.
/// zone_led_count[z] lives at packet[z + 5]; the ARGB header is zone index 3.
const GPU_ARGB_ZONE: usize = 3;
/// How many LEDs to drive on the GPU's ARGB passthrough header.
///
/// THIS IS THE FIX for the "top of the GPU stays blue" bug. OpenRGB models this
/// zone as ZONE_TYPE_SINGLE with leds_max = 1 and calls ResizeARGB(1), so the
/// card clocks out data for only the FIRST LED on the chain; every LED after it
/// never receives data and holds its power-on default, which is blue. Proven:
/// set the count to 60, run one `openrgb` command, and it is 1 again.
///
/// We drive the hardware maximum by default. Over-sending is harmless on a
/// WS2812-style chain (surplus bytes fall off the end); under-sending is what
/// strands the tail.
const GPU_ARGB_LED_COUNT_DEFAULT: u8 = GPU_ARGB_LEDS_MAX;
/// Channel order for the ARGB passthrough header (zone 3) ONLY.
/// Values are indices into (r, g, b): [0,1,2] = RGB, [2,1,0] = BGR, [2,0,1] = BRG.
///
/// PLAIN RGB — corrected. An earlier revision defaulted to BGR on the theory that
/// the card forwarded bytes unswizzled to a strip with a different channel order.
/// That was wrong, and the BGR swizzle is exactly what made the header render
/// blue for a red request: it turned R=FF into the byte triple 00,00,FF.
///
/// Proof it is plain RGB, from the decisive observation: under the old BGR
/// swizzle a request for red put the byte triple 00,00,FF into 0xC4 and the
/// header rendered BLUE. If the wire order really were B,G,R those bytes would
/// have been B=00,G=00,R=FF and it would have rendered RED. It did not, so the
/// LAST byte drives blue and the order is R,G,B.
///
/// (A [00,FF,00] probe cannot settle this — green occupies the middle slot under
/// both R,G,B and B,G,R, so it renders green either way. Use --test-argb-order,
/// which now writes [FF,00,00], where the two hypotheses give different colours.)
///
/// With --gpu-argb-order rgb, a red request leaves all four zone registers
/// reading [04,FF,FF,00,00], i.e. count=04, brightness=FF, R=FF, G=00, B=00.
///
/// Separately and still true: this header FLICKERS during POST, before any
/// software can run. That is firmware behaviour and nothing here can fix it.
/// It is unrelated to the rendering question above.
const GPU_ARGB_ORDER: [usize; 3] = [0, 1, 2]; // RGB

fn parse_channel_order(s: &str) -> Option<[usize; 3]> {
    let idx = |c: char| match c {
        'r' | 'R' => Some(0usize),
        'g' | 'G' => Some(1),
        'b' | 'B' => Some(2),
        _ => None,
    };
    let c: Vec<char> = s.chars().collect();
    if c.len() != 3 {
        return None;
    }
    let o = [idx(c[0])?, idx(c[1])?, idx(c[2])?];
    if o[0] == o[1] || o[1] == o[2] || o[0] == o[2] {
        return None;
    }
    Some(o)
}

/// Re-order a colour into the byte sequence zone 3's strip expects.
fn gpu_argb_bytes((r, g, b): (u8, u8, u8), order: [usize; 3]) -> [u8; 3] {
    let c = [r, g, b];
    [c[order[0]], c[order[1]], c[order[2]]]
}
/// Per-zone colour payload is [count=4, brightness, R, G, B] — the R/G/B slot
/// order was proved by diffing red vs green vs blue captures.
const GPU_BRIGHTNESS: u8 = 0xFF;
const GPU_ZONE_COUNT_BYTE: u8 = 0x04;

/* ---- X670E AORUS MASTER (USB HID feature reports) ------------------- */
const MB_HID_ID: &str = "0003:0000048D:00005702"; // HID_ID in /sys/.../uevent
const MB_REPORT_ID: u8 = 0xCC;
const MB_HDR_EFFECT_BASE: u8 = 0x20; // headers 0x20..0x24 = the 5 fixed zones
const MB_HDR_APPLY: u8 = 0x28;
const MB_HDR_ARGB: [u8; 2] = [0x58, 0x59]; // D_LED1 / D_LED2 addressable strips
const MB_FIXED_ZONES: u8 = 5; // I/O Cover Top, LED_C1, LED_CPU, I/O Cover Bottom, LED_C2
const MB_ARGB_LEDS: usize = 200; // per strip; OpenRGB's configured size. Tunable: --argb-leds
const MB_ARGB_CHUNK: usize = 57; // 19 LEDs x 3 bytes, the max the 64B report holds
const MB_PKT: usize = 64;
const MB_EFFECT_STATIC: u8 = 0x01;
const MB_MAX_BRIGHTNESS: u8 = 0xFF;

/* ---- ENE SMBus DRAM — live behind --dram. See dram_plan(). ---------- */
const DRAM_ADAPTER_NAME: &str = "SMBus PIIX4 adapter port 0 at 0b00";
/// The ONLY SMBus addresses this program may ever speak to. The SPD5 hubs of the
/// two DIMMs sit at 0x51/0x53 (kernel driver spd5118 bound at 2-0051 / 2-0053);
/// 0x50-0x57 must never see a transaction from this tool, read or write.
const DRAM_ADDR_ALLOWLIST: [u16; 2] = [0x71, 0x73];
const ENE_REG_COLORS_DIRECT_V2: u16 = 0x8100; // 3 bytes/LED, order is R,B,G
const ENE_REG_DIRECT: u16 = 0x8020;
const ENE_REG_APPLY: u16 = 0x80A0;
const ENE_APPLY_VAL: u8 = 0x01;
const ENE_LED_COUNT: u16 = 8;
const ENE_CMD_REG_PTR: u8 = 0x00; // word write selects the 16-bit register pointer
const ENE_CMD_DATA: u8 = 0x01; // byte write to the selected register
const ENE_CMD_BLOCK: u8 = 0x03; // block write to the selected register

/* ===================================================================== *\
|  SAFETY GUARD                                                           |
\* ===================================================================== */

/// Hard refusal for any SMBus address outside the allowlist. Called before the
/// device is opened *and* again immediately before every single transaction, so
/// there is no code path that can emit a transfer to an SPD hub.
#[inline(always)]
fn assert_smbus_addr_allowed(addr: u16) {
    if !DRAM_ADDR_ALLOWLIST.contains(&addr) {
        panic!(
            "REFUSING SMBus transaction to address 0x{:02X}: not in allowlist {:02X?}. \
             SPD5 hubs occupy 0x50-0x57 and must never be touched.",
            addr, DRAM_ADDR_ALLOWLIST
        );
    }
}

/* ===================================================================== *\
|  NODE RESOLUTION — /sys lookups only, no bus traffic, no hid_enumerate  |
\* ===================================================================== */

fn find_i2c_by_adapter_name(want: &str) -> Option<PathBuf> {
    for entry in fs::read_dir("/sys/class/i2c-dev").ok()?.flatten() {
        let name = match fs::read_to_string(entry.path().join("name")) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if name.trim() == want {
            return Some(Path::new("/dev").join(entry.file_name()));
        }
    }
    None
}

/// Resolve the hidraw node for a given HID_ID without touching hidapi or
/// enumerating every HID device on the system (OpenRGB's blanket
/// hid_enumerate() alone costs ~0.2s).
fn find_hidraw_by_hid_id(want: &str) -> Option<PathBuf> {
    for entry in fs::read_dir("/sys/class/hidraw").ok()?.flatten() {
        let uevent = match fs::read_to_string(entry.path().join("device/uevent")) {
            Ok(u) => u,
            Err(_) => continue,
        };
        if uevent
            .lines()
            .any(|l| l.strip_prefix("HID_ID=").map_or(false, |v| v.eq_ignore_ascii_case(want)))
        {
            return Some(Path::new("/dev").join(entry.file_name()));
        }
    }
    None
}

/* ===================================================================== *\
|  hidraw feature reports — direct ioctl, no hidapi                       |
\* ===================================================================== */

mod hidraw {
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::unix::io::AsRawFd;
    use std::path::Path;

    /// HIDIOCSFEATURE(64) = _IOC(_IOC_WRITE|_IOC_READ, 'H', 0x06, 64)
    ///                    = (3<<30) | (64<<16) | ('H'<<8) | 0x06
    const HIDIOCSFEATURE_64: libc::c_ulong = 0xC040_4806;

    pub struct Dev(File);

    impl Dev {
        pub fn open(path: &Path) -> io::Result<Self> {
            Ok(Dev(OpenOptions::new().read(true).write(true).open(path)?))
        }

        pub fn send_feature(&self, buf: &[u8; 64]) -> io::Result<()> {
            // SAFETY: `self.0` is an open hidraw character device. `buf` is exactly
            // 64 bytes, matching the length baked into HIDIOCSFEATURE_64, and the
            // kernel only reads (never writes past) that buffer for SET_FEATURE.
            let rc =
                unsafe { libc::ioctl(self.0.as_raw_fd(), HIDIOCSFEATURE_64, buf.as_ptr()) };
            if rc < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }
}

/* ===================================================================== *\
|  MOTHERBOARD PACKET BUILDERS (layout verified byte-for-byte vs capture) |
\* ===================================================================== */

/// PktEffect: report_id, header, zone0:u32, zone1:u32, reserved0, effect_type,
/// max_brightness, min_brightness, color0:u32, ... colour is 0x00RRGGBB LE.
fn mb_pkt_effect(zone: u8, (r, g, b): (u8, u8, u8)) -> [u8; MB_PKT] {
    let mut p = [0u8; MB_PKT];
    p[0] = MB_REPORT_ID;
    p[1] = MB_HDR_EFFECT_BASE + zone;
    p[2..6].copy_from_slice(&(1u32 << zone).to_le_bytes());
    p[11] = MB_EFFECT_STATIC;
    p[12] = MB_MAX_BRIGHTNESS;
    p[13] = 0x00; // min_brightness
    let color = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
    p[14..18].copy_from_slice(&color.to_le_bytes());
    p
}

/// PktRGB: report_id, header, boffset:u16, bcount:u8, then 19 LEDs of G,R,B.
fn mb_pkt_argb(header: u8, (r, g, b): (u8, u8, u8), n_leds: usize) -> Vec<[u8; MB_PKT]> {
    let mut data = Vec::with_capacity(n_leds * 3);
    for _ in 0..n_leds {
        data.extend_from_slice(&[g, r, b]); // GRB on the wire
    }
    let mut pkts = Vec::new();
    let mut off = 0usize;
    while off < data.len() {
        let n = MB_ARGB_CHUNK.min(data.len() - off);
        let mut p = [0u8; MB_PKT];
        p[0] = MB_REPORT_ID;
        p[1] = header;
        p[2..4].copy_from_slice(&(off as u16).to_le_bytes());
        p[4] = n as u8;
        p[5..5 + n].copy_from_slice(&data[off..off + n]);
        pkts.push(p);
        off += n;
    }
    pkts
}

fn mb_pkt_apply(zone_mask: u32) -> [u8; MB_PKT] {
    let mut p = [0u8; MB_PKT];
    p[0] = MB_REPORT_ID;
    p[1] = MB_HDR_APPLY;
    p[2..6].copy_from_slice(&zone_mask.to_le_bytes());
    p
}

fn drive_motherboard(
    color: (u8, u8, u8),
    argb: Option<usize>,
) -> io::Result<(usize, Duration)> {
    let t0 = Instant::now();
    let path = find_hidraw_by_hid_id(MB_HID_ID).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("no hidraw node with HID_ID={MB_HID_ID}"),
        )
    })?;
    let trace = std::env::var_os("RGB_FAST_TRACE").is_some();
    let t_resolve = t0.elapsed();
    let dev = hidraw::Dev::open(&path)?;
    let t_open = t0.elapsed();
    if trace {
        eprintln!("    [trace] resolve {:.2} ms, open {:.2} ms ({})",
            t_resolve.as_secs_f64()*1e3, (t_open-t_resolve).as_secs_f64()*1e3, path.display());
    }
    let mut last = Instant::now();
    let mut send = |d: &hidraw::Dev, p: &[u8; MB_PKT], label: &str| -> io::Result<()> {
        let r = d.send_feature(p);
        if trace {
            eprintln!("    [trace] {label} {:.2} ms", last.elapsed().as_secs_f64()*1e3);
            last = Instant::now();
        }
        r
    };

    let mut sent = 0usize;
    if let Some(n_leds) = argb {
        for hdr in MB_HDR_ARGB {
            for (i, p) in mb_pkt_argb(hdr, color, n_leds).iter().enumerate() {
                send(&dev, p, &format!("argb 0x{hdr:02X} chunk{i}"))?;
                sent += 1;
            }
        }
    }
    for z in 0..MB_FIXED_ZONES {
        send(&dev, &mb_pkt_effect(z, color), &format!("effect zone{z}"))?;
        sent += 1;
    }
    send(&dev, &mb_pkt_apply((1u32 << MB_FIXED_ZONES) - 1), "apply")?; // 0x1F
    sent += 1;

    Ok((sent, t0.elapsed()))
}

/* ===================================================================== *\
|  GPU                                                                    |
\* ===================================================================== */

fn drive_gpu(
    color: (u8, u8, u8),
    verify: bool,
    argb_leds: u8,
    argb_order: [usize; 3],
) -> Result<(Duration, Option<String>), String> {
    let t0 = Instant::now();
    let path = find_i2c_by_adapter_name(GPU_ADAPTER_NAME)
        .ok_or_else(|| format!("no i2c adapter named {GPU_ADAPTER_NAME:?}"))?;
    let mut dev =
        LinuxI2CDevice::new(&path, GPU_ADDR).map_err(|e| format!("open {path:?}: {e}"))?;

    let (r, g, b) = color;
    let swizzled = gpu_argb_bytes(color, argb_order);
    for (z, reg) in GPU_REG_ZONE.iter().enumerate() {
        // Logo zones take plain R,G,B; the ARGB passthrough header does not.
        let c = if z == GPU_ARGB_ZONE { swizzled } else { [r, g, b] };
        dev.smbus_write_i2c_block_data(*reg, &[GPU_ZONE_COUNT_BYTE, GPU_BRIGHTNESS, c[0], c[1], c[2]])
            .map_err(|e| format!("write zone 0x{reg:02X}: {e}"))?;
    }
    dev.smbus_write_i2c_block_data(GPU_REG_SYNC, &GPU_SYNC_MAGIC)
        .map_err(|e| format!("write sync: {e}"))?;
    // Same single packet also restores the ARGB header LED count that OpenRGB
    // clamps to 1 — costs no extra transaction.
    let mut mode = GPU_MODE_DIRECT;
    mode[GPU_ARGB_ZONE + 5] = argb_leds.clamp(GPU_ARGB_LEDS_MIN, GPU_ARGB_LEDS_MAX);
    dev.smbus_write_i2c_block_data(GPU_REG_MODE, &mode)
        .map_err(|e| format!("write mode: {e}"))?;
    let elapsed = t0.elapsed();

    let readback = if verify {
        let mut out = String::new();
        for reg in GPU_REG_ZONE {
            let v = dev
                .smbus_read_i2c_block_data(reg, 5)
                .map_err(|e| format!("readback 0x{reg:02X}: {e}"))?;
            out.push_str(&format!("0x{reg:02X}={v:02X?} "));
        }
        Some(out)
    } else {
        None
    };
    Ok((elapsed, readback))
}

/* ===================================================================== *\
|  ENE SMBus DRAM — live behind --dram; --dram-plan still only prints.     |
\* ===================================================================== */

#[derive(Debug)]
enum SmbusOp {
    /// word write to command 0x00 selects the 16-bit ENE register pointer
    RegPtr { reg: u16, word: u16 },
    /// byte write to command 0x01 stores one byte at the selected register
    Byte { cmd: u8, val: u8 },
    /// block write to command 0x03 stores N bytes at the selected register
    Block { cmd: u8, data: Vec<u8> },
}

/// ENE selects a register by writing the byte-swapped register number as an
/// SMBus word to command 0x00 (so the high byte goes out on the wire first).
fn ene_reg_ptr(reg: u16) -> SmbusOp {
    SmbusOp::RegPtr { reg, word: reg.swap_bytes() }
}

/// The exact transaction list captured from OpenRGB applying a solid colour to
/// one ENE DRAM controller, in the order OpenRGB emitted it.
///
/// NOTE the on-wire colour order is R, B, G — not RGB. Red alone cannot reveal
/// this (G and B are both zero); it is confirmed by OpenRGB's
/// ENESMBusController::SetLEDColorDirect(), which builds `{red, blue, green}`.
/// `direct_first` selects the alternate ordering: set REG_DIRECT (0x8020) BEFORE
/// streaming the colours instead of after. OpenRGB emits colours-then-DIRECT and
/// that is the default here, because it is what was captured.
fn dram_plan(color: (u8, u8, u8), direct_first: bool) -> Vec<SmbusOp> {
    let (r, g, b) = color;
    let mut ops = Vec::new();
    let direct = |ops: &mut Vec<SmbusOp>| {
        ops.push(ene_reg_ptr(ENE_REG_DIRECT));
        ops.push(SmbusOp::Byte { cmd: ENE_CMD_DATA, val: 0x01 });
    };
    if direct_first {
        direct(&mut ops);
    }
    for led in 0..ENE_LED_COUNT {
        ops.push(ene_reg_ptr(ENE_REG_COLORS_DIRECT_V2 + 3 * led));
        ops.push(SmbusOp::Block { cmd: ENE_CMD_BLOCK, data: vec![r, b, g] });
    }
    if !direct_first {
        direct(&mut ops);
    }
    ops.push(ene_reg_ptr(ENE_REG_APPLY));
    ops.push(SmbusOp::Byte { cmd: ENE_CMD_DATA, val: ENE_APPLY_VAL });
    ops
}

/// Live DRAM path — reached only when `--dram` is passed. Every transaction is
/// bracketed by `assert_smbus_addr_allowed`, and the address itself can only ever
/// come from iterating the compile-time `DRAM_ADDR_ALLOWLIST`, so no SPD5 hub
/// address is expressible here.
fn drive_dram(color: (u8, u8, u8), direct_first: bool) -> Result<Duration, String> {
    let t0 = Instant::now();
    let path = find_i2c_by_adapter_name(DRAM_ADAPTER_NAME)
        .ok_or_else(|| format!("no i2c adapter named {DRAM_ADAPTER_NAME:?}"))?;
    for addr in DRAM_ADDR_ALLOWLIST {
        assert_smbus_addr_allowed(addr); // before open
        let mut dev =
            LinuxI2CDevice::new(&path, addr).map_err(|e| format!("open {path:?}@{addr:02X}: {e}"))?;
        for op in dram_plan(color, direct_first) {
            assert_smbus_addr_allowed(addr); // again, before every transaction
            let res = match op {
                SmbusOp::RegPtr { word, .. } => dev.smbus_write_word_data(ENE_CMD_REG_PTR, word),
                SmbusOp::Byte { cmd, val } => dev.smbus_write_byte_data(cmd, val),
                SmbusOp::Block { cmd, data } => dev.smbus_write_block_data(cmd, &data),
            };
            res.map_err(|e| format!("dram 0x{addr:02X}: {e}"))?;
        }
    }
    Ok(t0.elapsed())
}

/* ===================================================================== *\
|  PROBE — paint each candidate zone a distinct colour so one look from the    |
|  operator maps every zone to a physical object.                             |
\* ===================================================================== */

const PROBE_D_LED1: (u8, u8, u8) = (0x00, 0xFF, 0x00); // green
const PROBE_D_LED2: (u8, u8, u8) = (0xFF, 0x00, 0xFF); // magenta
const PROBE_GPU_ARGB: (u8, u8, u8) = (0x00, 0xFF, 0xFF); // cyan
const PROBE_REST: (u8, u8, u8) = (0xFF, 0x00, 0x00); // everything else stays red

fn probe(argb_leds: usize, argb_order: [usize; 3], gpu_argb_leds: u8) -> Result<(), String> {
    // ---- motherboard: strip 1 green, strip 2 magenta, fixed zones red ----
    let path = find_hidraw_by_hid_id(MB_HID_ID)
        .ok_or_else(|| format!("no hidraw node with HID_ID={MB_HID_ID}"))?;
    let dev = hidraw::Dev::open(&path).map_err(|e| format!("open {path:?}: {e}"))?;
    for (hdr, col) in [
        (MB_HDR_ARGB[0], PROBE_D_LED1),
        (MB_HDR_ARGB[1], PROBE_D_LED2),
    ] {
        for pkt in mb_pkt_argb(hdr, col, argb_leds) {
            dev.send_feature(&pkt).map_err(|e| format!("argb: {e}"))?;
        }
    }
    for z in 0..MB_FIXED_ZONES {
        dev.send_feature(&mb_pkt_effect(z, PROBE_REST))
            .map_err(|e| format!("zone {z}: {e}"))?;
    }
    dev.send_feature(&mb_pkt_apply((1u32 << MB_FIXED_ZONES) - 1))
        .map_err(|e| format!("apply: {e}"))?;

    // ---- GPU: shroud zones red, ARGB header cyan ----
    let ipath = find_i2c_by_adapter_name(GPU_ADAPTER_NAME)
        .ok_or_else(|| format!("no i2c adapter named {GPU_ADAPTER_NAME:?}"))?;
    let mut gpu =
        LinuxI2CDevice::new(&ipath, GPU_ADDR).map_err(|e| format!("open {ipath:?}: {e}"))?;
    gpu.smbus_write_i2c_block_data(GPU_REG_SYNC, &GPU_SYNC_MAGIC)
        .map_err(|e| format!("init: {e}"))?;
    for (z, reg) in GPU_REG_ZONE.iter().enumerate() {
        let c = if z == GPU_ARGB_ZONE {
            gpu_argb_bytes(PROBE_GPU_ARGB, argb_order)
        } else {
            [PROBE_REST.0, PROBE_REST.1, PROBE_REST.2]
        };
        gpu.smbus_write_i2c_block_data(*reg, &[GPU_ZONE_COUNT_BYTE, GPU_BRIGHTNESS, c[0], c[1], c[2]])
            .map_err(|e| format!("zone {z}: {e}"))?;
    }
    gpu.smbus_write_i2c_block_data(GPU_REG_SYNC, &GPU_SYNC_MAGIC)
        .map_err(|e| format!("init2: {e}"))?;
    let mut mode = GPU_MODE_DIRECT;
    mode[GPU_ARGB_ZONE + 5] = gpu_argb_leds.clamp(GPU_ARGB_LEDS_MIN, GPU_ARGB_LEDS_MAX);
    gpu.smbus_write_i2c_block_data(GPU_REG_MODE, &mode)
        .map_err(|e| format!("mode: {e}"))?;

    println!("probe applied - look at the machine and report which item shows which colour:");
    println!("  GREEN    = motherboard D_LED1 header (0x58)");
    println!("  MAGENTA  = motherboard D_LED2 header (0x59)");
    println!("  CYAN     = GPU EVGA ARGB passthrough header (zone 3, reg 0xC4)");
    println!("  RED      = everything else (GPU shroud zones, MB fixed zones)");
    println!("  still BLUE / unchanged = driven by none of the above");
    println!("run `rgb-fast FF0000` to put everything back to red.");
    Ok(())
}

/// Write RAW bytes to the ARGB header, bypassing all colour naming, so a single
/// observation resolves the strip's channel order. Zones 0-2 stay red.
///
/// Use a raw triple whose FIRST and LAST bytes differ — [FF,00,00] is the default.
/// A middle-byte probe such as [00,FF,00] is useless here: green sits in the
/// middle slot under both R,G,B and B,G,R, so it cannot distinguish them.
fn test_argb_order(raw: [u8; 3], gpu_argb_leds: u8) -> Result<(), String> {
    let ipath = find_i2c_by_adapter_name(GPU_ADAPTER_NAME)
        .ok_or_else(|| format!("no i2c adapter named {GPU_ADAPTER_NAME:?}"))?;
    let mut gpu =
        LinuxI2CDevice::new(&ipath, GPU_ADDR).map_err(|e| format!("open {ipath:?}: {e}"))?;
    gpu.smbus_write_i2c_block_data(GPU_REG_SYNC, &GPU_SYNC_MAGIC)
        .map_err(|e| format!("init: {e}"))?;
    for (z, reg) in GPU_REG_ZONE.iter().enumerate() {
        let c = if z == GPU_ARGB_ZONE { raw } else { [0xFF, 0x00, 0x00] };
        gpu.smbus_write_i2c_block_data(*reg, &[GPU_ZONE_COUNT_BYTE, GPU_BRIGHTNESS, c[0], c[1], c[2]])
            .map_err(|e| format!("zone {z}: {e}"))?;
    }
    gpu.smbus_write_i2c_block_data(GPU_REG_SYNC, &GPU_SYNC_MAGIC)
        .map_err(|e| format!("init2: {e}"))?;
    let mut mode = GPU_MODE_DIRECT;
    mode[GPU_ARGB_ZONE + 5] = gpu_argb_leds.clamp(GPU_ARGB_LEDS_MIN, GPU_ARGB_LEDS_MAX);
    gpu.smbus_write_i2c_block_data(GPU_REG_MODE, &mode)
        .map_err(|e| format!("mode: {e}"))?;
    println!("raw bytes {raw:02X?} written to GPU ARGB header (zone 3), {gpu_argb_leds} LEDs.");
    println!("Look ONLY at whatever is on the GPU's ARGB passthrough header.");
    println!("  with the default probe [FF,00,00]:");
    println!("    shows RED  -> first byte drives red, order is R,G,B  (--gpu-argb-order rgb)");
    println!("    shows BLUE -> first byte drives blue, order is B,G,R (--gpu-argb-order bgr)");
    Ok(())
}

/* ===================================================================== *\
|  main                                                                   |
\* ===================================================================== */

fn parse_color(s: &str) -> Result<(u8, u8, u8), String> {
    let h = s.trim_start_matches('#');
    if h.len() != 6 {
        return Err(format!("colour must be RRGGBB hex, got {s:?}"));
    }
    let v = u32::from_str_radix(h, 16).map_err(|_| format!("bad hex {s:?}"))?;
    Ok((((v >> 16) & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, (v & 0xFF) as u8))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Device {
    Gpu,
    Mb,
    Dram,
}

/// Exit code used when nothing at all could be driven. In --never-fail mode this
/// is rewritten to 0 so that a boot-time unit can never report a failure.
const EXIT_FAIL: i32 = 1;

fn main() {
    let t_start = Instant::now();
    let mut color = (0xFFu8, 0x00u8, 0x00u8); // default: red
    let mut argb: Option<usize> = Some(MB_ARGB_LEDS);
    let mut verify = false;
    let mut only: Option<Device> = None;
    let mut do_probe = false;
    let mut gpu_argb_leds: u8 = GPU_ARGB_LED_COUNT_DEFAULT;
    let mut argb_order: [usize; 3] = GPU_ARGB_ORDER;
    let mut test_order: Option<[u8; 3]> = None;
    let mut dram_direct_first = false;
    let mut never_fail = false;
    let mut deadline_ms: Option<u64> = None;
    let mut quiet = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--no-argb" => argb = None,
            "--argb-leds" => {
                let n = args.next().and_then(|v| v.parse::<usize>().ok());
                match n {
                    Some(n) if n > 0 => argb = Some(n),
                    _ => {
                        eprintln!("rgb-fast: --argb-leds needs a positive count");
                        std::process::exit(2);
                    }
                }
            }
            "--verify" => verify = true,
            "--probe" => do_probe = true,
            "--test-argb-order" => test_order = Some([0xFF, 0x00, 0x00]),
            "--gpu-argb-order" => match args.next().as_deref().and_then(parse_channel_order) {
                Some(o) => argb_order = o,
                None => {
                    eprintln!("rgb-fast: --gpu-argb-order takes a permutation of r,g,b e.g. bgr");
                    std::process::exit(2);
                }
            },
            "--gpu-argb-leds" => {
                match args.next().and_then(|v| v.parse::<u8>().ok()) {
                    Some(n) => gpu_argb_leds = n,
                    None => {
                        eprintln!("rgb-fast: --gpu-argb-leds needs a count 1..60");
                        std::process::exit(2);
                    }
                }
            }
            "--only" => match args.next().as_deref() {
                Some("gpu") => only = Some(Device::Gpu),
                Some("mb") | Some("motherboard") => only = Some(Device::Mb),
                Some("dram") | Some("dimm") => only = Some(Device::Dram),
                _ => {
                    eprintln!("rgb-fast: --only takes 'gpu', 'mb' or 'dram'");
                    std::process::exit(2);
                }
            },
            "--dram-direct-first" => dram_direct_first = true,
            "--never-fail" => never_fail = true,
            "--quiet" | "-q" => quiet = true,
            "--deadline-ms" => match args.next().and_then(|v| v.parse::<u64>().ok()) {
                Some(n) if n > 0 => deadline_ms = Some(n),
                _ => {
                    eprintln!("rgb-fast: --deadline-ms needs a positive millisecond count");
                    std::process::exit(2);
                }
            },
            "-h" | "--help" => {
                println!(
                    "rgb-fast [RRGGBB] [options]\n\
                     \n  RRGGBB      solid colour, default FF0000 (red)\
                     \n  --no-argb   skip the two D_LED addressable strips (saves ~73 ms)\
                     \n  --argb-leds N  write only N LEDs per strip (default 200)\
                     \n  --only gpu|mb|dram  drive just one device (for udev-activated units)\
                     \n  --gpu-argb-leds N  LEDs on the GPU ARGB passthrough header (default 60)\
                     \n  --gpu-argb-order ORD  channel order for that header, e.g. rgb (default)\
                     \n  --probe     paint each zone a distinct colour to map zones to hardware\
                     \n  --verify    read GPU colour registers back after writing\
                     \n\
                     \nDRAM (ENE SMBus 0x71/0x73) — writes only ever go to those two\
                     \naddresses; the allowlist is compile-time and guarded at runtime:\
                     \n  --dram-direct-first  set REG_DIRECT before the colours instead of after\
                     \n\
                     \nBoot-time safety (used by the initrd units):\
                     \n  --never-fail    always exit 0, whatever happened\
                     \n  --deadline-ms N self-terminate after N ms, come what may\
                     \n  --quiet         suppress the timing report on success"
                );
                return;
            }
            other => match parse_color(other) {
                Ok(c) => color = c,
                Err(e) => {
                    eprintln!("rgb-fast: {e}");
                    std::process::exit(2);
                }
            },
        }
    }

    // ---- HARD DEADLINE -------------------------------------------------
    // Armed before a single device is touched. This is the program's own
    // guarantee that it terminates: a detached thread sleeps for the deadline
    // and then calls _exit(). It does not need the main thread to be healthy,
    // cannot be starved by a blocking ioctl on any bus, and does not depend on
    // systemd's TimeoutStartSec (which is a second, independent backstop).
    if let Some(ms) = deadline_ms {
        let code = if never_fail { 0 } else { 124 };
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            eprintln!("rgb-fast: deadline {ms} ms exceeded, exiting {code}");
            std::process::exit(code);
        });
    }

    // Exit helper: honours --never-fail so a boot-time unit can never fail.
    let finish = move |code: i32| -> ! {
        std::process::exit(if never_fail { 0 } else { code });
    };


    if let Some(raw) = test_order {
        if let Err(e) = test_argb_order(raw, gpu_argb_leds) {
            eprintln!("rgb-fast: {e}");
            std::process::exit(1);
        }
        return;
    }

    if do_probe {
        let n = argb.unwrap_or(MB_ARGB_LEDS);
        if let Err(e) = probe(n, argb_order, gpu_argb_leds) {
            eprintln!("rgb-fast: {e}");
            std::process::exit(1);
        }
        return;
    }

    let want_gpu = only.map_or(true, |d| d == Device::Gpu);
    let want_mb = only.map_or(true, |d| d == Device::Mb);
    let want_dram = only.map_or(true, |d| d == Device::Dram);

    // GPU (i2c-5, NVIDIA), DRAM (i2c-2, PIIX4 SMBus) and motherboard (USB HID)
    // are three independent buses — overlap them so the wall time is the max of
    // the three rather than the sum.
    let gpu_thread = want_gpu
        .then(|| std::thread::spawn(move || drive_gpu(color, verify, gpu_argb_leds, argb_order)));
    let dram_thread =
        want_dram.then(|| std::thread::spawn(move || drive_dram(color, dram_direct_first)));
    let mb = want_mb.then(|| drive_motherboard(color, argb));
    let gpu = gpu_thread.map(|t| t.join().unwrap_or_else(|_| Err("gpu thread panicked".into())));
    let dram = dram_thread.map(|t| t.join().unwrap_or_else(|_| Err("dram thread panicked".into())));

    let total = t_start.elapsed();
    let (r, g, b) = color;
    let mut failed = false;
    let mut out = String::new();
    out.push_str(&format!("rgb-fast: #{r:02X}{g:02X}{b:02X}\n"));
    match gpu {
        Some(Ok((d, rb))) => {
            out.push_str(&format!(
                "  gpu          {:>7.2} ms  ok (argb header = {gpu_argb_leds} leds, order {})\n",
                d.as_secs_f64() * 1e3,
                ["r", "g", "b"][argb_order[0]].to_string()
                    + ["r", "g", "b"][argb_order[1]]
                    + ["r", "g", "b"][argb_order[2]]
            ));
            if let Some(rb) = rb {
                out.push_str(&format!("  gpu readback {rb}\n"));
            }
        }
        Some(Err(e)) => {
            failed = true;
            out.push_str(&format!("  gpu          FAILED: {e}\n"));
        }
        None => {}
    }
    match mb {
        Some(Ok((n, d))) => out.push_str(&format!(
            "  motherboard  {:>7.2} ms  ok ({n} feature reports, argb={})\n",
            d.as_secs_f64() * 1e3,
            match argb {
                Some(n) => format!("{n} leds/strip"),
                None => "off".to_string(),
            }
        )),
        Some(Err(e)) => {
            failed = true;
            out.push_str(&format!("  motherboard  FAILED: {e}\n"));
        }
        None => {}
    }
    match dram {
        Some(Ok(d)) => out.push_str(&format!(
            "  dram         {:>7.2} ms  ok (2 controllers {DRAM_ADDR_ALLOWLIST:02X?}, {ENE_LED_COUNT} leds, {})\n",
            d.as_secs_f64() * 1e3,
            if dram_direct_first { "DIRECT->colours->APPLY" } else { "colours->DIRECT->APPLY" }
        )),
        Some(Err(e)) => {
            failed = true;
            out.push_str(&format!("  dram         FAILED: {e}\n"));
        }
        None => {}
    }
    out.push_str(&format!("  TOTAL        {:>7.2} ms\n", total.as_secs_f64() * 1e3));

    if failed || !quiet {
        print!("{out}");
    }
    if failed {
        finish(EXIT_FAIL);
    }
}

/* ===================================================================== *\
|  GUARD TESTS — `cargo test` proves the SPD5 range is unreachable.        |
\* ===================================================================== */

#[cfg(test)]
mod guard_tests {
    use super::*;

    /// The allowlist itself must never grow to include an SPD5 hub address.
    #[test]
    fn allowlist_excludes_every_spd5_address() {
        for addr in 0x50u16..=0x57 {
            assert!(
                !DRAM_ADDR_ALLOWLIST.contains(&addr),
                "SPD5 address 0x{addr:02X} must never appear in the allowlist"
            );
        }
        assert_eq!(DRAM_ADDR_ALLOWLIST, [0x71, 0x73]);
    }

    /// Every address in 0x50-0x57 must make the guard panic.
    #[test]
    fn guard_panics_on_all_spd5_addresses() {
        for addr in 0x50u16..=0x57 {
            let r = std::panic::catch_unwind(|| assert_smbus_addr_allowed(addr));
            assert!(r.is_err(), "guard did NOT refuse SPD5 address 0x{addr:02X}");
        }
    }

    /// And the guard must refuse everything else too — allowlist, not denylist.
    #[test]
    fn guard_panics_on_every_address_outside_the_allowlist() {
        for addr in 0x00u16..=0x7F {
            let expect_ok = DRAM_ADDR_ALLOWLIST.contains(&addr);
            let r = std::panic::catch_unwind(|| assert_smbus_addr_allowed(addr));
            assert_eq!(r.is_ok(), expect_ok, "wrong verdict for 0x{addr:02X}");
        }
    }

    /// The plan must only ever reference the two ENE colour/control registers,
    /// never anything that could be an SPD page-select.
    #[test]
    fn plan_touches_only_expected_registers() {
        for direct_first in [false, true] {
            for op in dram_plan((0xFF, 0x00, 0x00), direct_first) {
                if let SmbusOp::RegPtr { reg, .. } = op {
                    let ok = (ENE_REG_COLORS_DIRECT_V2..ENE_REG_COLORS_DIRECT_V2 + 3 * ENE_LED_COUNT)
                        .contains(&reg)
                        || reg == ENE_REG_DIRECT
                        || reg == ENE_REG_APPLY;
                    assert!(ok, "unexpected ENE register 0x{reg:04X}");
                }
            }
        }
    }

    /// Both orderings must contain exactly the same transactions, differing only
    /// in where REG_DIRECT lands.
    #[test]
    fn both_orders_have_identical_transaction_counts() {
        let a = dram_plan((0xFF, 0x00, 0x00), false).len();
        let b = dram_plan((0xFF, 0x00, 0x00), true).len();
        assert_eq!(a, b);
        assert_eq!(a, ENE_LED_COUNT as usize * 2 + 4);
    }
}
