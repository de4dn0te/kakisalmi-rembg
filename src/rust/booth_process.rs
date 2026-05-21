//! Museum Story Booth — Video Processing Pipeline (Rust)
//! =======================================================
//! Produces a panning photo-reel composite from a raw booth recording.
//!
//! Build
//! -----
//!   cargo build --release
//!   Cross-compile for Windows from Linux/macOS:
//!   cargo build --release --target x86_64-pc-windows-gnu
//!
//! Usage
//! -----
//!   booth process recording.mp4 ./output --speaker "Jane Smith"
//!   booth watch   ./watch_folder ./output --title "Community Voices"
//!
//! Dependencies
//! ------------
//!   ffmpeg + ffprobe must be available (configure paths in Config below)

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use notify::{EventKind, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};

// ── Configuration ─────────────────────────────────────────────────────────────

const FRAME_INTERVAL:  u32  = 5;    // Extract one key frame every N seconds
const PANEL_WIDTH:     u32  = 320;
const PANEL_HEIGHT:    u32  = 240;
const BORDER:          u32  = 8;    // White border around each panel in px
const PANEL_GAP:       u32  = 10;   // Gap between panels in px

const OUTPUT_WIDTH:    u32  = 1920;
const OUTPUT_HEIGHT:   u32  = 1080;
const OUTPUT_FPS:      u32  = 25;

const PAN_SPEED:       u32  = 55;   // Pixels/second the reel scrolls
const SUBJECT_SCALE:   f64  = 0.65; // Subject height as fraction of OUTPUT_HEIGHT

const GRAIN_STRENGTH:  u32  = 16;
const VIGNETTE_ANGLE:  &str = "PI/4";

const DEFAULT_EXHIBIT: &str = "Stories from the Community";

const VIDEO_EXTENSIONS: &[&str] = &[".mp4", ".mov", ".mxf", ".avi"];

// ── Runtime config (paths resolved relative to the executable) ────────────────

struct Config {
    ffmpeg:      PathBuf,
    ffprobe:     PathBuf,
    font_bold:   PathBuf,
    font_regular: PathBuf,
}

impl Config {
    fn new() -> Self {
        // Executable lives in build/; repo root is one level up
        let exe     = env::current_exe().expect("cannot locate executable");
        let repo    = exe.parent().unwrap().parent().unwrap();
        let fonts   = repo.join("lib").join("fonts");

        // ffmpeg lives at lib/ffmpeg/bin/ in the repo,
        // matching the school machine layout where it was installed from source
        let ffmpeg_bin = repo.join("lib").join("ffmpeg").join("bin");

        Config {
            ffmpeg:       ffmpeg_bin.join("ffmpeg.exe"),
            ffprobe:      ffmpeg_bin.join("ffprobe.exe"),
            font_bold:    fonts.join("DejaVuSans-Bold.ttf"),
            font_regular: fonts.join("DejaVuSans.ttf"),
        }
    }
}

// ── Metadata ──────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct Sidecar {
    content_id:       String,
    exhibit:          String,
    speaker:          String,
    date_recorded:    String,
    duration_seconds: f64,
    pipeline_version: String,
}

// ── Job (what gets processed) ─────────────────────────────────────────────────

#[derive(Clone)]
struct Job {
    title:   String,
    speaker: String,
    date:    String,
}

impl Job {
    fn new(title: &str, speaker: &str) -> Self {
        Job {
            title:   title.to_string(),
            speaker: speaker.to_string(),
            date:    chrono_date(),
        }
    }
}

fn chrono_date() -> String {
    // std only; no chrono crate needed for a simple formatted date
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // Days since epoch → approximate date string
    let days  = secs / 86400;
    let (y, m, d) = days_to_ymd(days);
    let months = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"];
    format!("{} {:02}, {}", months[(m - 1) as usize], d, y)
}

/// Naive Gregorian conversion (no external crate needed).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970u64;
    loop {
        let leap = is_leap(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year { break; }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: &[u64] = if leap {
        &[31,29,31,30,31,30,31,31,30,31,30,31]
    } else {
        &[31,28,31,30,31,30,31,31,30,31,30,31]
    };
    let mut month = 1u64;
    for &md in month_days {
        if days < md { break; }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool { y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) }

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Run a command; stream stderr to our stdout; panic on non-zero exit.
fn run(label: &str, program: &Path, args: &[&str]) {
    if !label.is_empty() {
        println!("   {label}");
    }
    let preview = format!("{} {}", program.display(), args.join(" "));
    println!("   $ {:.120}", preview);

    let status = Command::new(program)
        .args(args)
        .stderr(Stdio::inherit())
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", program.display()));

    if !status.success() {
        panic!("command failed with status {status}");
    }
}

/// Run ffprobe and return its JSON output as a parsed value.
fn probe(ffprobe: &Path, path: &Path) -> serde_json::Value {
    let out = Command::new(ffprobe)
        .args(["-v", "quiet", "-print_format", "json",
               "-show_format", "-show_streams",
               path.to_str().unwrap()])
        .output()
        .expect("ffprobe failed");

    serde_json::from_slice(&out.stdout).expect("ffprobe JSON parse error")
}

fn get_duration(ffprobe: &Path, path: &Path) -> f64 {
    let v = probe(ffprobe, path);
    v["format"]["duration"]
        .as_str()
        .unwrap()
        .parse::<f64>()
        .unwrap()
}

fn get_video_size(ffprobe: &Path, path: &Path) -> (u32, u32) {
    let v = probe(ffprobe, path);
    let streams = v["streams"].as_array().unwrap();
    for s in streams {
        if s["codec_type"] == "video" {
            let w = s["width"].as_u64().unwrap()  as u32;
            let h = s["height"].as_u64().unwrap() as u32;
            return (w, h);
        }
    }
    panic!("no video stream in {}", path.display());
}

/// Escape characters special to FFmpeg's drawtext filter.
fn esc(s: &str) -> String {
    s.replace('\\', r"\\")
     .replace('\'', r"\'")
     .replace(':', r"\:")
}

// ── Stage 1: Extract key frames ───────────────────────────────────────────────

fn extract_frames(cfg: &Config, input: &Path, frames_dir: &Path) -> Vec<PathBuf> {
    println!("\n[1/5] Extracting key frames...");
    fs::create_dir_all(frames_dir).unwrap();

    let vf = format!(
        "fps=1/{FRAME_INTERVAL},\
         scale={PANEL_WIDTH}:{PANEL_HEIGHT}:force_original_aspect_ratio=increase,\
         crop={PANEL_WIDTH}:{PANEL_HEIGHT}"
    );
    let out_pattern = frames_dir.join("frame_%04d.jpg");

    run("", &cfg.ffmpeg, &[
        "-y", "-i", input.to_str().unwrap(),
        "-vf", &vf,
        "-q:v", "2",
        out_pattern.to_str().unwrap(),
    ]);

    let mut frames: Vec<PathBuf> = fs::read_dir(frames_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "jpg").unwrap_or(false))
        .collect();
    frames.sort();

    println!("   → {} frames extracted", frames.len());
    frames
}

// ── Stage 2: Build filmstrip ──────────────────────────────────────────────────

fn build_filmstrip(
    cfg:       &Config,
    frames:    &[PathBuf],
    tmp_dir:   &Path,
) -> (PathBuf, u32, u32) {
    println!("\n[2/5] Building filmstrip image...");
    assert!(!frames.is_empty(), "no frames to stitch");

    let n        = frames.len() as u32;
    let panel_w  = PANEL_WIDTH  + BORDER * 2;
    let panel_h  = PANEL_HEIGHT + BORDER * 2;
    let strip_w  = n * panel_w + (n - 1) * PANEL_GAP;
    let strip_h  = panel_h;
    let out_path = tmp_dir.join("filmstrip.png");

    // Build inputs list
    let mut args: Vec<String> = vec!["-y".into()];
    for f in frames {
        args.push("-i".into());
        args.push(f.to_str().unwrap().into());
    }

    let sepia =
        "colorchannelmixer=\
         rr=0.393:rg=0.769:rb=0.189:\
         gr=0.349:gg=0.686:gb=0.168:\
         br=0.272:bg=0.534:bb=0.131";

    let mut filter_parts: Vec<String> = Vec::new();

    // Per-frame: scale → white border pad → sepia
    for i in 0..frames.len() {
        filter_parts.push(format!(
            "[{i}:v]scale={PANEL_WIDTH}:{PANEL_HEIGHT}:\
             force_original_aspect_ratio=increase,\
             crop={PANEL_WIDTH}:{PANEL_HEIGHT},\
             pad={panel_w}:{panel_h}:{BORDER}:{BORDER}:color=white,\
             {sepia}[p{i}]"
        ));
    }

    // Add right-side gap then hstack
    for i in 0..frames.len() {
        let extra = if i < frames.len() - 1 { PANEL_GAP } else { 0 };
        filter_parts.push(format!(
            "[p{i}]pad={}:{panel_h}:0:0:color=0x1a1a1a[g{i}]",
            panel_w + extra
        ));
    }

    let stack_inputs: String = (0..frames.len()).map(|i| format!("[g{i}]")).collect();
    filter_parts.push(format!("{stack_inputs}hstack=inputs={}[strip]", frames.len()));

    args.extend([
        "-filter_complex".into(), filter_parts.join(";"),
        "-map".into(), "[strip]".into(),
        "-frames:v".into(), "1".into(),
        out_path.to_str().unwrap().into(),
    ]);

    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run("Stitching panels with hstack…", &cfg.ffmpeg, &arg_refs);

    (out_path, strip_w, strip_h)
}

// ── Stage 3: Composite ────────────────────────────────────────────────────────

fn composite(
    cfg:        &Config,
    input:      &Path,
    strip_path: &Path,
    strip_w:    u32,
    strip_h:    u32,
    output:     &Path,
    job:        &Job,
) {
    println!("\n[3/5] Compositing panning reel + subject video...");

    let duration       = get_duration(&cfg.ffprobe, input);
    let (src_w, src_h) = get_video_size(&cfg.ffprobe, input);

    // Subject dimensions
    let subj_h = (OUTPUT_HEIGHT as f64 * SUBJECT_SCALE) as u32;
    let subj_w = (subj_h as f64 * src_w as f64 / src_h as f64) as u32;
    let subj_x = (OUTPUT_WIDTH  - subj_w) / 2;
    let subj_y = (OUTPUT_HEIGHT - subj_h) / 2;

    // Tile copies
    let total_travel = (duration * PAN_SPEED as f64) as u32 + OUTPUT_WIDTH;
    let tile_copies  = (total_travel / strip_w + 2).max(2);
    let strip_y      = (OUTPUT_HEIGHT - strip_h) / 2;
    let pan_x        = format!("-(t*{PAN_SPEED})");

    // Drawtext
    let title   = esc(&job.title);
    let date    = esc(&job.date);
    let speaker = esc(&job.speaker);
    let fb      = cfg.font_bold.to_str().unwrap();
    let fr      = cfg.font_regular.to_str().unwrap();

    let mut drawtext = format!(
        "drawtext=fontfile='{fb}':text='{title}':\
         fontcolor=white:fontsize=28:alpha=0.85:\
         x=(w-text_w)/2:y=h-70:\
         shadowcolor=black:shadowx=1:shadowy=1,\
         drawtext=fontfile='{fr}':text='{date}':\
         fontcolor=0xdddddd:fontsize=20:alpha=0.7:\
         x=(w-text_w)/2:y=h-36:\
         shadowcolor=black:shadowx=1:shadowy=1"
    );
    if !speaker.is_empty() {
        drawtext.push_str(&format!(
            ",drawtext=fontfile='{fr}':text='{speaker}':\
             fontcolor=0xffd580:fontsize=22:alpha=0.90:\
             x=(w-text_w)/2:y=30:\
             shadowcolor=black:shadowx=1:shadowy=1"
        ));
    }

    let filter_complex = [
        format!("[1:v]tile={tile_copies}x1[strip_tiled]"),
        format!("[strip_tiled]crop={OUTPUT_WIDTH}:{strip_h}:'{pan_x}':0[reel]"),
        format!("color=c=0x111111:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:r={OUTPUT_FPS}[bg]"),
        format!("[bg][reel]overlay=0:{strip_y}[bg_reel]"),
        format!("[0:v]scale={subj_w}:{subj_h}[subject]"),
        format!("[bg_reel][subject]overlay={subj_x}:{subj_y}[with_subject]"),
        format!("[with_subject]noise=alls={GRAIN_STRENGTH}:allf=t+u[grainy]"),
        format!("[grainy]vignette=angle={VIGNETTE_ANGLE}:mode=forward[vignetted]"),
        format!("[vignetted]{drawtext}[out]"),
    ].join(";");

    let duration_str  = format!("{:.3}", duration);
    let fps_str       = OUTPUT_FPS.to_string();

    run("Rendering composite (this may take a while)…", &cfg.ffmpeg, &[
        "-y",
        "-i", input.to_str().unwrap(),
        "-i", strip_path.to_str().unwrap(),
        "-filter_complex", &filter_complex,
        "-map", "[out]",
        "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-t", &duration_str,
        "-r", &fps_str,
        output.to_str().unwrap(),
    ]);
}

// ── Stage 4: Archive copy ─────────────────────────────────────────────────────

fn archive_copy(cfg: &Config, display: &Path, archive: &Path) {
    println!("\n[4/5] Writing archive copy (ProRes HQ)...");
    run("", &cfg.ffmpeg, &[
        "-y",
        "-i",       display.to_str().unwrap(),
        "-c:v",     "prores_ks",
        "-profile:v", "3",
        "-c:a",     "pcm_s16le",
        archive.to_str().unwrap(),
    ]);
}

// ── Stage 5: Sidecar metadata ─────────────────────────────────────────────────

fn write_sidecar(output: &Path, job: &Job, duration: f64) {
    println!("\n[5/5] Writing sidecar metadata...");
    let stem    = output.file_stem().unwrap().to_str().unwrap();
    let exhibit = if job.title.is_empty() { DEFAULT_EXHIBIT.into() } else { job.title.clone() };

    let data = Sidecar {
        content_id:       stem.into(),
        exhibit,
        speaker:          job.speaker.clone(),
        date_recorded:    job.date.clone(),
        duration_seconds: (duration * 100.0).round() / 100.0,
        pipeline_version: "1.0 (rust)".into(),
    };

    let sidecar = output.with_extension("json");
    let json    = serde_json::to_string_pretty(&data).unwrap();
    fs::write(&sidecar, json).unwrap();
    println!("   → {}", sidecar.display());
}

// ── Orchestrator ──────────────────────────────────────────────────────────────

fn process(cfg: &Config, input: &Path, output_dir: &Path, job: &Job) {
    fs::create_dir_all(output_dir).unwrap();

    let stem        = input.file_stem().unwrap().to_str().unwrap();
    let display_out = output_dir.join(format!("{stem}_display.mp4"));
    let archive_out = output_dir.join(format!("{stem}_archive.mov"));

    println!("\n{}", "=".repeat(60));
    println!("  Processing: {}", input.display());
    println!("{}", "=".repeat(60));

    let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let tmp     = tmp_dir.path();

    let frames = extract_frames(cfg, input, &tmp.join("frames"));
    let (strip_path, strip_w, strip_h) = build_filmstrip(cfg, &frames, tmp);
    composite(cfg, input, &strip_path, strip_w, strip_h, &display_out, job);

    let duration = get_duration(&cfg.ffprobe, input);
    write_sidecar(&display_out, job, duration);
    archive_copy(cfg, &display_out, &archive_out);

    println!("\n✓  Display copy  → {}", display_out.display());
    println!("✓  Archive copy  → {}", archive_out.display());
}

// ── Watch ─────────────────────────────────────────────────────────────────────

fn watch(cfg: &Config, watch_dir: &Path, output_dir: &Path, default_title: &str) {
    use notify::event::CreateKind;
    use std::sync::mpsc;

    let (tx, rx) = mpsc::channel::<notify::Result<notify::Event>>();
    let mut watcher = notify::recommended_watcher(tx)
        .expect("failed to create watcher");
    watcher.watch(watch_dir, RecursiveMode::NonRecursive)
        .expect("failed to watch directory");

    println!("Watching {:?} for new recordings…  (Ctrl+C to stop)", watch_dir);

    for res in rx {
        match res {
            Ok(event) => {
                if !matches!(event.kind, EventKind::Create(CreateKind::File)) {
                    continue;
                }
                for path in event.paths {
                    let ext = path.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| format!(".{e}").to_lowercase())
                        .unwrap_or_default();

                    if !VIDEO_EXTENSIONS.contains(&ext.as_str()) {
                        continue;
                    }

                    // Wait for camera software to finish writing
                    thread::sleep(Duration::from_secs(4));

                    println!("\n★  New recording detected: {}", path.display());
                    let job        = Job::new(default_title, "");
                    let output_dir = output_dir.to_path_buf();
                    let cfg_ffmpeg  = cfg.ffmpeg.clone();
                    let cfg_ffprobe = cfg.ffprobe.clone();
                    let cfg_fb      = cfg.font_bold.clone();
                    let cfg_fr      = cfg.font_regular.clone();

                    thread::spawn(move || {
                        let cfg = Config {
                            ffmpeg:       cfg_ffmpeg,
                            ffprobe:      cfg_ffprobe,
                            font_bold:    cfg_fb,
                            font_regular: cfg_fr,
                        };
                        process(&cfg, &path, &output_dir, &job);
                    });
                }
            }
            Err(e) => eprintln!("watcher error: {e}"),
        }
    }
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = Config::new();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  booth process <input> <output_dir> [--title ...] [--speaker ...]");
        eprintln!("  booth watch   <watch_dir> <output_dir> [--title ...]");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "process" => {
            if args.len() < 4 {
                eprintln!("Usage: booth process <input> <output_dir>");
                std::process::exit(1);
            }
            let input      = PathBuf::from(&args[2]);
            let output_dir = PathBuf::from(&args[3]);
            let title      = flag_value(&args, "--title")
                .unwrap_or_else(|| DEFAULT_EXHIBIT.into());
            let speaker    = flag_value(&args, "--speaker")
                .unwrap_or_default();
            let job = Job::new(&title, &speaker);
            process(&cfg, &input, &output_dir, &job);
        }
        "watch" => {
            if args.len() < 4 {
                eprintln!("Usage: booth watch <watch_dir> <output_dir>");
                std::process::exit(1);
            }
            let watch_dir  = PathBuf::from(&args[2]);
            let output_dir = PathBuf::from(&args[3]);
            let title      = flag_value(&args, "--title")
                .unwrap_or_else(|| DEFAULT_EXHIBIT.into());
            watch(&cfg, &watch_dir, &output_dir, &title);
        }
        cmd => {
            eprintln!("Unknown command {cmd:?}. Use 'process' or 'watch'.");
            std::process::exit(1);
        }
    }
}

/// Pull the value after a named flag, e.g. --speaker "Jane" → Some("Jane")
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}