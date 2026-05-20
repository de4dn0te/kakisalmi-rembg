#!/usr/bin/env python3
"""
Museum Story Booth — Video Processing Pipeline (MoviePy)
=========================================================
Produces a panning photo-reel composite from a raw booth recording.
This version uses MoviePy + Pillow for compositing, which means all
the pipeline logic is readable Python rather than FFmpeg filter strings.

Install
-------
  pip install moviepy pillow numpy watchdog

  MoviePy wraps FFmpeg under the hood for encoding; FFmpeg must be on PATH.
  Unlike the pure-FFmpeg version, there is no ImageMagick dependency —
  all text rendering uses Pillow directly.

Usage
-----
  python booth_moviepy.py process recording.mp4 ./output --speaker "Jane Smith"
  python booth_moviepy.py watch   ./watch_folder ./output --title "Community Voices"
"""

import argparse
import json
import os
import sys
import time
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    # MoviePy v1.x
    from moviepy.editor import ColorClip, CompositeVideoClip, ImageClip, VideoFileClip
except ModuleNotFoundError:
    # MoviePy v2.x (removed .editor submodule)
    from moviepy import ColorClip, CompositeVideoClip, ImageClip, VideoFileClip


# ── Configuration ──────────────────────────────────────────────────────────────

FRAME_INTERVAL = 5       # Extract one key frame every N seconds
PANEL_WIDTH    = 320     # Each filmstrip panel width in px
PANEL_HEIGHT   = 240     # Each filmstrip panel height in px
BORDER         = 8       # White border around each panel in px
PANEL_GAP      = 10      # Horizontal gap between panels in px

OUTPUT_WIDTH   = 1920
OUTPUT_HEIGHT  = 1080
OUTPUT_FPS     = 25

PAN_SPEED      = 55      # Pixels/second the reel scrolls
SUBJECT_SCALE  = 0.65   # Subject video height as fraction of OUTPUT_HEIGHT

GRAIN_STRENGTH = 16      # Standard deviation of noise (0–50)
VIGNETTE_POWER = 2.5     # Higher = stronger vignette falloff

EXHIBIT_NAME   = "Stories from the Community"

# System fonts — Pillow will fall back to its built-in if these are absent
FONT_PATH_BOLD    = "../../lib/fonts/DejaVuSans-Bold.ttf"
FONT_PATH_REGULAR = "../../lib/fonts/DejaVuSans.ttf"

FFMPEG_BIN  = r"C:\Users\x119877\ffmpeg\bin\ffmpeg.exe"
FFPROBE_BIN = r"C:\Users\x119877\ffmpeg\bin\ffprobe.exe"

# Tell MoviePy where ffmpeg lives (needed when ffmpeg is not on PATH)
os.environ["FFMPEG_BINARY"]  = FFMPEG_BIN
os.environ["FFPROBE_BINARY"] = FFPROBE_BIN


# ── Font loader ────────────────────────────────────────────────────────────────

def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except (IOError, OSError):
        return ImageFont.load_default()


# ── Sepia ──────────────────────────────────────────────────────────────────────

def apply_sepia(img: Image.Image) -> Image.Image:
    """Apply a classic sepia tone to a PIL image."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    arr[:, :, 0] = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255)
    arr[:, :, 1] = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255)
    arr[:, :, 2] = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# ── Grain ──────────────────────────────────────────────────────────────────────

def make_grain_effect(strength: int = GRAIN_STRENGTH):
    """
    Returns a MoviePy fl_image function that adds temporal film grain.
    Using a closure lets us precompute the RNG once and vary per-frame.
    """
    rng = np.random.default_rng()

    def add_grain(frame: np.ndarray) -> np.ndarray:
        noise = rng.normal(0, strength, frame.shape).astype(np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return add_grain


# ── Vignette ──────────────────────────────────────────────────────────────────

def make_vignette_mask(width: int, height: int, power: float = VIGNETTE_POWER) -> np.ndarray:
    """
    Returns a (H, W, 1) float32 mask where 1.0 = centre (no darkening)
    and 0.0 = corners (fully dark). Applied once, reused every frame.
    """
    cx, cy = width / 2, height / 2
    y, x   = np.ogrid[:height, :width]
    dist   = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    mask   = np.clip(1.0 - (dist ** power) * 0.6, 0.0, 1.0)
    return mask[:, :, np.newaxis]   # broadcast over RGB channels


def make_vignette_effect(width: int, height: int):
    """Returns a MoviePy fl_image function that applies a vignette."""
    mask = make_vignette_mask(width, height)

    def apply_vignette(frame: np.ndarray) -> np.ndarray:
        return (frame.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)

    return apply_vignette


# ── Stage 1: Extract key frames ───────────────────────────────────────────────

def extract_frames(input_path: Path, frames_dir: Path) -> list[Path]:
    """
    Pull frames from the video at FRAME_INTERVAL seconds using MoviePy.
    MoviePy gives us direct numpy array access — no FFmpeg subprocess needed.
    """
    print("\n[1/5] Extracting key frames...")
    frames_dir.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(str(input_path)) as clip:
        duration  = clip.duration
        times     = list(range(0, int(duration), FRAME_INTERVAL))
        out_paths = []

        for i, t in enumerate(times):
            frame     = clip.get_frame(t)                   # numpy (H, W, 3)
            img       = Image.fromarray(frame)
            img       = img.resize(
                (PANEL_WIDTH, PANEL_HEIGHT), Image.LANCZOS
            )
            out_path  = frames_dir / f"frame_{i:04d}.jpg"
            img.save(out_path, quality=92)
            out_paths.append(out_path)

    print(f"   → {len(out_paths)} frames extracted")
    return out_paths


# ── Stage 2: Build filmstrip image ────────────────────────────────────────────

def build_filmstrip(frames: list[Path]) -> tuple[Image.Image, int, int]:
    """
    Stitch frames into a single wide sepia filmstrip PIL image.
    Returns (image, strip_width, strip_height).

    Doing this in Pillow is far more readable than the equivalent
    FFmpeg hstack filter chain.
    """
    print("\n[2/5] Building filmstrip image...")

    if not frames:
        raise RuntimeError("No frames to stitch — check your input video.")

    panel_w = PANEL_WIDTH  + BORDER * 2
    panel_h = PANEL_HEIGHT + BORDER * 2
    n       = len(frames)
    strip_w = n * panel_w + (n - 1) * PANEL_GAP
    strip_h = panel_h

    strip = Image.new("RGB", (strip_w, strip_h), color=(26, 26, 26))

    for i, frame_path in enumerate(frames):
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((PANEL_WIDTH, PANEL_HEIGHT), Image.LANCZOS)
        img = apply_sepia(img)

        # White-bordered panel
        panel = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
        panel.paste(img, (BORDER, BORDER))

        x = i * (panel_w + PANEL_GAP)
        strip.paste(panel, (x, 0))

    print(f"   → Filmstrip: {strip_w} × {strip_h} px  ({n} panels)")
    return strip, strip_w, strip_h


# ── Stage 3: Composite ────────────────────────────────────────────────────────

def composite(
    input_path:  Path,
    strip_img:   Image.Image,
    strip_w:     int,
    strip_h:     int,
    output_path: Path,
    metadata:    dict,
) -> None:
    """
    The main creative stage — pure MoviePy compositing:
      • Dark background
      • Filmstrip ImageClip panning behind the speaker
      • Speaker video centred and scaled
      • Film grain + vignette applied as per-frame functions
      • Title card burned in via Pillow (no ImageMagick needed)
    """
    print("\n[3/5] Compositing panning reel + subject video...")

    clip     = VideoFileClip(str(input_path))
    duration = clip.duration

    # ── Subject clip (centred) ────────────────────────────────────────────────
    subj_h   = int(OUTPUT_HEIGHT * SUBJECT_SCALE)
    subj_w   = int(subj_h * clip.w / clip.h)
    subj_x   = (OUTPUT_WIDTH  - subj_w) // 2
    subj_y   = (OUTPUT_HEIGHT - subj_h) // 2
    subject  = clip.resized((subj_w, subj_h))

    # ── Tiled filmstrip (wide enough to cover full pan travel) ────────────────
    total_travel = int(duration * PAN_SPEED) + OUTPUT_WIDTH
    tile_copies  = max(2, total_travel // strip_w + 2)
    tiled_w      = tile_copies * strip_w
    strip_y      = (OUTPUT_HEIGHT - strip_h) // 2

    tiled = Image.new("RGB", (tiled_w, strip_h), color=(26, 26, 26))
    for i in range(tile_copies):
        tiled.paste(strip_img, (i * strip_w, 0))

    tiled_np   = np.array(tiled)
    reel_clip  = (
        ImageClip(tiled_np)
        .with_duration(duration)
        # Pan: x moves left over time; y stays fixed at strip_y
        .with_position(lambda t: (int(-t * PAN_SPEED), strip_y))
    )

    # ── Title card (Pillow-rendered, overlaid as ImageClip) ───────────────────
    title_img = render_title_card(OUTPUT_WIDTH, OUTPUT_HEIGHT, metadata)
    title_clip = (
        ImageClip(np.array(title_img))
        .with_duration(duration)
        .with_opacity(1.0)
        .with_position((0, 0))
    )

    # ── Background ────────────────────────────────────────────────────────────
    background = ColorClip(
        size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
        color=(17, 17, 17),
        duration=duration,
    )

    # ── Composite layers (bottom → top) ──────────────────────────────────────
    composite_clip = CompositeVideoClip([
        background,
        reel_clip,
        subject.with_position((subj_x, subj_y)),
        title_clip,
    ], size=(OUTPUT_WIDTH, OUTPUT_HEIGHT))

    # ── Per-frame effects: grain then vignette ────────────────────────────────
    grain    = make_grain_effect(GRAIN_STRENGTH)
    vignette = make_vignette_effect(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    final    = composite_clip.image_transform(lambda f: vignette(grain(f)))

    # ── Render ────────────────────────────────────────────────────────────────
    print("   Rendering… (this may take a while)")
    final.write_videofile(
        str(output_path),
        fps=OUTPUT_FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate="8000k",
        audio=True,
    )
    clip.close()


def render_title_card(width: int, height: int, metadata: dict) -> Image.Image:
    """
    Render a transparent title card as a PIL RGBA image using Pillow.
    Returns an image that can be composited directly — no ImageMagick needed.
    """
    img  = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    title   = metadata.get("title",   EXHIBIT_NAME)
    date    = metadata.get("date",    datetime.now().strftime("%B %d, %Y"))
    speaker = metadata.get("speaker", "")

    font_title  = load_font(FONT_PATH_BOLD,    28)
    font_sub    = load_font(FONT_PATH_REGULAR, 20)
    font_speaker = load_font(FONT_PATH_REGULAR, 22)

    def centred_text(text, y, font, color):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw   = bbox[2] - bbox[0]
        x    = (width - tw) // 2
        # Shadow
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 160))
        draw.text((x,     y    ), text, font=font, fill=color)

    centred_text(title, height - 70, font_title,   (255, 255, 255, 217))
    centred_text(date,  height - 36, font_sub,     (221, 221, 221, 178))
    if speaker:
        centred_text(speaker, 30,   font_speaker,  (255, 213, 128, 230))

    return img


# ── Stage 4: Archive copy ─────────────────────────────────────────────────────

def archive_copy(display_path: Path, archive_path: Path) -> None:
    """Transcode the display MP4 to a lossless ProRes HQ archive."""
    print("\n[4/5] Writing archive copy (ProRes HQ)...")
    subprocess.run([
        FFMPEG_BIN, "-y",
        "-i",    str(display_path),
        "-c:v",  "prores_ks", "-profile:v", "3",
        "-c:a",  "pcm_s16le",
        str(archive_path),
    ], check=True, stderr=subprocess.DEVNULL)


# ── Stage 5: Sidecar metadata ─────────────────────────────────────────────────

def write_sidecar(output_path: Path, metadata: dict, duration: float) -> None:
    print("\n[5/5] Writing sidecar metadata...")
    data = {
        "content_id":       output_path.stem,
        "exhibit":          metadata.get("title",   EXHIBIT_NAME),
        "speaker":          metadata.get("speaker", ""),
        "date_recorded":    metadata.get("date",    datetime.now().isoformat()),
        "duration_seconds": round(duration, 2),
        "pipeline_version": "1.0 (moviepy)",
    }
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(json.dumps(data, indent=2))
    print(f"   → {sidecar}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def process(input_path: str, output_dir: str, metadata: dict = None) -> tuple[Path, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = {}
    metadata.setdefault("title", EXHIBIT_NAME)
    metadata.setdefault("date",  datetime.now().strftime("%B %d, %Y"))

    stem        = input_path.stem
    display_out = output_dir / f"{stem}_display.mp4"
    archive_out = output_dir / f"{stem}_archive.mov"

    print(f"\n{'='*60}")
    print(f"  Processing: {input_path.name}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        frames              = extract_frames(input_path, tmp / "frames")
        strip_img, sw, sh   = build_filmstrip(frames)
        composite(input_path, strip_img, sw, sh, display_out, metadata)

        with VideoFileClip(str(input_path)) as c:
            duration = c.duration

        write_sidecar(display_out, metadata, duration)
        archive_copy(display_out, archive_out)

    print(f"\n✓  Display copy  → {display_out}")
    print(f"✓  Archive copy  → {archive_out}")
    return display_out, archive_out


# ── Watchdog automation ───────────────────────────────────────────────────────

def run_watchdog(watch_dir: str, output_dir: str, default_title: str = EXHIBIT_NAME):
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        sys.exit("watchdog not installed — run:  pip install watchdog")

    EXTENSIONS = {".mp4", ".mov", ".mxf", ".avi"}

    class BoothHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() not in EXTENSIONS:
                return
            time.sleep(4)
            print(f"\n★  New recording detected: {path.name}")
            try:
                process(str(path), output_dir, {
                    "title": default_title,
                    "date":  datetime.now().strftime("%B %d, %Y"),
                })
            except Exception as exc:
                print(f"\n✗  Pipeline failed for {path.name}: {exc}")

    observer = Observer()
    observer.schedule(BoothHandler(), watch_dir, recursive=False)
    observer.start()
    print(f"Watching {watch_dir!r} for new recordings…  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Museum story booth — MoviePy pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="Process a single recording")
    p.add_argument("input")
    p.add_argument("output_dir")
    p.add_argument("--title",   default=EXHIBIT_NAME)
    p.add_argument("--speaker", default="")

    w = sub.add_parser("watch", help="Watch a folder and auto-process")
    w.add_argument("watch_dir")
    w.add_argument("output_dir")
    w.add_argument("--title", default=EXHIBIT_NAME)

    args = parser.parse_args()

    if args.cmd == "process":
        process(args.input, args.output_dir, {
            "title":   args.title,
            "speaker": args.speaker,
            "date":    datetime.now().strftime("%B %d, %Y"),
        })
    elif args.cmd == "watch":
        run_watchdog(args.watch_dir, args.output_dir, args.title)


if __name__ == "__main__":
    main()