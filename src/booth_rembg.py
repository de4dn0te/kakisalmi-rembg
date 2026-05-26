#!/usr/bin/env python3
"""
Video background removal using rembg.
Outputs a transparent .webm/.mov, or composites onto a solid/image background.

Usage:
    # Transparent output (WebM with alpha)
    python remove_bg.py input.mp4 output.webm

    # Composite onto a colour
    python remove_bg.py input.mp4 output.mp4 --bg-color 0,255,0

    # Composite onto an image
    python remove_bg.py input.mp4 output.mp4 --bg-image background.jpg

    # Use a specific rembg model (default: u2net)
    python remove_bg.py input.mp4 output.webm --model birefnet-general

Options:
    --model         rembg model name (see MODEL NOTES below)
    --bg-color      R,G,B background colour (0-255)
    --bg-image      Path to a background image/video frame
    --fps           Override output FPS (default: match source)
    --start         Start time in seconds (default: 0)
    --end           End time in seconds (default: end of video)
    --workers       Parallel worker threads (default: 2)
    --no-gpu        Disable GPU/ONNX GPU provider

MODEL NOTES:
    u2net               Default. Good general-purpose, fast.
    u2net_human_seg     Tuned for people — better than u2net for humans.
    birefnet-general    Best quality overall. Slower, higher VRAM.
    birefnet-portrait   Best for portrait/bust shots of people.
    isnet-general-use   Strong edges, good alternative to birefnet.
    silueta             Lightweight, fast, less accurate.
"""

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from rembg import new_session, remove
from tqdm import tqdm


# --- Helpers ---

def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open video: {path}")
    return cap


def video_meta(cap: cv2.VideoCapture) -> dict:
    return {
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def make_writer(path: str, fps: float, width: int, height: int, alpha: bool) -> cv2.VideoWriter:
    ext = Path(path).suffix.lower()

    if alpha:
        if ext == ".webm":
            fourcc = cv2.VideoWriter_fourcc(*"VP90")
        elif ext in (".mov", ".avi"):
            fourcc = cv2.VideoWriter_fourcc(*"png ")   # PNG codec for lossless alpha
        else:
            print(f"[warn] Alpha channel requested but '{ext}' may not support it. "
                  "Use .webm or .mov for transparency.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, fps, (width, height), isColor=True)
    else:
        if ext == ".mp4":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif ext == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        elif ext == ".webm":
            fourcc = cv2.VideoWriter_fourcc(*"VP90")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, fps, (width, height))


def load_bg_image(path: str, width: int, height: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"[error] Cannot read background image: {path}")
    return cv2.resize(img, (width, height))


def composite_on_color(rgba: np.ndarray, bg_color: tuple[int, int, int]) -> np.ndarray:
    """Blend RGBA frame onto a solid colour. Returns BGR."""
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg    = rgba[:, :, :3].astype(np.float32)
    bg    = np.full_like(fg, bg_color[::-1], dtype=np.float32)  # RGB→BGR
    out   = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    return out


def composite_on_image(rgba: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Blend RGBA frame onto a BGR background image. Returns BGR."""
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg    = rgba[:, :, :3].astype(np.float32)
    bg_f  = bg.astype(np.float32)
    out   = (fg * alpha + bg_f * (1.0 - alpha)).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

_session_local = threading.local()

def process_frame(
    frame_bgr: np.ndarray,
    model_name: str,
    providers: list[str],
) -> np.ndarray:
    """Remove background from a single BGR frame. Returns RGBA numpy array."""
    # Each thread gets its own rembg session (not thread-safe to share)
    if not hasattr(_session_local, "session"):
        _session_local.session = new_session(model_name, providers=providers)

    # rembg expects PIL or bytes; convert BGR→RGB bytes via PNG
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    success, buf = cv2.imencode(".png", rgb)
    if not success:
        raise RuntimeError("Failed to encode frame as PNG")

    result_bytes = remove(buf.tobytes(), session=_session_local.session)
    result_arr   = np.frombuffer(result_bytes, dtype=np.uint8)
    rgba         = cv2.imdecode(result_arr, cv2.IMREAD_UNCHANGED)  # RGBA

    if rgba is None or rgba.shape[2] != 4:
        raise RuntimeError("rembg did not return an RGBA image")
    return rgba


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    cap  = open_video(args.input)
    meta = video_meta(cap)

    fps    = args.fps or meta["fps"]
    W, H   = meta["width"], meta["height"]
    total  = meta["total"]

    # Seek to start frame
    start_frame = int((args.start or 0) * meta["fps"])
    end_frame   = int(args.end * meta["fps"]) if args.end else total

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    n_frames = end_frame - start_frame

    # Determine output mode
    ext        = Path(args.output).suffix.lower()
    alpha_mode = (args.bg_color is None and args.bg_image is None)

    # ONNX providers
    providers = ["CPUExecutionProvider"] if args.no_gpu else \
                ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Background image (loaded once)
    bg_img = None
    if args.bg_image:
        bg_img = load_bg_image(args.bg_image, W, H)

    # Output writer
    writer = make_writer(args.output, fps, W, H, alpha=alpha_mode)

    print(f"[info] Input  : {args.input}  ({W}×{H} @ {meta['fps']:.2f} fps, {total} frames)")
    print(f"[info] Output : {args.output}  ({'transparent' if alpha_mode else 'composited'})")
    print(f"[info] Model  : {args.model}")
    print(f"[info] Frames : {start_frame}–{end_frame} ({n_frames} frames)")
    print(f"[info] Workers: {args.workers}")

    # Read all frames into memory (batched for thread safety)
    # For very long videos you may want to chunk this
    print("[info] Reading frames...")
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    print(f"[info] Processing {len(frames)} frames with rembg ({args.model})...")

    # Process frames in parallel
    results = [None] * len(frames)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {
            pool.submit(process_frame, f, args.model, providers): i
            for i, f in enumerate(frames)
        }
        with tqdm(total=len(frames), unit="frame") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rgba = future.result()
                    results[idx] = rgba
                except Exception as e:
                    print(f"\n[warn] Frame {idx} failed: {e} — using blank frame")
                    results[idx] = np.zeros((H, W, 4), dtype=np.uint8)
                pbar.update(1)

    # Write output in order
    print("[info] Writing output video...")
    for rgba in tqdm(results, unit="frame"):
        if alpha_mode:
            # Write RGBA as BGRA
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            writer.write(bgra)
        elif bg_img is not None:
            bgr = composite_on_image(rgba, bg_img)
            writer.write(bgr)
        else:
            bgr = composite_on_color(rgba, args.bg_color)
            writer.write(bgr)

    writer.release()
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"[done] Saved → {args.output}  ({size_mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser(
        description="Remove background from a video using rembg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input",  help="Input video path")
    p.add_argument("output", help="Output video path (.webm/.mov for alpha, .mp4 for composite)")

    args = p.parse_args()

    run(args)


if __name__ == "__main__":
    main()