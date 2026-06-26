import argparse
import io
import os
import ffmpeg
import pathlib
import threading
import numpy as np
from queue import Queue
from shutil import rmtree
from PIL import Image
from rembg import new_session, remove

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,managed_memory:True"
)

# Parse args
parser = argparse.ArgumentParser(
    description="Applies rembg background removal to the frames of a video"
)
parser.add_argument("input", type=str, help="Input video")
parser.add_argument(
    "-o", type=str, default="export/output.mov", help="Define output path"
)
parser.add_argument(
    "--model",
    type=str,
    default="u2net_human_seg",
    help="rembg model to use (default: birefnet-general-lite)",
)
parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of concurrent processing workers (default: 1)",
)
parser.add_argument(
    "--smooth",
    type=int,
    default=3,
    help="Temporal mask smoothing window size in frames (default: 3, 0 to disable)",
)
parser.add_argument(
    "--read-ahead",
    type=int,
    default=8,
    help="Number of frames to read ahead into buffer (default: 8)",
)
parser.add_argument(
    "--write-buffer",
    type=int,
    default=8,
    help="Number of processed frames to buffer before writing (default: 8)",
)
parser.add_argument(
    "--smooth-workers",
    type=int,
    default=os.cpu_count() or 4,
    help="Number of threads to use for temporal mask smoothing (default: cpu count)",
)
args = parser.parse_args()


def is_oom_error(exc):
    text = str(exc).lower()
    return any(
        phrase in text
        for phrase in (
            "out of memory",
            "cuda out of memory",
            "failed to allocate",
            "oom",
            "memory error",
        )
    )


def image_to_png_bytes(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def remove_with_mask_fallback(image_bytes, session, scales=(1.0, 0.8, 0.6, 0.4)):
    original = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    width, height = original.size
    last_exc = None

    for scale in scales:
        try:
            if scale == 1.0:
                return remove(image_bytes, session=session)

            resized = original.resize(
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
            with io.BytesIO() as buffer:
                resized.save(buffer, format="PNG")
                scaled_bytes = buffer.getvalue()

            scaled_output = remove(scaled_bytes, session=session)
            if isinstance(scaled_output, (bytes, bytearray)):
                scaled_output_bytes = bytes(scaled_output)
            elif isinstance(scaled_output, np.ndarray):
                with io.BytesIO() as buffer:
                    Image.fromarray(scaled_output).save(buffer, format="PNG")
                    scaled_output_bytes = buffer.getvalue()
            elif isinstance(scaled_output, Image.Image):
                with io.BytesIO() as buffer:
                    scaled_output.save(buffer, format="PNG")
                    scaled_output_bytes = buffer.getvalue()
            else:
                raise RuntimeError(
                    "Unexpected rembg remove() result type during mask fallback."
                )
            alpha = (
                Image.open(io.BytesIO(scaled_output_bytes))
                .convert("RGBA")
                .getchannel("A")
            )
            alpha = alpha.resize((width, height), Image.Resampling.LANCZOS)

            output_full = original.copy()
            output_full.putalpha(alpha)
            return image_to_png_bytes(output_full)

        except Exception as exc:
            last_exc = exc
            if not is_oom_error(exc):
                raise
            if scale == scales[-1]:
                raise RuntimeError(
                    "Insufficient GPU memory: background removal failed even after fallback downscales."
                ) from exc
            print(
                f"OOM detected during removal at scale {scale:.2f}; retrying with lower-resolution mask...",
                flush=True,
            )

    if last_exc is not None:
        raise RuntimeError(
            "Background removal failed unexpectedly during fallback."
        ) from last_exc
    raise RuntimeError("Background removal failed unexpectedly.")


# Extract video info
probe = ffmpeg.probe(args.input)
video_stream = next(
    (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
)
if video_stream is None:
    raise ValueError(f"No video stream found in input file: {args.input}")
width = int(video_stream["width"])
height = int(video_stream["height"])
whstr = str(width) + "x" + str(height)
framerate = video_stream["avg_frame_rate"]

frames_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "frames")
processed_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "processed")

# Extract input video frames
# Always start from a clean slate: a prior crashed/interrupted run may have
# left frames behind (possibly from a different input video, e.g. different
# resolution), which would silently corrupt this run.
rmtree(frames_dir, ignore_errors=True)
os.mkdir(frames_dir)
stream = ffmpeg.input(args.input)
stream = ffmpeg.output(stream, os.path.join(frames_dir, "%04d.bmp"))
ffmpeg.run(stream)

_SENTINEL = object()

try:
    # Process frames with pipelined reader -> processors -> writer
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    files = sorted(os.listdir(frames_dir))
    total_files = len(files)

    print(f"Loading rembg session (model={args.model})...", flush=True)
    session = new_session(
        args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    read_queue = Queue(maxsize=args.read_ahead)
    write_queue = Queue(maxsize=args.write_buffer)
    errors = []
    active_processors = [
        args.workers
    ]  # list so processor() can mutate without nonlocal
    active_processors_lock = threading.Lock()

    def reader():
        try:
            for idx, file in enumerate(files, 1):
                frame_path = os.path.join(frames_dir, file)
                with open(frame_path, "rb") as f:
                    data = f.read()
                read_queue.put((idx, file, data))
                os.remove(frame_path)
        except Exception as e:
            errors.append(e)
        finally:
            # One sentinel per worker so each one knows when to stop
            for _ in range(args.workers):
                read_queue.put(_SENTINEL)

    def processor():
        try:
            while True:
                item = read_queue.get()
                if item is _SENTINEL:
                    break
                idx, file, input_data = item
                print(f"Processing frame {idx}/{total_files}: {file}", flush=True)
                output_data = remove_with_mask_fallback(input_data, session=session)
                write_queue.put((idx, file, output_data))
        except Exception as e:
            errors.append(e)
        finally:
            # Signal writer only when the last processor finishes
            with active_processors_lock:
                active_processors[0] -= 1
                if active_processors[0] == 0:
                    write_queue.put(_SENTINEL)

    def writer():
        try:
            while True:
                item = write_queue.get()
                if item is _SENTINEL:
                    break
                idx, file, output_data = item
                with open(os.path.join(processed_dir, file), "wb") as f:
                    f.write(output_data)
                print(f"Written frame {idx}/{total_files}: {file}", flush=True)
        except Exception as e:
            errors.append(e)

    reader_thread = threading.Thread(target=reader, daemon=True)
    processor_threads = [
        threading.Thread(target=processor, daemon=True) for _ in range(args.workers)
    ]
    writer_thread = threading.Thread(target=writer, daemon=True)

    reader_thread.start()
    for t in processor_threads:
        t.start()
    writer_thread.start()

    reader_thread.join()
    for t in processor_threads:
        t.join()
    writer_thread.join()

    if errors:
        raise errors[0]

    # Temporal mask smoothing
    if args.smooth > 0:
        files = sorted(os.listdir(processed_dir))
        total = len(files)
        window = args.smooth
        half = window // 2
        print(
            f"Applying temporal mask smoothing (window={window}, "
            f"workers={args.smooth_workers})...",
            flush=True,
        )

        smoothing_errors = []
        progress_lock = threading.Lock()
        progress_count = [0]
        n_workers = max(1, args.smooth_workers)

        # Write smoothed frames to a separate directory rather than
        # overwriting processed_dir in place. Overlapping windows mean a
        # frame can be a *read* dependency for several write_idx tasks;
        # writing in place risked one thread reading a file while another
        # was mid-save on it (truncated/corrupt PNG -> shape errors).
        # Reading on demand (no full-clip RAM cache) avoids the OOM/disk
        # blowup from caching every frame's alpha channel at once.
        smoothed_dir = processed_dir + "_smoothed"
        if not os.path.isdir(smoothed_dir):
            os.mkdir(smoothed_dir)

        def get_alpha(idx):
            file = files[idx]
            img = Image.open(os.path.join(processed_dir, file)).convert("RGBA")
            return np.array(img)[:, :, 3].astype(np.float32)

        def smooth_frame(write_idx):
            try:
                start = max(0, write_idx - half)
                end = min(total - 1, write_idx + half)
                alphas = np.stack([get_alpha(j) for j in range(start, end + 1)])
                smoothed_alpha = np.mean(alphas, axis=0).astype(np.uint8)

                filename = files[write_idx]
                out_img = Image.open(os.path.join(processed_dir, filename)).convert(
                    "RGBA"
                )
                out_arr = np.array(out_img)
                out_arr[:, :, 3] = smoothed_alpha
                Image.fromarray(out_arr).save(os.path.join(smoothed_dir, filename))

                with progress_lock:
                    progress_count[0] += 1
                    print(
                        f"Smoothed frame {progress_count[0]}/{total}: {filename}",
                        flush=True,
                    )
            except Exception as e:
                smoothing_errors.append(e)

        smooth_queue = Queue()
        for write_idx in range(total):
            smooth_queue.put(write_idx)

        def smoothing_worker():
            while True:
                try:
                    write_idx = smooth_queue.get_nowait()
                except Exception:
                    return
                if smoothing_errors:
                    return
                smooth_frame(write_idx)

        smoothing_threads = [
            threading.Thread(target=smoothing_worker, daemon=True)
            for _ in range(n_workers)
        ]
        for t in smoothing_threads:
            t.start()
        for t in smoothing_threads:
            t.join()

        if smoothing_errors:
            rmtree(smoothed_dir, ignore_errors=True)
            raise smoothing_errors[0]

        # Swap the smoothed frames in as the new processed_dir contents.
        rmtree(processed_dir)
        os.rename(smoothed_dir, processed_dir)

    # Output video
    output_file = pathlib.Path(args.o)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    stream = ffmpeg.input(
        os.path.join(processed_dir, "%04d.bmp"),
        r=framerate,
        f="image2",
        s=whstr,
        pix_fmt="yuva444p10le",
    )
    stream = ffmpeg.output(
        stream, args.o, vcodec="prores_ks", **{"profile:v": "4", "bits_per_mb": "5000"}
    )
    ffmpeg.run(stream)

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    print("Removing temporary files...")
    rmtree(processed_dir, ignore_errors=True)
    rmtree(frames_dir, ignore_errors=True)
