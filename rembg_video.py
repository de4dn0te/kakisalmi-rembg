import argparse
import os
import ffmpeg
import pathlib
import queue
import shutil
import threading
import numpy as np
from PIL import Image
from rembg.bg import remove
from rembg import new_session

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
    default="birefnet-general-lite",
    help="rembg model to use (default: birefnet-general-lite)",
)
parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="Number of concurrent processing workers (default: 4)",
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
    "--skip-extract", action="store_true", help="Skips ffmpeg frame extraction"
)
parser.add_argument(
    "--skip-process", action="store_true", help="Skips rembg frame processing"
)
parser.add_argument(
    "--skip-smooth", action="store_true", help="Skips temporal mask smoothing"
)
args = parser.parse_args()

# Extract video info
probe = ffmpeg.probe(args.input)
video_stream = next(
    (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
)
width = int(video_stream["width"])
height = int(video_stream["height"])
whstr = str(width) + "x" + str(height)
framerate = video_stream["avg_frame_rate"]

frames_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "frames")
processed_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "processed")

# Extract input video frames
if not args.skip_extract:
    if not os.path.isdir(frames_dir):
        os.mkdir(frames_dir)
    stream = ffmpeg.input(args.input)
    stream = ffmpeg.output(stream, os.path.join(frames_dir, "%04d.png"))
    ffmpeg.run(stream)

_SENTINEL = object()

try:
    # Process frames with pipelined reader -> processors -> writer
    if not args.skip_process:
        if not os.path.isdir(processed_dir):
            os.mkdir(processed_dir)

        files = sorted(os.listdir(frames_dir))
        total_files = len(files)

        print(f"Loading rembg session (model={args.model})...", flush=True)
        session = new_session(args.model)

        read_queue = queue.Queue(maxsize=args.read_ahead)
        write_queue = queue.Queue(maxsize=args.write_buffer)
        errors = []
        active_processors = [
            args.workers
        ]  # list so processor() can mutate without nonlocal
        active_processors_lock = threading.Lock()

        def reader():
            try:
                for idx, file in enumerate(files, 1):
                    with open(os.path.join(frames_dir, file), "rb") as f:
                        data = f.read()
                    read_queue.put((idx, file, data))
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
                    output_data = remove(input_data, session=session)
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

    # Temporal mask smoothing — streaming sliding window, only `window` frames in RAM at once
    if not args.skip_smooth and args.smooth > 0:
        files = sorted(os.listdir(processed_dir))
        total = len(files)
        window = args.smooth
        half = window // 2
        print(f"Applying temporal mask smoothing (window={window})...", flush=True)

        buf = {}  # read_idx -> (filename, np.ndarray RGBA)

        for read_idx in range(total + half):
            # Load next frame into buffer
            if read_idx < total:
                file = files[read_idx]
                img = Image.open(os.path.join(processed_dir, file)).convert("RGBA")
                buf[read_idx] = (file, np.array(img))

            # The frame we can now finalize (has full right-side context)
            write_idx = read_idx - half
            if 0 <= write_idx < total:
                start = max(0, write_idx - half)
                end = min(total - 1, write_idx + half)
                alphas = np.stack(
                    [
                        buf[j][1][:, :, 3].astype(np.float32)
                        for j in range(start, end + 1)
                    ]
                )
                smoothed_alpha = np.mean(alphas, axis=0).astype(np.uint8)
                filename, arr = buf[write_idx]
                out_arr = arr.copy()
                out_arr[:, :, 3] = smoothed_alpha
                Image.fromarray(out_arr).save(os.path.join(processed_dir, filename))
                print(f"Smoothed frame {write_idx + 1}/{total}: {filename}", flush=True)

                # Drop the frame that's no longer needed by any future window
                drop_idx = write_idx - half
                if drop_idx in buf:
                    del buf[drop_idx]

    # Output video
    output_file = pathlib.Path(args.o)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    stream = ffmpeg.input(
        os.path.join(processed_dir, "%04d.png"),
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
    shutil.rmtree(processed_dir, ignore_errors=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
