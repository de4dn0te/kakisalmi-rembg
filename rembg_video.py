import argparse
import asyncio
import os
import ffmpeg
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from rembg.bg import remove
from rembg import new_session

# Parse args
parser = argparse.ArgumentParser(description='Applies rembg to the frames of a video')
parser.add_argument('input', type=str, help='Input video')
parser.add_argument('-o', type=str, default="export/output.mov", help="Define output path")
parser.add_argument('-a', action="store_true", help="Turns on alpha matting during background removal")
parser.add_argument('-af', type=int, default=240, help="Alpha matting foreground threshold")
parser.add_argument('-ab', type=int, default=10, help="Alpha matting background threshold")
parser.add_argument('-ae', type=int, default=10, help="Alpha matting erode size")
parser.add_argument('--skip-extract', action="store_true", help='Skips ffmpeg frame extraction')
parser.add_argument('--skip-process', action="store_true", help='Skips rembg frame processing')
parser.add_argument('--workers', type=int, default=4, help='Number of concurrent processing workers (default: 4)')
args = parser.parse_args()

# Extract video info
probe = ffmpeg.probe(args.input)
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
width = int(video_stream['width'])
height = int(video_stream['height'])
whstr = str(width) + 'x' + str(height)
framerate = video_stream['avg_frame_rate']

# Extract input video frames
if not args.skip_extract:
    frames_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "frames")
    if not os.path.isdir(frames_dir):
        os.mkdir(frames_dir)

    stream = ffmpeg.input(args.input)
    stream = ffmpeg.output(stream, os.path.join(frames_dir, "%04d.png"))
    ffmpeg.run(stream)

# Process frames with rembg (async + GPU)
try:
    if not args.skip_process:
        files_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "frames")
        processed_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "processed")
        if not os.path.isdir(processed_dir):
            os.mkdir(processed_dir)

        files = sorted(os.listdir(files_dir))
        total_files = len(files)

        # Load the model once and share across all workers
        print("Loading rembg GPU session...", flush=True)
        session = new_session("u2net_human_seg")  # Replace with what model is wanted

        def process_frame(args_tuple):
            idx, file, files_dir, processed_dir, rembg_args = args_tuple
            in_path = os.path.join(files_dir, file)
            out_path = os.path.join(processed_dir, file)
            print(f"Processing frame {idx}/{total_files}: {file}", flush=True)
            with open(in_path, "rb") as i:
                input_data = i.read()
                output_data = remove(
                    input_data,
                    session=session,
                    alpha_matting=rembg_args.a,
                    alpha_matting_foreground_threshold=rembg_args.af,
                    alpha_matting_background_threshold=rembg_args.ab,
                    alpha_matting_erode_size=rembg_args.ae,
                )
            with open(out_path, "wb") as o:
                o.write(output_data)
            print(f"Completed frame {idx}/{total_files}: {file}", flush=True)

        async def process_all_frames():
            loop = asyncio.get_running_loop()
            tasks_args = [
                (idx, file, files_dir, processed_dir, args)
                for idx, file in enumerate(files, 1)
            ]
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    loop.run_in_executor(executor, process_frame, task_args)
                    for task_args in tasks_args
                ]
                await asyncio.gather(*futures)

        asyncio.run(process_all_frames())
    
    # Output video
    output_file = pathlib.Path(args.o)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    stream = ffmpeg.input(os.path.join(processed_dir, "%04d.png"), r=framerate, f='image2', s=whstr, pix_fmt='yuva444p10le')
    stream = ffmpeg.output(stream, args.o, vcodec='prores_ks', **{'profile:v': '4', 'bits_per_mb': '5000'})
    ffmpeg.run(stream)

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    print("Removing temporary files...")
    shutil.rmtree(processed_dir)
    shutil.rmtree(frames_dir)
