import argparse
import os
import ffmpeg
import pathlib
import shutil
from rembg.bg import remove

#Parse args
parser = argparse.ArgumentParser(description='Applies rembg to the frames of a video')
parser.add_argument('input', type=str, help='Input video')
parser.add_argument('-o', type=str, default="export/output.mp4", help="Define output path")
parser.add_argument('-a', action="store_true", help="Turns on alpha matting during background removal")
parser.add_argument('-af', type=int, default=240, help="Alpha matting foreground threshold")
parser.add_argument('-ab', type=int, default=10, help="Alpha matting background threshold")
parser.add_argument('-ae', type=int, default=10, help="Alpha matting erode size")
parser.add_argument('--skip-extract', action="store_true", help='Skips ffmpeg frame extraction')
parser.add_argument('--skip-process', action="store_true", help='Skips rembg frame processing')
args = parser.parse_args()

#Extract video info
probe = ffmpeg.probe(args.input)
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
width = int(video_stream['width'])
height = int(video_stream['height'])
whstr = str(width) + 'x' + str(height)
framerate = video_stream['avg_frame_rate']

#Extract input video frames
if not args.skip_extract:
  frames_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "frames")
  if not os.path.isdir(frames_dir):
    os.mkdir(frames_dir)

  stream = ffmpeg.input(args.input)
  stream = ffmpeg.output(stream, os.path.join(frames_dir, "%04d.png"))
  ffmpeg.run(stream)

#Process frames with rembg
if not args.skip_process:
  files_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "frames")
  processed_dir = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "processed")
  if not os.path.isdir(processed_dir):
    os.mkdir(processed_dir)

  files = sorted(os.listdir(files_dir))
  total_files = len(files)
  for idx, file in enumerate(files, 1):
    print(f"Processing frame {idx}/{total_files}: {file}", flush=True)
    with open(os.path.join(files_dir, file), "rb") as i:
      with open(os.path.join(processed_dir, file), "wb") as o:
          input = i.read()
          output = remove(input, alpha_matting=args.a, alpha_matting_foreground_threshold=args.af, alpha_matting_background_threshold=args.ab, alpha_matting_erode_size=args.ae)
          o.write(output)
    print(f"Completed frame {idx}/{total_files}", flush=True)

#Output video
output_file = pathlib.Path(args.o)
output_file.parent.mkdir(exist_ok=True, parents=True)

stream = ffmpeg.input(os.path.join(processed_dir, "%04d.png"), r=framerate, f='image2', s=whstr, pix_fmt='yuv420p')
stream = ffmpeg.output(stream, args.o, vcodec='libx264', crf=25)
ffmpeg.run(stream)

#Cleanup
print("Removing temporary files...")
shutil.rmtree(processed_dir)
shutil.rmtree(frames_dir)