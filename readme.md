# rembg_for_video

Uses [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) and [rembg](https://github.com/danielgatis/rembg) to attempt removal of a background from a video file.

Based on project [rembg_from_video](https://github.com/seth-tribbey/rembg_from_video) by [seth-tribbey](https://github.com/seth-tribbey)

## Installation:

Currently i've gotten rembg to work only with Python **3.12** because of onnxruntime's shenanigans X/
```bash
python3.12 -m venv .venv
pip install -r requirements.txt
```
> **Script for fixing the cudnn path on Linux:** <br>
> export LD_LIBRARY_PATH=/path/to/kakisalmi/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

&nbsp;

## Usage:
```
python .\rembg_video.py [-h] [--help] [-o] [--model] [--workers] [--smooth] [--smooth-workers] [--buffer-size] [--output-type] input
```
<style>
table {
  border-collapse: separate;
  border-spacing: 0;
  border-radius: 5px;
  overflow: hidden;
}
th,
td {
  border: 1px solid #a0a0a0;
  padding: 8px 10px;
  border-radius: 8px;
  text-align: center;
}
td:first-child,
th:first-child {
  text-align: center;
}
td:nth-child(2),
th:nth-child(2) {
  text-align: left;
}
</style>
<div style="display: flex; gap: 20px; align-items: flex-start;">
  <table>
    <tr><th colspan="2">Optional Arguments:</th></tr>
    <tr><td>-o</td><td>Output path </td></tr>
    <tr><td>-h --help</td><td>Show help</td></tr>
    <tr><td>--model</td><td>Choose model for rembg</td></tr>
    <tr><td>--workers</td><td>Number of concurrent process workers</td></tr>
    <tr><td>--smooth</td><td>Size of window for temporal smoothing</td></tr>
    <tr><td>--smooth-workers</td><td>Number of CPU threads for temporal smoothing</td></tr>
    <tr><td>--buffer-size</td><td>Set size of processing buffer</td></tr>
    <tr><td>--output-type</td><td>Choose how video is exported 
    <br>"complete" = Full Color .mov <br>"mask" = Only mask <br>"mask_seq" = Mask image sequence)
  </table>

  <table>
    <tr><th colspan="2">Positional Arguments:</th></tr>
    <tr><td>input</td><td>Input Video</td></tr>
  </table>
</div>

## General Options:
    -h, --help                      Print this help text and exit
    --version                       Print program version and exit
    -U, --update                    Update this program to the latest version
    --no-update                     Do not check for updates (default)
    --update-to [CHANNEL]@[TAG]     Upgrade/downgrade to a specific version.