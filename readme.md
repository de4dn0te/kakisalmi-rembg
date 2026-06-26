# rembg_for_video

Uses [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) and [rembg](https://github.com/danielgatis/rembg) to attempt removal of a background from a video file.

Based on project [rembg_from_video](https://github.com/seth-tribbey/rembg_from_video) by [seth-tribbey](https://github.com/seth-tribbey)

## Installation:

### For Newer Architecture (GeForce 1600+ Series)
```bash
python -m venv .venv
pip install -r requirements.txt
```

### For Pascal Arhitecture (GeForce 1000 Series)
```bash
python3.12 -m venv .venv
pip install -r requirements(3.12).txt
```
> **Script for fixing the cudnn path on Linux:** <br>
> export LD_LIBRARY_PATH=/path/to/kakisalmi/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

&nbsp;

## Usage:
```
python .\rembg_video.py [-h] [--help] [-o] [--model] [--workers] [--smooth] [--read-ahead] [--write-buffer] input
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
    <tr><td>-o</td><td>Output Path</td></tr>
    <tr><td>-h --help</td><td>Show Help</td></tr>
    <tr><td>--model</td><td>Choose model for rembg</td></tr>
  </table>

  <table>
    <tr><th colspan="2">Positional Arguments:</th></tr>
    <tr><td>input</td><td>Input Video</td></tr>
  </table>
</div>

optional arguments:
  -h, --help      show this help message and exit
  -o              Set output path (Default: export/output.mov)
  -
  
  Tip: [Alpha matting can be used to refine the results](https://github.com/danielgatis/rembg#advance-usage)