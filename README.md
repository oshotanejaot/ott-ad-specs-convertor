## OTT Ad Specs Convertor

A local web tool for converting and compressing video (and reframing images) to match delivery specs required by OTT and streaming ad platforms - Prime Video, Netflix, Hulu, and similar.

## Features

- Convert codec (H.264, H.265, VP9), resolution, video bitrate, frame rate, and color space (SDR/HDR)
- Set audio codec (AAC, MP3, AC3), bitrate, sample rate, and channel layout (mono, stereo, 5.1)
- Trim clips to a specific start/end time
- Compress video by target quality (CRF) without changing other settings
- Content-aware image reframing (seam carving) for thumbnails/creative assets
- Job history and progress tracking, all processing runs locally via ffmpeg

## Requirements

- Python 3.9+
- ffmpeg and ffprobe installed and available on PATH

## Setup

pip install -r requirements.txt
python3 app.py

The app runs on http://localhost:5050 by default (set the PORT environment variable to change it).

## License

This project is proprietary. All rights reserved. You may view the source, but you may NOT copy, fork, redistribute, or host this project (or any derivative of it) on any website, server, or platform without written permission from the copyright holder. See the LICENSE file for full terms.
