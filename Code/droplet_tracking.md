# Droplet Tracking Script

This `README.md` file is relevant only to the `droplet_tracking.py` file within this repository.

## Overview

`droplet_tracking.py` is a Python script designed to track droplets in a video file. The script allows users to manually select an area of interest in the video, within which droplets will be detected and tracked throughout the video's duration. The output is a video file with visualized tracking information, saved alongside the original video.

## Usage

1. **Set the video path:**
    - Update the `video_path` variable in the script to point to the video file you want to analyze.
    ```python
    video_path = 'path/to/your/video.mp4'
    ```

2. **Run the script:**
    - Execute the script in your Python environment.

3. **User input:**
    - The script will prompt you to enter the expected number of droplets in the video.
    - You'll need to manually select the area of interest in the video by clicking and dragging your mouse to create a circular area.

4. **Control the video playback:**
    - **Space bar:** Pause/Resume the video.
    - **Enter key:** Replay the video from the beginning.
    - **`r` key:** Reset the selection area.
    - Use the trackbar to navigate to a specific frame in the video.

5. **Output:**
    - The processed video with droplet tracking information will be saved in the same directory as the input video, with `_result` appended to the filename.

## Key Features

- **Manual Area Selection:** Select a circular area in the video where droplets are expected.
- **Droplet Tracking:** Tracks and identifies droplets within the selected area.
- **Video Output:** Generates a new video file showing the tracked droplets with their paths and IDs.

## Notes

- The script uses background subtraction to detect droplets, so it's optimized for videos where droplets contrast well against the background.
- The tracking algorithm uses the Hungarian method (linear sum assignment) to match detected droplets to previously tracked ones.

## Dependencies

- **OpenCV:** Used for video processing, background subtraction, and object detection.
- **NumPy:** Provides support for array operations.
- **SciPy:** Utilized for spatial calculations and optimization algorithms.
