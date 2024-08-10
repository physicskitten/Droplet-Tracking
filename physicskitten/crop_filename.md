# Crop_Filename.py README.md

## Overview

The script processes a video to detect a circular area of interest in the first frame. It then crops the video around this circle and applies a mask to highlight the detected area. The processed video is saved with a red circle drawn around the detected area. Additionally, the script provides a trackbar for frame navigation and displays the current playback time. This code was developed in attempt to pre-process and reduce external background noise from videos before running through "circle_detection.py" previously developed by TJ-coding.

## Usage

### 1. Set the video path:
- Update the `video_path` variable in the script to point to the video file you want to analyze.
  ```python
  video_path = 'E:/USB/Documents/VTestVid_9.mp4'

### 2. Run the script:
- Execute the script in your Python environment.

### 3. Interact with the video:
- The script will automatically start processing and displaying the video with the detected circle highlighted.
  - **Pause/Resume:** Press the Space bar to pause or resume the video playback.
  - **Trackbar Navigation:** Use the trackbar to navigate to specific frames in the video.

### 4. Output:
- The processed video will be saved in the same directory as the script with `Crop_` prefixed to the original video name.

## Key Features

- **Circle Detection:** Automatically detects a circular area of interest in the first frame of the video.
- **Video Cropping:** Crops the video around the detected circle and applies a circular mask.
- **Red Circle Overlay:** Draws a red circle around the detected area to highlight it in the output video.
- **Trackbar for Navigation:** Allows for easy navigation through the video frames using a trackbar.

## Notes

- The script uses the Hough Circle Transform method to detect circles in the video.
- The crop is adjusted to center around the black circle within a copper circle detected in the frame.
- A trackbar is provided for easy navigation through the video frames.

## How It Works

- **Circle Detection:** The script reads the first frame of the video and converts it to grayscale. It then applies a median blur and uses the Hough Circle Transform to detect circles.
  
- **Crop and Mask:** The detected circle's coordinates are used to crop the frame. A mask is created to focus on the detected circle, with the rest of the frame blacked out. A red circle is drawn around the detected area.
  
- **Video Playback:** The script plays the video with the processed frames. A trackbar is provided for easy navigation, and the user can pause or resume playback using the Space bar.
  
- **Output:** The processed frames are saved into a new video file with the same FPS as the original.

