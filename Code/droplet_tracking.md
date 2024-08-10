# Droplet_Tracking.py README.md

## Overview

`droplet_tracking.py` is a Python script that tracks droplets within a video, allowing manual selection of an area of interest. The output is a new video with tracking information overlaid, saved in the same directory as the original video.

## Usage

### Set the Video Path

Modify the `video_path` variable in the script to the path of your video file.

```python
video_path = 'path/to/your/video.mp4'```

# Run the Script
Execute the script in your Python environment.

## User Input
- Enter the expected number of droplets.
- Select the area of interest by clicking and dragging to create a circular region.

## Playback Controls
- **Space bar**: Pause/Resume playback.
- **Enter key**: Restart the video.
- **r key**: Reset the selection area.
- **Trackbar**: Navigate through frames.

## Output
The script saves a processed video with `_result` appended to the original filename, showing tracked droplets.

## Key Features
- **Manual Area Selection**: Define a circular region where droplets are tracked.
- **Droplet Tracking**: Detects and tracks droplets within the selected area.
- **Video Output**: Outputs a new video file with visualized tracking paths.
- **Background Subtraction**: Optimized for videos where droplets contrast with the background.
- **Hungarian Algorithm**: Uses the Hungarian method for matching detected droplets to tracked ones.

## How the Script Works

1. **Initialization**
   - Imports required libraries (`cv2`, `os`, `sys`, `numpy`, `scipy`) and sets the video path.
   - Creates a `VideoCapture` object to load the video and checks for successful opening.

2. **User Input & Video Properties**
   - Prompts for the expected number of droplets.
   - Retrieves the video’s total frames and FPS to manage playback.

3. **Region of Interest (ROI) Selection**
   - Allows the user to define a circular ROI by clicking and dragging.
   - The selected area is dynamically visualized as the user adjusts the circle.

4. **Frame Processing**
   - **Circular Masking**: Applies a mask to the frame, isolating the selected ROI.
   - **Background Subtraction**: Converts the masked frame to grayscale and applies a background subtractor to detect moving droplets.
   - **Contour Detection**: Detects contours in the foreground mask, filtering centroids to those within the ROI.

5. **Droplet Tracking**
   - Maintains a dictionary of tracked objects and their centroids.
   - **Centroid Assignment**: Matches detected centroids to tracked droplets using the Hungarian algorithm.
   - **Path Visualization**: Stores and displays each droplet's path with distinct colors.

6. **Frame Display & Navigation**
   - Displays processed frames with overlaid tracking data.
   - Allows control over playback and ROI selection via user inputs and trackbar.

7. **Video Output**
   - Saves the processed video with the suffix `_result`, retaining the original frame rate and resolution.

8. **Cleanup**
   - Releases resources and closes OpenCV windows upon completion.

## Key Functions

- `select_area(event, x, y, flags, param)`: Handles mouse events for defining and resizing the ROI.
- `apply_circular_mask(frame, center, radius)`: Applies a circular mask to isolate the ROI.
- `process_frame(frame)`: Detects and tracks droplets, updating positions and paths.
- `generate_random_color()`: Generates distinct colors for tracking paths.
- `process_and_display_frame(frame, pos)`: Manages frame processing and visualization.

## Dependencies
- **OpenCV**: For video processing, background subtraction, and object detection.
- **NumPy**: For numerical array operations.
- **SciPy**: For spatial calculations and optimization.
