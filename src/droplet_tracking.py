import cv2
import os
import sys
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import time

# Path to the video file
video_path = 'E:/USB/Documents/VTestVid_13.mp4'

# Extract the file name and directory
video_dir = os.path.dirname(video_path)
video_name = os.path.basename(video_path)
video_base, video_ext = os.path.splitext(video_name)

# Output video path (changed to include "_result")
output_video_path = os.path.join(video_dir, f"{video_base}_result{video_ext}")

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    cv2.imshow(video_name, np.zeros((1, 1, 3), dtype=np.uint8))  # Display an empty frame
    while True:
        if cv2.waitKey(25) != -1 or cv2.getWindowProperty(video_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    sys.exit()

# Get the total number of frames and FPS of the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration_seconds = total_frames / fps

# Prompt the user for the expected number of droplets
expected_droplets = int(input("Enter the expected number of droplets in the video: "))

# Variables for circular area selection
circle_center = None
circle_radius = 0
selecting_area = False
frame = None
last_frame_cropped = None  # Initialize last_frame_cropped

# Initialize tracking time dictionary
tracked_droplet_times = {}

# Mouse callback function for selecting and resizing the tracking area
def select_area(event, x, y, flags, param):
    global circle_center, circle_radius, selecting_area, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        circle_center = (x, y)
        selecting_area = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_area and circle_center:
            circle_radius = int(np.sqrt((x - circle_center[0])**2 + (y - circle_center[1])**2))
    elif event == cv2.EVENT_LBUTTONUP:
        selecting_area = False

    # Draw the circle on the frame
    if frame is not None:
        display_frame = frame.copy()
        if circle_center:
            cv2.circle(display_frame, circle_center, circle_radius, (0, 255, 0), 2)
        cv2.imshow(window_title, display_frame)

# Create a window with the specified title
window_title = f"{video_name}"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_title, select_area)

# Function to mask a frame based on the circular area
def apply_circular_mask(frame, center, radius):
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

# Function to process frame based on the selected area
def process_frame(frame):
    global circle_center, circle_radius
    global tracked_objects, object_id, paths, colors, tracked_droplet_times  # Reference global variables

    if circle_center and circle_radius > 0:
        # Apply the circular mask to the full frame
        masked_frame = apply_circular_mask(frame, circle_center, circle_radius)
        # Convert to grayscale for background subtraction
        gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray_frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_centroids = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Check if the centroid is inside the circle
                    if np.linalg.norm(np.array([cx, cy]) - np.array(circle_center)) <= circle_radius:
                        detected_centroids.append((cx, cy))
                        cv2.circle(masked_frame, (cx, cy), 3, (255, 0, 0), -1)

        if tracked_objects:
            object_ids = list(tracked_objects.keys())
            object_centroids = np.array(list(tracked_objects.values()))

            if detected_centroids:
                detected_centroids = np.array(detected_centroids)
                distances = distance.cdist(object_centroids, detected_centroids)
                rows, cols = linear_sum_assignment(distances)

                new_tracked_objects = {}
                for row, col in zip(rows, cols):
                    new_tracked_objects[object_ids[row]] = tuple(detected_centroids[col])
                    # Reset tracking time for matched droplets
                    tracked_droplet_times[object_ids[row]] = time.time()
                
                tracked_objects = new_tracked_objects
            else:
                # Update tracking duration and remove droplets if not tracked for more than 3 seconds
                to_remove = []
                for obj_id, track_start_time in tracked_droplet_times.items():
                    if time.time() - track_start_time > 3:
                        to_remove.append(obj_id)
                
                for obj_id in to_remove:
                    tracked_droplet_times.pop(obj_id, None)
                    tracked_objects.pop(obj_id, None)

        for centroid in detected_centroids:
            if len(tracked_objects) < expected_droplets:
                if not any(np.array_equal(centroid, tracked_objects[obj]) for obj in tracked_objects):
                    if len(tracked_objects) < expected_droplets:
                        tracked_objects[object_id] = centroid
                        paths[object_id] = [centroid]
                        colors[object_id] = generate_random_color()
                        tracked_droplet_times[object_id] = time.time()  # Start tracking time for new droplets
                        object_id += 1

        for obj_id, (cx, cy) in tracked_objects.items():
            cv2.putText(masked_frame, f"{obj_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            paths[obj_id].append((cx, cy))
            for j in range(1, len(paths[obj_id])):
                cv2.line(masked_frame, paths[obj_id][j-1], paths[obj_id][j], colors[obj_id], 2)

        return masked_frame
    return frame

# Background subtractor (using MOG2 method)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Initialize trackers
object_id = 0
tracked_objects = {}  # {id: (x, y)}
paths = {}  # {id: [(x1, y1), (x2, y2), ...]}
colors = {}  # {id: (B, G, R)}

def generate_random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

def process_and_display_frame(frame, pos):
    global object_id, tracked_objects, paths, colors, last_frame_cropped

    frame_processed = process_frame(frame)
    current_time = pos / fps
    current_minutes = int(current_time // 60)
    current_seconds = int(current_time % 60)
    current_time_str = f"{current_minutes:02}:{current_seconds:02}"
    total_minutes = int(duration_seconds // 60)
    total_seconds = int(duration_seconds % 60)
    total_hours = int(total_minutes // 60)
    total_minutes %= 60
    total_duration_str = f"{total_hours:02}:{total_minutes:02}:{total_seconds:02}"

    cv2.imshow(window_title, frame_processed)
    last_frame_cropped = frame_processed  # Update last_frame_cropped with the current frame

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'MJPG' for .avi files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Create a trackbar for navigation
cv2.createTrackbar('Position', window_title, 0, total_frames - 1, lambda pos: cap.set(cv2.CAP_PROP_POS_FRAMES, pos))

paused = True  # Start with the video paused
current_pos = 0

# Initially display the first frame
ret, frame = cap.read()
if ret:
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    process_and_display_frame(frame, current_pos)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video.")
            break
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        process_and_display_frame(frame, current_pos)
        if circle_center and circle_radius > 0:
            last_frame_cropped = process_frame(frame)

        # Write the processed frame to the output video file
        if frame is not None:
            processed_frame = process_frame(frame)
            out.write(processed_frame)

    key = cv2.waitKey(25)
    if key == 32:  # Space bar to pause/resume
        paused = not paused
        if paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            if last_frame_cropped is not None:
                cv2.imshow(window_title, last_frame_cropped)
            else:
                cv2.imshow(window_title, frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    elif key == 13:  # Enter key to replay
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        object_id = 0
        tracked_objects = {}
        paths = {}
        colors = {}
        tracked_droplet_times = {}  # Reset tracking times
        paused = False
    elif key == ord('r'):  # 'r' key to reset selection
        circle_center = None
        circle_radius = 0
    elif key != -1 or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
        break

if last_frame_cropped is not None:
    cv2.imshow(window_title, last_frame_cropped)
    print("Paused on the last frame. Press any key to exit.")
    while True:
        if cv2.waitKey(25) != -1 or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
