import cv2
import os
import sys
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# Path to the video file
video_path = 'E:/USB/Documents/VTestVid_10.mp4'

# Extract the file name from the path and the current script name
video_name = os.path.basename(video_path)
script_name = os.path.basename(__file__)
window_title = f"{video_name} ({script_name})"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    cv2.imshow(window_title, np.zeros((1, 1, 3), dtype=np.uint8))  # Display an empty frame
    while True:
        if cv2.waitKey(25) != -1 or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    sys.exit()

# Get the total number of frames and FPS of the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration_seconds = total_frames / fps

# Prompt the user for the expected number of droplets
expected_droplets = int(input("Enter the expected number of droplets in the video: "))

# Function to detect the circle and calculate the crop coordinates
def get_crop_coordinates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=100, param2=30, minRadius=0, maxRadius=0
    )

    if circles is not None:
        circles = circles[0, :].astype("int")
        x, y, r = circles[0]
        black_circle_r = int(0.75 * r)
        side_length = 2 * black_circle_r
        start_x = max(0, x - black_circle_r)
        start_y = max(0, y - black_circle_r)
        end_x = start_x + side_length
        end_y = start_y + side_length

        height, width, _ = frame.shape
        if end_x > width:
            start_x = width - side_length
        if end_y > height:
            start_y = height - side_length

        return start_x, start_y, side_length
    return None

# Function to crop the frame using precomputed coordinates
def crop_frame(frame, crop_coords):
    start_x, start_y, side_length = crop_coords
    return frame[start_y:start_y + side_length, start_x:start_x + side_length]

# Background subtractor (using MOG2 method)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Read the first frame and determine the crop coordinates
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

crop_coords = get_crop_coordinates(first_frame)
if not crop_coords:
    print("Error: Could not detect the circle in the first frame.")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

# Apply the crop to the first frame
first_frame_cropped = crop_frame(first_frame, crop_coords)

# Function to handle trackbar movement
def on_trackbar_move(pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if ret:
        process_and_display_frame(frame, pos)

# Initialize trackers
object_id = 0
tracked_objects = {}  # {id: (x, y)}
paths = {}  # {id: [(x1, y1), (x2, y2), ...]}
colors = {}  # {id: (B, G, R)}

def generate_random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())

def process_and_display_frame(frame, pos):
    global object_id, tracked_objects, paths, colors

    frame_cropped = crop_frame(frame, crop_coords)
    fgmask = fgbg.apply(frame_cropped)
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
                detected_centroids.append((cx, cy))
                # x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(frame_cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame_cropped, (cx, cy), 3, (255, 0, 0), -1)

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
            
            tracked_objects = new_tracked_objects
        else:
            tracked_objects.clear()

    for centroid in detected_centroids:
        if len(tracked_objects) < expected_droplets:
            if not any(np.array_equal(centroid, tracked_objects[obj]) for obj in tracked_objects):
                if len(tracked_objects) < expected_droplets:
                    tracked_objects[object_id] = centroid
                    paths[object_id] = [centroid]
                    colors[object_id] = generate_random_color()
                    object_id += 1

    for obj_id, (cx, cy) in tracked_objects.items():
        cv2.putText(frame_cropped, f"{obj_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        paths[obj_id].append((cx, cy))
        for j in range(1, len(paths[obj_id])):
            cv2.line(frame_cropped, paths[obj_id][j-1], paths[obj_id][j], colors[obj_id], 2)

    current_time = pos / fps
    current_minutes = int(current_time // 60)
    current_seconds = int(current_time % 60)
    current_time_str = f"{current_minutes:02}:{current_seconds:02}"
    total_minutes = int(duration_seconds // 60)
    total_seconds = int(duration_seconds % 60)
    total_hours = int(total_minutes // 60)
    total_minutes %= 60
    total_duration_str = f"{total_hours:02}:{total_minutes:02}:{total_seconds:02}"

    cv2.imshow(window_title, frame_cropped)

# Create a trackbar for navigation
cv2.createTrackbar('Position', window_title, 0, total_frames - 1, on_trackbar_move)

paused = False
current_pos = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video.")
            break
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        process_and_display_frame(frame, current_pos)
        last_frame_cropped = crop_frame(frame, crop_coords)

    key = cv2.waitKey(25)
    if key == 32:  # Space bar to pause/resume
        paused = not paused
        if paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            cv2.imshow(window_title, last_frame_cropped)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    elif key == 13:  # Enter key to replay
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        object_id = 0
        tracked_objects = {}
        paths = {}
        colors = {}
        paused = False
    elif key != -1 or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
        break

if not ret:
    cv2.imshow(window_title, last_frame_cropped)
    print("Paused on the last frame. Press any key to exit.")
    while True:
        if cv2.waitKey(25) != -1 or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
