import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import glob
import os

import kalman_filter as kf

def plot_circles(circles, bubble_id, color = (0, 0, 255)):
    # Convert values from float to integer
    circles = np.uint16(np.around(circles))
    # Circles: (x,y,radius,votes)
    for i in range(len(circles)):
        circle = circles[i]
        id = bubble_id[i]
        x = circle[0]
        y = circle[1]
        radius = circle[2]
        # draw the outer circle(frame, point, radius, color, thickness)
        # cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(frame, (x, y), 1, color, 2)
        cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

def get_circles(frame):
    blur_frame = cv2.GaussianBlur(frame, (11, 11), 50)
    gray_frame = cv2.cvtColor(blur_frame, 127, 255, cv2.COLOR_BGR2GRAY)
    # HoughCircles(Image, method, dp, minDist, param1: higher threshold of edge detection, param2: accumulator threshold )
    # 40
    circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 2, 150, param1=50, param2=18, minRadius=7, maxRadius=12)

    if circles is not None:
        if len(circles) > 0:
            circles = circles[0]
        else:
            circles = []
    else:
        circles = []
    return circles

def make_cost_matrix(tracking_circles, detecting_circles):
    if len(tracking_circles) != 0 and len(detecting_circles) != 0:
        # Extract circle centers
        tracking_pos = np.array([[[circle[0], circle[1]]] for circle in tracking_circles])
        detecting_pos = np.array([[[circle[0], circle[1]]] for circle in detecting_circles])
        # make nxm and mxn matrix
        tracking_matrix = np.repeat(tracking_pos, len(detecting_pos), axis=1)
        detecting_matrix = np.repeat(detecting_pos, len(tracking_matrix), axis=1)
        # Find the distance between centers
        distance_matrix = np.transpose(tracking_matrix, axes=(1, 0, 2)) - detecting_matrix
        distance_matrix = np.linalg.norm(distance_matrix, axis=2)
    else:
        distance_matrix = []
    return distance_matrix

def track_circles(tracking_circles, detecting_circles, min_cost = None):
    cost_matrix = make_cost_matrix(tracking_circles.values(), detecting_circles)
    if len(cost_matrix) > 0:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Do not match pair if greater than min_cost
        if not min_cost is None:
            cost_list = cost_matrix[row_ind, col_ind]
            paired_column = []
            paired_row = []
            for i in range(len(cost_list)):
                if cost_list[i] < min_cost:
                    paired_column.append(col_ind[i])
                    paired_row.append(row_ind[i])
            col_ind = paired_column
            row_ind = paired_row
    else:
        col_ind, row_ind = [], []
    return col_ind, row_ind

def assign_ids_to_detected(detected_circles_len, previous_ids, col_indecie, row_indecies):
    """
    col_indecies and row_indecies rovide index for mapping circles to previous frames to next frame
    next_ids[row_index] -> previous_ids[col_indecix]
    """
    # initialize next_ids
    next_ids = [-1 for i in range(detected_circles_len)]
    paired_ids = set() # Used to age untracked ids
    # Assign ids to new circles using previous circles ids [Inherit Id from previously detected objects]
    for i in range(len(col_indecie)):
        index_to_map_to = col_indecie[i]
        index_to_map = row_indecies[i]
        next_ids[index_to_map] = previous_ids[index_to_map_to]
        paired_ids.add(previous_ids[index_to_map_to])
    # Assign new ids to circles that are left over [Assigning ids to newly detected objects]
    # only called when detected_circles_len > tracked_circles_len
    new_id = max(previous_ids) if len(previous_ids) !=0 else 0
    # circles that were given new ids [not tracked from previously known bubbles]
    newly_detected = {"ids": [], "index": set()} #index in detecting_circles that were given a brand new id [not trackted]
    for i in range(len(next_ids)):
        if next_ids[i] == -1:
            next_ids[i] = new_id
            newly_detected["ids"].append(new_id)
            newly_detected["index"].add(i)
            new_id += 1

    return next_ids,  newly_detected, paired_ids

def generate_training_data(frame,frame_id, circles, bubble_ids, image_size, file_path):
    # image_size must be odd
    if image_size%2 == 0:
        raise ValueError("Image size must be odd")
    # File name to start from
    # existing_matches = glob.glob(f'{file_path}\\*.jpg')
    # Remove the file path and .jpg
    # existing_file_names = [int(match.split("\\")[-1][:-4]) for match in existing_matches]
    # next_file_name = max(existing_file_names) if len(existing_file_names) != 0 else 0
    for i in range(len(circles)):
        circle = circles[i]
        id = bubble_ids[i]
        # Crop image
        x = circle[0]
        y = circle[1]
        top_left = [int(x-(image_size-1)/2), int(y-(image_size-1)/2)]
        bottom_right = [top_left[0] + image_size, top_left[1] + image_size]
        if top_left[0] >= 0 and top_left[1] >= 0 and bottom_right[0] < frame.shape[0] and bottom_right[1] < frame.shape[1]:
            cropped_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            # Save file
            success = cv2.imwrite(f"{file_path}/{frame_id}-{id}.jpg", cropped_frame)
            if success is False:
                raise ValueError(f"{file_path}/{frame_id}-{id}.jpg")
            # next_file_name += 1

def predict_tracking_circles(tracking_circles, tracking_kalman_filters):
    for (key, value) in tracking_kalman_filters.items():
        value.predict()
        tracking_circles[key][0:2] = value.get_position()
    return tracking_circles, tracking_kalman_filters

def assign_tracking_circles(tracking_circles, tracking_kalman_filters, detecting_circles, bubble_ids, newly_detected):
    """
    for i in newly_detected_index:
        id = bubble_ids[i]
        tracking_circles[id] = detecting_circles[i]
    """
    for i in range(len(detecting_circles)):
        id = bubble_ids[i]
        position_vector = [detecting_circles[i][0], detecting_circles[i][1]]
        if i in newly_detected["index"]:
            # Add newly detected circles as new circle
            tracking_kalman_filters[id] = kf.KalmanFilter(position_vector, 0.01, 1, 0.0005)
            tracking_circles[id] = detecting_circles[i]
        else:
            tracking_kalman_filters[id].update(position_vector)
            updated_position = tracking_kalman_filters[id].get_position()
            tracking_circles[id][0:2] = [updated_position[0], updated_position[1]]
    return tracking_circles, tracking_kalman_filters

def kill_old_tracked_circles(tracking_circles, tracking_kalman_filters, tracking_age, paired_ids, newly_detected, max_age = 7):
    for id in newly_detected["ids"]:
        tracking_age[id] = 0
    for id in list(tracking_circles.keys()):
        if id in paired_ids:
            tracking_age[id] = 0
        else:
            tracking_age[id] += 1
            if tracking_age[id] > max_age:
                del tracking_circles[id]
                del tracking_kalman_filters[id]
                del tracking_age[id]
    return tracking_circles, tracking_kalman_filters, tracking_age
if __name__ == "__main__":
    # a = track_circles([[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]])
    # print(a)
    file_name = "Walker2.MOV"
    vcap = cv2.VideoCapture(f"C:\\Users\\jinno\\OneDrive\\デスクトップ\\bscProject\\video_source\\{file_name}")
    TRAINING_DATA_PATH = "training_data/unlabeled"
    read_success, frame = vcap.read()
    tracking_circles = None
    tracking_kalman_filters = None
    tracking_age = None
    bubble_ids = []
    SHOW_EDGE_FRAME = True
    frame_id = 0
    while read_success:
        # Normalize image
        cv2.normalize(frame, frame, 0, 255, norm_type=cv2.NORM_MINMAX)
        detecting_circles = get_circles(frame)
        if not tracking_circles is None and tracking_kalman_filters is not None and tracking_age is not None and not detecting_circles is None:
            predict_tracking_circles(tracking_circles, tracking_kalman_filters)
            # tracking_circles, tracking_kalman_filters = predict_tracking_circles(tracking_circles, tracking_kalman_filters)
            cols, rows = track_circles(tracking_circles, detecting_circles, min_cost=10)
            bubble_ids, newly_detected, paired_ids = assign_ids_to_detected(len(detecting_circles), list(tracking_circles.keys()), cols, rows)
            #generate_training_data(frame, frame_id, detecting_circles, bubble_ids, 65, TRAINING_DATA_PATH)
        else:
            # First iteration
            bubble_ids, newly_detected, paired_ids = assign_ids_to_detected(len(detecting_circles), bubble_ids, [], [])
            if tracking_circles is None:
                # Initialize tracking circles
                tracking_circles = {}
                tracking_kalman_filters = {}
                tracking_age = {}

        tracking_circles, tracking_kalman_filters = assign_tracking_circles(tracking_circles, tracking_kalman_filters,
                                                                            detecting_circles, bubble_ids, newly_detected)
        tracking_circles, tracking_kalman_filters, tracking_age = kill_old_tracked_circles(tracking_circles,
                                                                                           tracking_kalman_filters,
                                                                                           tracking_age, paired_ids,
                                                                                           newly_detected)
        print(f"time: {str(round(frame_id*1/30, 2))}" ) if frame_id % 30 == 0 else None


        # Edge detection to see what hough circle is looking at for debugging
        blur_frame = cv2.GaussianBlur(frame, (11, 11), 25)
        gray_frame = cv2.cvtColor(blur_frame, 127, 255, cv2.COLOR_BGR2GRAY)
        #20
        edge_frame = cv2.Canny(gray_frame, 0, 50)

        # plot_circles(detecting_circles, bubble_ids)
        plot_circles(list(tracking_circles.values()), list(tracking_circles.keys()))
        plot_circles(detecting_circles, list(bubble_ids), color=(255, 0, 0))
        cv2.putText(frame, "time: "+str(round(frame_id*1/30, 2)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        """cv2.imshow("test", frame)
        cv2.waitKey(0)"""

        # Save frames:
        if frame_id == 0:
            i = 0
            while os.path.exists(f'video_log/{file_name.split(".")[0]}_result_{i}.avi'):
                i += 1
            # This part is OS dependent
            video_writer = cv2.VideoWriter(f'video_log/{file_name.split(".")[0]}_result_{i}.avi',
                                           cv2.VideoWriter_fourcc(*"DIVX"),
                                           30, (frame.shape[1], frame.shape[0]))
        else:
            video_writer.write(frame)
        frame_id += 1
        read_success, frame = vcap.read()