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
        cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

def get_circles(frame):
    blur_frame = cv2.GaussianBlur(frame, (11, 11), 50)
    gray_frame = cv2.cvtColor(blur_frame, 127, 255, cv2.COLOR_BGR2GRAY)
    # HoughCircles(Image, method, dp, minDist, param1: higher threshold of edge detection, param2: accumulator threshold )
    # 40
    dp = 2
    circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, dp, int(24/dp), param1=40, param2=11,
                               minRadius=int(14/dp), maxRadius=int(24/dp))

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

def assign_ids_to_detected(detected_circles, previous_index2ids, current_index_of_bubble, previous_index_of_bubble):
    """
    col_indecies and row_indecies rovide index for mapping circles to previous frames to next frame
    next_ids[row_index] -> previous_ids[col_indecix]
    """
    # initialize circle_index_to_id with -1
    not_paired_circle_index = set([i for i in range(len(detected_circles))])
    paired_circles = {}
    paired_index = set()
    # Assign ids to new circles using previous circles ids [Inherit Id from previously detected objects]
    for i in range(len(current_index_of_bubble)):
        index_to_map = current_index_of_bubble[i]
        index_to_map_to = previous_index_of_bubble[i]
        id = previous_index2ids[index_to_map_to]
        circle = detected_circles[index_to_map]
        paired_circles[id] = circle
        not_paired_circle_index.remove(i)
    # Assign new ids to circles that are left over [Assigning ids to newly detected objects]
    # only called when detected_circles_len > tracked_circles_len
    new_id = max(previous_index2ids) if len(previous_index2ids) !=0 else 0
    # circles that were given new ids [not tracked from previously known bubbles]
    newly_detected_circles = {}
    for i in list(not_paired_circle_index):
        newly_detected_circles[new_id] = detected_circles[i]
        new_id += 1

    return newly_detected_circles, paired_circles

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

def assign_tracking_circles(tracking_circles, tracking_kalman_filters, detecting_circles, bubble_ids, newly_detected):
    for i in range(len(detecting_circles)):
        id = bubble_ids[i]
        position_vector = [detecting_circles[i][0], detecting_circles[i][1]]
        if i in newly_detected["index"]:
            # Add newly detected circles as new circle
            tracking_kalman_filters[id] = kf.KalmanFilter(position_vector, 0.01, 0.01, 0.0005)
            tracking_circles[id] = detecting_circles[i]
        else:
            tracking_kalman_filters[id].update(position_vector)
            updated_position = tracking_kalman_filters[id].get_position()
            tracking_circles[id][0:2] = [updated_position[0], updated_position[1]]
    return tracking_circles, tracking_kalman_filters

class TrackingPoints:
    def __init__(self, max_age):
        # Contains position, and radius of circles
        self.circles = {}
        self.kalman_filters = {}
        self.age = {}
        self.age_since_last_detected = {}
        self.MAX_AGE = max_age

    def assign_new_points(self, new_circles):
        """
        :param new_points:  {id, [x,y,radius]}
        """
        for (id, circle) in new_circles.items():
            self.kalman_filters[id] = kf.KalmanFilter(circle[:2], 0.01, 1, 0.0005)
            self.circles[id] = [self.kalman_filters[id].X[0, 0], self.kalman_filters[id].X[0, 1], new_circles[id][2]]
            self.age[id] = 0
            self.age_since_last_detected[id] = 0

    def update_points(self, update_circles):
        """
        :param update_circles: {id, [x,y,radius]}
        :return:
        """
        for (id, circle) in update_circles.items():
            self.kalman_filters[id].update(circle[:2])
            self.circles[id] = self.kalman_filters[id].get_position() + [circle[2]]
            self.age_since_last_detected[id] = 0

    def predict_points(self):
        for id in self.kalman_filters:
            self.kalman_filters[id].predict()
            self.circles[id][:2] = self.kalman_filters[id].get_position()

    def remove_old_points(self):
        for id in list(self.circles):
            self.age[id] += 1
            self.age_since_last_detected[id] += 1
            if self.age_since_last_detected[id] > self.MAX_AGE:
                del self.circles[id]
                del self.kalman_filters[id]
                del self.age[id]
                del self.age_since_last_detected[id]

if __name__ == "__main__":
    file_name = "C1.mp4"
    vcap = cv2.VideoCapture(f"C:\\Users\\jinno\\OneDrive\\デスクトップ\\bscProject\\video_source\\{file_name}")
    TRAINING_DATA_PATH = "training_data/unlabeled"
    read_success, frame = vcap.read()
    tracking_points = None
    bubble_index2ids = {}
    SHOW_EDGE_FRAME = True
    frame_id = 0
    while read_success:
        # Normalize image
        cv2.normalize(frame, frame, 0, 255, norm_type=cv2.NORM_MINMAX)
        detecting_circles = get_circles(frame)
        if tracking_points is not None and detecting_circles is not None:
            tracking_points.predict_points()
            # tracking_circles, tracking_kalman_filters = predict_tracking_circles(tracking_circles, tracking_kalman_filters)
            previous_index_of_bubble, current_index_of_bubble = track_circles(tracking_points.circles, detecting_circles, min_cost=30)
            newly_detected_circles, paired_circles = assign_ids_to_detected(detecting_circles, list(tracking_points.circles.keys()), current_index_of_bubble, previous_index_of_bubble)
            #generate_training_data(frame, frame_id, detecting_circles, bubble_ids, 65, TRAINING_DATA_PATH)
        else:
            # First iteration
            newly_detected_circles, paired_circles = assign_ids_to_detected(detecting_circles, bubble_index2ids, [], [])
            if tracking_points is None:
                # Initialize tracking circles
                tracking_points = TrackingPoints(10)

        tracking_points.assign_new_points(newly_detected_circles)
        tracking_points.update_points(paired_circles)
        tracking_points.remove_old_points()

        print(f"time: {str(round(frame_id*1/30, 2))}" ) if frame_id % 30 == 0 else None


        """# Edge detection to see what hough circle is looking at for debugging
        blur_frame = cv2.GaussianBlur(frame, (11, 11), 25)
        gray_frame = cv2.cvtColor(blur_frame, 127, 255, cv2.COLOR_BGR2GRAY)
        #20
        edge_frame = cv2.Canny(gray_frame, 0, 40)"""

        # plot_circles(detecting_circles, bubble_ids)
        plot_circles(list(tracking_points.circles.values()), list(tracking_points.circles.keys()))
        plot_circles(list(newly_detected_circles.values())+list(paired_circles.values()), list(newly_detected_circles.keys())+list(paired_circles.keys()), color=(255, 0, 0))
        cv2.putText(frame, "time: "+str(round(frame_id*1/30, 2)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        """cv2.imshow("test", edge_frame)
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
            print(f"{file_name.split('.')[0]}_result_{i}.avi")
        else:
            video_writer.write(frame)
        frame_id += 1
        read_success, frame = vcap.read()