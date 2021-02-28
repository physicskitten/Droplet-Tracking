import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import glob
import os

import kalman_filter as kf
import training_data_generator as tg

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
        cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color)

def get_circles(frame):
    blur_frame = cv2.GaussianBlur(frame, (11, 11), 50)
    gray_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2GRAY)
    if gray_frame.shape != frame.shape and frame.shape != blur_frame.shape:
        raise ValueError("Shape has been altered during filter")
    # HoughCircles(Image, method, dp, minDist, param1: higher threshold of edge detection, param2: accumulator threshold )
    circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, Parameters.dp, Parameters.min_dist,
                               param1=Parameters.edge_detection_upper_threshold, param2=Parameters.accumulator_threshold,
                               minRadius=Parameters.min_rad, maxRadius=Parameters.max_rad)

    if circles is not None:
        if len(circles) > 0:
            circles = circles[0]
        else:
            circles = []
    else:
        circles = []
    return circles

def make_cost_matrix(tracking_circles, detecting_circles, tracking_age):
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
        # Add age weight
        # wv = 1/(age+1), younger points get higher cost [less likely to be picked]
        age_weight_vector = Parameters.max_weight_due_age * 1/((np.array(tracking_age)+1)/Parameters.age_for_weight_to_be_half)
        age_weight_matrix = np.tile(np.transpose(age_weight_vector), (len(detecting_circles), 1))
        cost_matrix = distance_matrix + age_weight_matrix
    else:
        cost_matrix = []
    return cost_matrix

def track_circles(tracking_circles, detecting_circles, tracking_age, min_cost = None):
    cost_matrix = make_cost_matrix(tracking_circles, detecting_circles, tracking_age)
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
    # Assign ids to new circles using previous circles ids [Inherit Id from previously detected objects]
    for i in range(len(current_index_of_bubble)):
        index_to_map = current_index_of_bubble[i]
        index_to_map_to = previous_index_of_bubble[i]
        id = previous_index2ids[index_to_map_to]
        circle = detected_circles[index_to_map]
        paired_circles[id] = circle
        not_paired_circle_index.remove(current_index_of_bubble[i])
    # Assign new ids to circles that are left over [Assigning ids to newly detected objects]
    # only called when detected_circles_len > tracked_circles_len
    new_id = max(previous_index2ids) if len(previous_index2ids) !=0 else 0
    # circles that were given new ids [not tracked from previously known bubbles]
    newly_detected_circles = {}
    for i in list(not_paired_circle_index):
        newly_detected_circles[new_id] = detected_circles[i]
        new_id += 1

    return newly_detected_circles, paired_circles

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
            self.kalman_filters[id] = kf.KalmanFilter(circle[:2], Parameters.dt, Parameters.std_model,
                                                      Parameters.std_measurement)
            self.circles[id] = self.kalman_filters[id].get_position() + [new_circles[id][2]]
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


class Parameters:
    # Kalman filter
    dt = 1/30
    std_model = 256
    std_measurement = 0.5
    # Preventing identity switch
    max_age = 15
    min_cost = 60
    max_weight_due_age = 10
    age_for_weight_to_be_half = 5
    # Detecting circles
    dp = 2
    min_dist = int(24 / dp)
    edge_detection_upper_threshold = 30
    # edge_detection_upper_threshold = 40
    accumulator_threshold = 11
    min_rad = int(14 / dp)
    max_rad = int(24 / dp)


class TrainingDataSeed:
    def __init__(self, name, src_video, output_video,true_ids):
        """
        :param name: (str) name of this seed
        :param src_video: (str) name of src video
        :param true_ids: [int] ids of true positives"""
        self.name = name
        self.src_video = src_video
        self.output_video = output_video
        self.true_ids = set(true_ids)
        self.points_in_frame = {} # {"frame_id": {"particle_ids": [int],"nodes": [[int]]}

    def add_points(self, frame_id, particle_ids, nodes):
        if frame_id not in self.points_in_frame:
            self.points_in_frame[frame_id] = {"particle_ids": [], "nodes": []}
        self.points_in_frame[frame_id]["particle_ids"] += particle_ids
        """for node in nodes:
            if node[0] > 490 or node[1] > 490:
                plot_circles([[200, 200, 1]], [999], color=(0, 200, 0))
                cv2.imshow("test", frame)
                cv2.waitKey(0)
                raise ValueError(f"node outside frame: {node}")"""
        self.points_in_frame[frame_id]["nodes"] += [node[0:2] for node in nodes]

def find_output_file_name(src_file_name):
    i = 0
    output_file_name = f'{src_file_name.split(".")[0]}_result_{i}.avi'
    while os.path.exists(f'../video_log/{output_file_name}'):
        i += 1
        output_file_name = f'{src_file_name.split(".")[0]}_result_{i}.avi'
    return output_file_name

if __name__ == "__main__":
    file_name = "C1a.mp4"
    output_file_name = find_output_file_name(file_name)
    vcap = cv2.VideoCapture(f"..\\video_source\\{file_name}")
    read_success, frame = vcap.read()
    tracking_points = None
    bubble_index2ids = {}
    frame_id = 0
    training_data_seed = TrainingDataSeed("C1a", file_name, output_file_name, [20])
    # This part is OS dependent
    video_writer = cv2.VideoWriter(f'../video_log/{output_file_name}',
                                   cv2.VideoWriter_fourcc(*"DIVX"),
                                   1 / Parameters.dt, (frame.shape[1], frame.shape[0]))
    while read_success:
        # Normalize image
        cv2.normalize(frame, frame, 0, 255, norm_type=cv2.NORM_MINMAX)
        detecting_circles = get_circles(frame)

        if tracking_points is not None and detecting_circles is not None:
            tracking_points.predict_points()
            previous_index_of_bubble, current_index_of_bubble = track_circles(list(tracking_points.circles.values()),
                                                                              detecting_circles,
                                                                              list(tracking_points.age.values()),
                                                                              min_cost=Parameters.min_cost)
            newly_detected_circles, paired_circles = assign_ids_to_detected(detecting_circles, list(tracking_points.circles.keys()), current_index_of_bubble, previous_index_of_bubble)
        else:
            # First iteration
            newly_detected_circles, paired_circles = assign_ids_to_detected(detecting_circles, bubble_index2ids, [], [])
            if tracking_points is None:
                # Initialize tracking circles
                tracking_points = TrackingPoints(Parameters.max_age)

        tracking_points.assign_new_points(newly_detected_circles)
        tracking_points.update_points(paired_circles)
        tracking_points.remove_old_points()

        print(f"time: {str(round(frame_id*1/30, 2))}") if frame_id % 30 == 0 else None


        """# Edge detection to see what hough circle is looking at for debugging
        blur_frame = cv2.GaussianBlur(frame, (11, 11), 25)
        gray_frame = cv2.cvtColor(blur_frame, 127, 255, cv2.COLOR_BGR2GRAY)
        #20
        edge_frame = cv2.Canny(gray_frame, 0, 40)"""

        # plot_circles(detecting_circles, bubble_ids)
        plot_circles(list(tracking_points.circles.values()), list(tracking_points.circles.keys()))
        # plot_circles(list(newly_detected_circles.values())+list(paired_circles.values()), list(newly_detected_circles.keys())+list(paired_circles.keys()), color=(255, 0, 0))
        plot_circles(list(newly_detected_circles.values()), list(newly_detected_circles.keys()), color=(255, 0, 0))
        plot_circles(list(paired_circles.values()), list(paired_circles.keys()), color=(200, 100, 0))

        cv2.putText(frame, "time: "+str(round(frame_id*Parameters.dt, 2)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        """cv2.imshow("test", edge_frame)
        cv2.waitKey(0)"""

        # Add detected points to training_data
        training_data_seed.add_points(frame_id,
                                        list(newly_detected_circles.keys()) + list(paired_circles.keys()),
                                        list(newly_detected_circles.values())+list(paired_circles.values()))

        # Save frames:
        video_writer.write(frame)
        frame_id += 1
        read_success, frame = vcap.read()
    video_writer.release()
    tg.generate_training_data(training_data_seed)