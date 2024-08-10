import numpy as np
import cv2

import queue
import pickle
from pathlib import Path


def crop_image(frame, crop_size, center):
    """
    :param frame:
    :param crop_size: int must be odd
    :param center:
    :return:
    """
    # image_size must be odd
    if crop_size % 2 == 0:
        raise ValueError("crop_size must be odd")
    # Crop image
    x = center[1]
    y = center[0]
    frame_x = frame.shape[1]
    frame_y = frame.shape[0]
    top_left = [int(x - (crop_size - 1) / 2), int(y - (crop_size - 1) / 2)]
    bottom_right = [top_left[0] + crop_size, top_left[1] + crop_size]
    if top_left[0] >= 0 and top_left[1] >= 0 and bottom_right[0] < frame_x and bottom_right[1] < frame_y:
        cropped_frame = frame[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    else:
        # Check if cropping frame is out of bound
        top_left_x_out_of_bound = top_left[0] < 0
        top_left_y_out_of_bound = top_left[1] < 0
        bottom_right_x_out_of_bound = bottom_right[0] >= frame_x
        bottom_right_y_out_of_bound = bottom_right[1] >= frame_y
        top_left_padding = [0, 0]
        bottom_right_padding = [0, 0]
        # If out of bound, change crop limit
        if top_left_x_out_of_bound:
            top_left_padding[0] = -1*top_left[0]
            top_left[0] = 0
        if top_left_y_out_of_bound:
            top_left_padding[1] = -1*top_left[1]
            top_left[1] = 0
        if bottom_right_x_out_of_bound:
            bottom_right_padding[0] = bottom_right[0] - frame_x
            bottom_right[0] = frame_x
        if bottom_right_y_out_of_bound:
            bottom_right_padding[1] = bottom_right[1] - frame_y
            bottom_right[1] = frame_y
        cropped_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        # Pad frame, if out of bound
        if top_left_x_out_of_bound:
            padding = np.tile(0, (cropped_frame.shape[0], top_left_padding[0], 3))
            cropped_frame = np.hstack((padding, cropped_frame))
        if top_left_y_out_of_bound:
            padding = np.tile(0, (top_left_padding[1], cropped_frame.shape[1], 3))
            cropped_frame = np.vstack((padding, cropped_frame))
        if bottom_right_x_out_of_bound:
            padding = np.tile(0, (cropped_frame.shape[0], bottom_right_padding[0], 3))
            cropped_frame = np.hstack((cropped_frame, padding))
        if bottom_right_y_out_of_bound:
            padding = np.tile(0, (bottom_right_padding[1], cropped_frame.shape[1], 3))
            cropped_frame = np.vstack((cropped_frame, padding))

        if center[0] > frame_x or center[1] > frame_y:
            raise ValueError(f"center outside frame, center: {center}, frame: {frame.shape}")
        if cropped_frame.shape != (65, 65, 3):
            cv2.circle(frame, (center[0], center[1]), 1, (255, 0, 0), 2)
            cv2.circle(frame, (top_left[0], top_left[1]), 1, (0, 255, 0), 2)
            cv2.circle(frame, (bottom_right[0], bottom_right[1]), 1, (0, 0, 255), 2)
            cv2.imshow("frame", frame)
            cv2.waitKey(0)
            raise ValueError(f"cropped to wrong size: {cropped_frame.shape}")
    return cropped_frame

def generate_training_data_video(frame_queue, frame_id, training_data_seed, dir):
    # Crop the images then save in directory
    points = training_data_seed.points_in_frame[frame_id]
    Path(dir+"/video").mkdir(parents=True, exist_ok=True)
    for i in range(len(points["nodes"])):
        crop_size = 65

        file_name = f"fid{frame_id}_pid{points['particle_ids'][i]}.avi"
        video_writer = cv2.VideoWriter(f"{dir}/video/{file_name}",
                                       cv2.VideoWriter_fourcc(*"DIVX"),
                                       1, (crop_size, crop_size))
        for frame in frame_queue:
            cropped_frame = crop_image(frame, crop_size, points["nodes"][i])
            video_writer.write(cropped_frame.astype(np.uint8))
            """if file_name == "fid4_pid1.avi":
                cv2.imshow("frame", cropped_frame.astype(np.uint8))
                cv2.waitKey(0)"""
        video_writer.release()

def generate_training_data_image(frame_queue, frame_id, training_data_seed, dir):
    # Crop the images then save in directory
    points = training_data_seed.points_in_frame[frame_id]
    Path(dir+"/image").mkdir(parents=True, exist_ok=True)
    for i in range(len(points["nodes"])):
        crop_size = 65
        file_name = f"fid{frame_id}_pid{points['particle_ids'][i]}.jpg"
        cropped_image = crop_image(frame_queue[-1], crop_size, points["nodes"][i])
        success = cv2.imwrite(f"{dir}/image/{file_name}", cropped_image)
        if success is False:
            raise ValueError(f"Failed to save image: {dir}/image/{file_name}")


def generate_training_data(training_data_seed):
    file_name = training_data_seed.src_video
    vcap = cv2.VideoCapture(f"../video_source/{file_name}")
    read_success, frame = vcap.read()
    frame_queue = [frame]
    frame_id = 0
    dir = f"../training_data/{training_data_seed.name}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(training_data_seed, open(f"{dir}/training_data_seed.pickle", "wb"))
    while read_success:
        if len(frame_queue) == 5:
            """cv2.imshow("frame", frame)
            cv2.waitKey(0)"""
            generate_training_data_image(frame_queue, frame_id, training_data_seed, dir)
            generate_training_data_video(frame_queue, frame_id, training_data_seed, dir)
            frame_queue.pop(0)
        frame_queue.append(frame)
        frame_id += 1
        read_success, frame = vcap.read()
