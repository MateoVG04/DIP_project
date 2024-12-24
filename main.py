import sys

import numpy as np
from matplotlib import pyplot as plt

from deformation.calculate_deformation import CalculateDeformation
from humoment.hu_moment import HuMoment
from object_tracking.circle_tracker import CircleTracker
from object_tracking.object_tracking_basic import select_bounding_box, track_motion
from object_tracking.test_tracked_objects import TEST_TRACKED_RECTS
from playvideo.play_video import PlayVideo
from trajectory_analysis.test_rect_data import DATA_LIST
from trajectory_analysis.trajectory_analysis import TrajectoryAnalysis


def deformation():
    playVideo = PlayVideo()
    tracked_objects = TEST_TRACKED_RECTS  # Pre-computed

    # Calculating deformation
    calc_deformation = CalculateDeformation(tracked_objects=tracked_objects)
    playVideo.to_draw_rectangles = calc_deformation.objects_per_frame
    playVideo.draw_lines = True

    # Create scatter plots
    internal_lengths: list[list[int]] = []
    for frame in calc_deformation.objects_per_frame:
        internal_length_for_frame = calc_deformation.calculate_internal_length(frame)
        internal_lengths.append(internal_length_for_frame)

    colors = ["blue", "green", "red", "yellow"]
    labels = ["Left cup to left circle", "Left circle to right circle", "Right circle to right cup",
              "Left cup to right cup"]
    for color, label, internal_length in zip(colors, labels, calc_deformation.per_frame_to_linear(internal_lengths)):
        # Cumulative Sum smoothens bumps caused by detecting jitter
        cumsum_vec = np.cumsum(np.insert(internal_length, 0, 0))
        window_width = 5
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

        x = list(range(0, len(ma_vec)))
        plt.scatter(x, ma_vec, label=f"{label}", color=f"{color}", alpha=0.7)

    # Add title and labels
    plt.title("Change in connection line lengths")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Add legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


def hu_moment_pre_created_data(file):
    hu_moment = HuMoment()
    hu_moment.play_video("Data/" + str(file) + ".avi")


def hu_moment_own_data():
    hu_moment = HuMoment()
    hu_moment.process_frames_and_save("Data/Ballenwerper_sync_380fps_006.npy", "Data/Humoment_own_data.avi",
                                      "Data/ReferenceImage.png")
    hu_moment.play_video("Data/Humoment_own_data.avi")


def object_tracking_pre_created_data():
    playVideo = PlayVideo()
    boxes_list = DATA_LIST
    trajectoryAnalysis = TrajectoryAnalysis(playVideo=playVideo, boxes_list=boxes_list)
    trajectoryAnalysis.display_angle()
    trajectoryAnalysis.display_speed()
    trajectoryAnalysis.show_vibrations()


def object_tracking_own_data():
    playVideo = PlayVideo()
    frame = playVideo.video_data[0]
    bounding_box = select_bounding_box(video=playVideo, frame=frame)
    boxes_list = track_motion(video=playVideo, initial_bounding_box=bounding_box)
    trajectoryAnalysis = TrajectoryAnalysis(playVideo=playVideo, boxes_list=boxes_list)
    trajectoryAnalysis.display_angle()
    trajectoryAnalysis.display_speed()
    trajectoryAnalysis.show_vibrations()


def circle_tracker():
    video_data = np.lib.format.open_memmap('Data/Ballenwerper_sync_380fps_006.npy', mode='r+')
    circle_tracker = CircleTracker(video_data)
    circle_tracker.playVideoNormal()


def play_frame_difference():
    playVideo = PlayVideo()
    playVideo.playVideoDifference()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "deformation":
            deformation()
        elif sys.argv[1] == "hu_moment_pre_created_data":
            if len(sys.argv) > 2:
                if sys.argv[2] == "LeftCup":
                    hu_moment_pre_created_data("LeftCup")
                elif sys.argv[2] == "RightCup":
                    hu_moment_pre_created_data("RightCup")
            else:
                print("No arguments provided")
        elif sys.argv[1] == "hu_moment_own_data":
            hu_moment_own_data()
        elif sys.argv[1] == "object_tracking_pre_created_data":
            object_tracking_pre_created_data()
        elif sys.argv[1] == "object_tracking_own_data":
            object_tracking_own_data()
        elif sys.argv[1] == "circle_tracking":
            circle_tracker()
        elif sys.argv[1] == "play_frame_difference":
            play_frame_difference()
    else:
        print("Invalid command.")
