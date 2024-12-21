import numpy as np

from PlayVideo import PlayVideo
from deformation.calculate_deformation import CalculateDeformation

from object_tracking.object_tracking_basic import track_motion, select_bounding_box
from TrajectoryAnlysis import TrajectoryAnalysis
from object_tracking.test_tracked_objects import TEST_TRACKED_RECTS
from object_tracking.track_object import TrackRectangles
from trajectory_analysis.test_rect_data import DATA_LIST
import matplotlib.pyplot as plt

playVideo = PlayVideo()
frame = playVideo.video_data[0]

to_track_objects = [
    (193, 1193, 210, 203), # Left bucket
    (736, 1142, 57, 54), # Rotor attachment point
    (1180, 941, 67, 62), # Lever attachment point
    (1437, 517, 243, 240) # Right bucket
]
# tracked_objects = []
# for to_track in to_track_objects:
#     tracker = TrackRectangles(video=playVideo, initial_bounding_box=to_track)
#     tracked_objects.append(tracker.track_rectangle())
tracked_objects = TEST_TRACKED_RECTS  # Pre-computed

calc_deformation = CalculateDeformation(tracked_objects=tracked_objects)
playVideo.to_draw_rectangles = calc_deformation.objects_per_frame
playVideo.draw_lines = True

# Calculating deformation


# Create scatter plots


internal_lengths: list[list[int]] = []
for frame in calc_deformation.objects_per_frame:
    internal_length_for_frame = calc_deformation.calculate_internal_length(frame)
    internal_lengths.append(internal_length_for_frame)

colors = ["blue", "green", "red", "yellow"]
for color, internal_length in zip(colors, calc_deformation.per_frame_to_linear(internal_lengths)):
    # Cummulative Sum smoothens bumps caused by detecting jitter
    cumsum_vec = np.cumsum(np.insert(internal_length, 0, 0))
    window_width = 5
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    x = list(range(0, len(ma_vec)))
    plt.scatter(x, ma_vec, label=f"Line", color=f"{color}", alpha=0.7)


# Add title and labels
plt.title("Change in connection line lengths")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# playVideo.playVideoNormal()

"""
boxes_list is gecomment omdat we gebruik gaan maken van de data die we hebben gemeten en opgeslagen in
test_rect_data
#bounding_box = select_bounding_box(video=playVideo, frame=frame)
#boxes_list = track_motion(video=playVideo,initial_bounding_box=bounding_box)
"""
# trajectoryAnlysis.displayPath()
# plt.figure(figsize=(15,8))
# plt.subplot(211)
# trajectoryAnlysis.display_angle()
# plt.subplot(232)
# trajectoryAnlysis.display_speed()
# plt.show()

# playVideo.playVideoNormal()
# frame = playVideo.video_data[0]
# bounding_box = select_bounding_box(video=playVideo, frame=frame)
# boxes_list = DATA_LIST
# trajectoryAnlysis = TrajectoryAnalysis(playVideo=playVideo,boxes_list=boxes_list)
# trajectoryAnlysis.display_angle()


# Assuming the shape is (num_frames, height, width, channels)
#playVideo.videoPlayer()
#time.sleep(2)
#playVideo.playVideoDifference()

