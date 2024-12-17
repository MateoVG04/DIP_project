from PlayVideo import PlayVideo

from object_tracking.object_tracking_basic import track_motion, select_bounding_box
from TrajectoryAnlysis import TrajectoryAnalysis
from trajectory_analysis.test_rect_data import DATA_LIST

playVideo = PlayVideo()
frame = playVideo.video_data[0]

"""
boxes_list is gecomment omdat we gebruik gaan maken van de data die we hebben gemeten en opgeslagen in
test_rect_data
#bounding_box = select_bounding_box(video=playVideo, frame=frame)
#boxes_list = track_motion(video=playVideo,initial_bounding_box=bounding_box)
"""
boxes_list = DATA_LIST
trajectoryAnlysis = TrajectoryAnalysis(playVideo=playVideo,boxes_list=boxes_list)
trajectoryAnlysis.display_angle()


# Assuming the shape is (num_frames, height, width, channels)
#playVideo.videoPlayer()
#time.sleep(2)
#playVideo.playVideoDifference()

