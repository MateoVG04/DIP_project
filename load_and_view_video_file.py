from PlayVideo import PlayVideo

from object_tracking.object_tracking_basic import track_motion, select_bounding_box
from TrajectoryAnlysis import TrajectoryAnalysis

playVideo = PlayVideo()
frame = playVideo.video_data[0]
bounding_box = select_bounding_box(video=playVideo, frame=frame)
boxes_list = track_motion(video=playVideo,initial_bounding_box=bounding_box)
trajectoryAnlysis = TrajectoryAnalysis(playVideo=playVideo,boxes_list=boxes_list)
trajectoryAnlysis.displayPath()


# Assuming the shape is (num_frames, height, width, channels)
#playVideo.videoPlayer()
#time.sleep(2)
#playVideo.playVideoDifference()

