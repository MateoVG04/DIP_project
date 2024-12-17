import numpy as np
import cv2
from PlayVideo import PlayVideo
import time

from object_tracking.object_tracking_basic import track_motion, select_bounding_box

playVideo = PlayVideo()


# Assuming the shape is (num_frames, height, width, channels)
#playVideo.videoPlayer()
frame = playVideo.video_data[0]
bounding_box = select_bounding_box(video=playVideo, frame=frame)
boxes_list = track_motion(video=playVideo,initial_bounding_box=bounding_box)
playVideo.displayPath(boxes_list)
#time.sleep(2)
#playVideo.playVideoDifference()

