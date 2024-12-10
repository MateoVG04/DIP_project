import numpy as np
import cv2
from PlayVideo import PlayVideo
import time

playVideo = PlayVideo()

# Assuming the shape is (num_frames, height, width, channels)
#playVideo.playVideoNormal()
#time.sleep(2)
playVideo.playVideoDifference()

