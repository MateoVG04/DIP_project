import cv2

from numpy_array_to_video import NumpyArrayToVideo

array_to_video = NumpyArrayToVideo(cv2.VideoWriter_fourcc(*'XVID'), 17)
array_to_video.transform_all_arrays()
