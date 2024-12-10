import os

import cv2
import numpy as np


class NumpyArrayToVideo():
    def __init__(self, fourcc):
        self.fourcc = fourcc

    def transform(self, input_file, output_file):
        frames = np.load(input_file)
        num_frames, height, width = frames.shape[:3]
        out = cv2.VideoWriter(output_file, self.fourcc, self.fps, (width, height), False)

        for i in range(num_frames):
            frame = frames[i]
            out.write(frame)

        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved as {output_file}")

    def transform_all_arrays(self):
        for i in range(1, 8):
            self.transform("Data/npy_video_arrays/Ballenwerper_sync_380fps_00" + str(i) + ".npy",
                           "Data/npy_video_arrays/Ballenwerper_sync_380fps_00" + str(i) + ".avi")

    def get_frames(self, file):
        video = cv2.VideoCapture(file)
        fps = video.get(cv2.CAP_PROP_FPS)
        return fps
