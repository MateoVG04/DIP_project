from typing import Optional

import numpy as np
import cv2
import Disp_Difference_Frames
from Disp_Difference_Frames import difference


class PlayVideo():
    def __init__(self):
        self.windowName = "Temp"
        self.screen_width, self.screen_height = self.get_screen_resolution()
        self.video_data = self.load_video_lazy()
        self.fps = 380

    def get_screen_resolution(self):
        screen = cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        screen_width = cv2.getWindowImageRect(self.windowName)[2]
        screen_height = cv2.getWindowImageRect(self.windowName)[3]
        cv2.destroyWindow(self.windowName)
        return screen_width,screen_height

    @classmethod
    def load_video_eager(cls, file_path: Optional[str] = None):
        if file_path is None:
            file_path = 'Data/Ballenwerper_sync_380fps_006.npy'
        # Eager loaded data file (all in memory before continuing)
        return np.load(file_path)

    @classmethod
    def load_video_lazy(cls, file_path: Optional[str] = None):
        if file_path is None:
            file_path = 'Data/Ballenwerper_sync_380fps_006.npy'

        # Lazy loading means the data is only loaded into memory when it is needed (frame-by-frame)
        return np.lib.format.open_memmap(filename=file_path, mode='r+')

    def playVideoNormal(self):
        for frame in self.video_data:
            # Display the frame
            resizedFrame = cv2.resize(frame, (self.screen_width, self.screen_height), interpolation=cv2.INTER_AREA)
            cv2.imshow('Video', resizedFrame)

            # Wait for 25ms before moving to the next frame (40 FPS)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed.")
                break


    def playVideoDifference(self):
        Previousframe = None  # Initialize Previousframe as None
        # Assuming the shape is (num_frames, height, width, channels)
        for frame in self.video_data:
            # Apply Gaussian blur to reduce noise
            blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)

            if Previousframe is not None:
                # Compute absolute difference between frames
                frame_diff = cv2.absdiff(blur_frame, Previousframe)

                # Display the frame difference
                cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Video', 800, 600)
                cv2.imshow('Video', 255 - frame_diff)

            # Update the Previousframe
            Previousframe = blur_frame
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
            # Wait for 25ms before moving to the next frame (approx. 40 FPS)


