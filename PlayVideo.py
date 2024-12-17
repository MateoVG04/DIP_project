from typing import Optional

import numpy as np
import cv2
from openpyxl.styles.builtins import total
import matplotlib.pyplot as plt


class PlayVideo:
    def __init__(self, file_path: Optional[str] = None):
        self.windowName = "Temp"
        self.screen_width, self.screen_height = self.get_screen_resolution()
        self.video_data: np.memmap = self.load_video_lazy(file_path=file_path)
        self.fps = 380
        self.is_counter = False
        self.counter = 0

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
            file_path = 'Data/Ballenwerper_sync_380fps_006 - Copy.npy'

        # Lazy loading means the data is only loaded into memory when it is needed (frame-by-frame)
        return np.lib.format.open_memmap(filename=file_path, mode='r+')

    def frameCounter(self):
        if not self.is_counter:
            self.counter = 1
            self.is_counter = True
        else:
            self.counter +=1
        totalFrames = len(self.video_data)
        return self.counter, totalFrames

    def resize_frame(self, frame):
        return cv2.resize(frame, (self.screen_width, self.screen_height), interpolation=cv2.INTER_AREA)

    def displayFrame(self, frame):
        # Display the frame
        resizedFrame = self.resize_frame(frame)
        cv2.imshow('Video', resizedFrame)
        counter, totalFrames = self.frameCounter()
        cv2.setWindowTitle("Video", "frame " + str(counter) + " of " + str(totalFrames))

        # Wait for 25ms before moving to the next frame (40 FPS)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return
    def hitMissOperation(self,frame,hitMissElement):
        # You combine them using a single matrix where the foreground has 1s, the background has -1s, and unused areas have 0s.
        hitMiss = cv2.morphologyEx(frame,cv2.MORPH_HITMISS,hitMissElement)
        return hitMiss

    def playVideoNormal(self):
        hitMissElement = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0,],
                                     [0, 0, 0, 1, 1, 1, 0, 0, 0,],
                                     [0, 0, 1, 1, 1, 1, 1, 0, 0,],
                                     [0, 1, 1, 1, 1, 1, 1, 1, 0,],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1,],
                                     [0, 1, 1, 1, 1, 1, 1, 1, 0,],
                                     [0, 0, 1, 1, 1, 1, 1, 0, 0,],
                                     [0, 0, 0, 1, 1, 1, 0, 0, 0,],
                                     [0, 0, 0, 0, 1, 0, 0, 0, 0,]])
        for frame in self.video_data:
            # Display the frame
            resizedFrame = cv2.resize(frame, (self.screen_width, self.screen_height), interpolation=cv2.INTER_AREA)
            #resizedFrame = cv2.GaussianBlur(resizedFrame, (5, 5), 0)
            #resizedFrame = cv2.threshold(resizedFrame,50,255,cv2.THRESH_BINARY)[1]
            #resizedFrame = self.hitMissOperation(resizedFrame,hitMissElement)
            cv2.imshow('Video', resizedFrame)
            counter, totalFrames = self.frameCounter()
            cv2.setWindowTitle("Video","frame "+str(counter)+" of "+str(totalFrames))
            # Wait for 25ms before moving to the next frame (40 FPS)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            return

    def playVideoNormal(self):
        for frame in self.video_data:
            # Display the frame
            self.displayFrame(frame)
        self.is_counter = False
        self.counter = 0


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
                counter, totalFrames = self.frameCounter()
                cv2.setWindowTitle("Video", "frame " + str(counter) + " of " + str(totalFrames))

            # Update the Previousframe
            Previousframe = blur_frame
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
            # Wait for 25ms before moving to the next frame (approx. 40 FPS)

        self.is_counter = False
        self.counter = 0





