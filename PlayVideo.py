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

    def load_video_eager(self):
        # Eager loaded data file (all in memory before continuing)
        return np.load("Data/Ballenwerper_sync_380fps_006.npy")

    def load_video_lazy(self):
        # Lazy loading means the data is only loaded into memory when it is needed (frame-by-frame)
        return np.lib.format.open_memmap('Data/Ballenwerper_sync_380fps_006.npy', mode='r+')

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
        # Release the display window
        cv2.destroyAllWindows()

    def playVideoDifference(self):
        difference(self.video_data)

