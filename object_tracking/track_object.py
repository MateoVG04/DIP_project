import cv2
import cv2.typing

import numpy as np

from OldCode.PlayVideo import PlayVideo


class TrackRectangles:
    def __init__(self, video: PlayVideo, initial_bounding_box: cv2.typing.Rect):
        """
        :param video:
        :param initial_bounding_box: Should match in dimension with the input video
        """
        self.video = video
        self.initial_bounding_box: cv2.typing.Rect = initial_bounding_box
        self.track_algorithm = cv2.TrackerMIL

    def track_rectangle(self) -> list[cv2.typing.Rect]:
        # Defining First Frame
        frame: cv2.typing.MatLike = np.ascontiguousarray(self.video.video_data[0])

        # Initialize tracker with frame and initial bounding box
        tracker = self.track_algorithm.create()
        tracker.init(frame, self.initial_bounding_box)

        # Resulting track Data
        result_list: list[cv2.typing.Rect] = []

        # Frame by frame analysis
        for frame in self.video.video_data:
            # Update tracker
            ok, bbox = tracker.update(frame)

            if not ok:
                raise Exception('Tracking error')

            # Append tracked frame
            result_list.append(bbox)
        return result_list
