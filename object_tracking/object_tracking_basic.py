from typing import Literal
import cv2
from cv2.typing import MatLike
import numpy as np

from PlayVideo import PlayVideo

tracker_literal = Literal["KCF"]

def match_tracker(tracker: tracker_literal):
    match tracker:
        case "KCF":
            return cv2.TrackerMIL

def select_bounding_box(frame) -> cv2.typing.Rect:
    return cv2.selectROI(frame, False)

def track_motion(video: np.memmap,
                 initial_bounding_box: cv2.typing.Rect | None = None,
                 algorithm: tracker_literal = "KCF",
                 ):
    """

    :param video:
    :param algorithm: Literal for which Algorithm to use
    :param initial_bounding_box: The bounding box for the to be tracked object
    """

    # Initialize tracker with frame and initial bounding box
    tracker = cv2.TrackerMIL.create()

    frame: MatLike = np.ascontiguousarray(video[0])
    tracker.init(frame, initial_bounding_box)

    # Frame by frame analysis
    for frame in video:
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, "Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

def main():
    video = PlayVideo(file_path="../Data/Ballenwerper_sync_380fps_006.npy")

    frame = video.video_data[0]
    bounding_box = select_bounding_box(frame=frame)
    track_motion(video=video.video_data, initial_bounding_box=bounding_box)

if __name__ == "__main__":
    main()