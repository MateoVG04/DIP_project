import cv2
import numpy as np

class CircleTracker:
    def __init__(self, video_data):
        self.video_data = video_data
        self.screen_width = 800
        self.screen_height = 600
        self.selected_circle = None  # Holds the selected circle (x, y, radius)
        self.clicked = False
        self.tracking_window = None  # The window to track the circle
        self.template = None  # Template of the selected region for tracking

    def select_circle(self, event, x, y, flags, param):
        """Handles circle selection on mouse click."""
        frame = param['frame']
        if frame is None:
            print("Frame is None; cannot detect circles.")
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse clicked at ({x}, {y})")  # Debug: Confirm click

            # Preprocess the frame for circle detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Perform Hough Circle Detection
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                       param1=50, param2=20, minRadius=5, maxRadius=15)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Find the circle closest to the click
                closest_circle = None
                min_distance = float('inf')
                for circle in circles:
                    cx, cy, r = circle
                    distance = (cx - x)**2 + (cy - y)**2
                    if distance < min_distance:
                        min_distance = distance
                        closest_circle = circle

                if closest_circle is not None:
                    self.selected_circle = tuple(closest_circle)
                    self.clicked = True

                    # Initialize the tracking window and template
                    cx, cy, r = self.selected_circle
                    self.tracking_window = (max(0, cx - r), max(0, cy - r), r * 2, r * 2)
                    self.template = gray[self.tracking_window[1]:self.tracking_window[1] + self.tracking_window[3],
                                         self.tracking_window[0]:self.tracking_window[0] + self.tracking_window[2]]
                else:
                    print("No circle found near the click.")
            else:
                print("No circles detected.")

    def playVideoNormal(self):
        """ Main method to display and track the selected circle """
        cv2.namedWindow('Video')

        # Frame parameter dictionary to pass to the callback
        param = {'frame': None}
        cv2.setMouseCallback('Video', self.select_circle, param)

        for frame in self.video_data:
            resizedFrame = cv2.resize(frame, (self.screen_width, self.screen_height), interpolation=cv2.INTER_AREA)

            # Convert grayscale frame to BGR for display purposes
            displayFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_GRAY2BGR) if len(resizedFrame.shape) == 2 else resizedFrame
            grayFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY) if len(resizedFrame.shape) == 3 else resizedFrame
            param['frame'] = displayFrame  # Update current frame for callback

            # Track the selected circle if available
            if self.clicked and self.template is not None and self.tracking_window is not None:
                # Perform template matching to track the circle
                result = cv2.matchTemplate(grayFrame, self.template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                # Update the circle's position based on the best match
                if max_val > 0.6:  # Confidence threshold
                    cx, cy = max_loc[0] + self.tracking_window[2] // 2, max_loc[1] + self.tracking_window[3] // 2
                    self.selected_circle = (cx, cy, self.selected_circle[2])  # Keep the same radius

                    # Update the tracking window
                    r = self.selected_circle[2]
                    self.tracking_window = (max(0, cx - r), max(0, cy - r), r * 2, r * 2)
                    self.template = grayFrame[self.tracking_window[1]:self.tracking_window[1] + self.tracking_window[3],
                                              self.tracking_window[0]:self.tracking_window[0] + self.tracking_window[2]]

                else:
                    print("Lost track of the circle.")

            # Draw the selected circle if available
            if self.selected_circle is not None:
                cx, cy, r = self.selected_circle
                cv2.circle(displayFrame, (cx, cy), r, (0, 255, 0), 2)  # Green circle outline
                cv2.circle(displayFrame, (cx, cy), 2, (0, 0, 255), -1)  # Red center dot

            # Display the frame
            cv2.imshow('Video', displayFrame)

            # Wait for the 'q' key to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
