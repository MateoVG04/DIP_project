import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import time


class HuMoment:
    def __init__(self):
        self.reference_image = None
        self.reference_hu = None

    def calculate_hu_moments(self, image):
        """Calculate Hu moments for a grayscale binary image."""
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments)

        # Log transform to bring hu moments in the same range
        for i in range(7):
            hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-6)
        return hu_moments

    def compare_hu_moments(self, hu_moments, threshold=0.1):
        """
        Compare two sets of Hu moments using a threshold.
        """
        diff = np.abs(self.reference_hu - hu_moments)
        return np.all(diff < threshold)

    def process_single_frame(self, frame):
        """
        Process a single frame: detect objects, match Hu moments, and draw bounding boxes.
        """
        if len(frame.shape) == 3:  # If the frame is RGB, convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:  # Assume it's already grayscale
            gray = frame
        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Convert to binary
        _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Ignore very small contours for performance
            area = cv2.contourArea(contour)
            if area > 500:
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                hu_moments = self.calculate_hu_moments(mask)
                if self.compare_hu_moments(hu_moments, threshold=0.3):
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame

    def process_and_write_frame(self, index, frame, hu_moment_obj):
        """Process a frame and return the processed frame with its index."""
        processed_frame = hu_moment_obj.process_single_frame(frame)
        return index, processed_frame

    def process_frames_and_save(self, input_npy, output_video_path, reference_image):
        """
        Process frames in parallel using ThreadPoolExecutor, match objects with reference images,
        and save the processed video to an output file.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            print("Directory {} doesn't exist.".format(output_dir))
            return

        # Fixed resolution
        target_width = 1920
        target_height = 1080
        target_size = (target_width, target_height)
        print(f"Target resolution: {target_width}x{target_height}")

        # Load frames
        print("Loading input frames...")
        frames = np.load(input_npy)
        total_frames = len(frames)
        print(f"Total frames: {total_frames}")

        # Calculate hu moments for reference image
        print("Processing reference images...")
        # Read image as grayscale
        self.reference_image = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
        # Convert to binary
        _, reference_binary = cv2.threshold(self.reference_image, 128, 255, cv2.THRESH_BINARY)
        # Calculate reference hu moments
        self.reference_hu = self.calculate_hu_moments(reference_binary)
        print("Reference Hu moments calculated.")

        # Initialize Video Writer
        print("Initializing video writer...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_height, frame_width = frames[0].shape[:2]
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, target_size)
        print("Video writer initialized.")

        # Start processing frames in parallel
        print("Processing frames in parallel...")
        start_time = time.time()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_and_write_frame, i, frame, self) for i, frame in enumerate(frames)]
            results = [future.result() for future in futures]  # Gather all results

            # Sort frames by index to maintain order
            print("Sorting frames...")
            results.sort(key=lambda x: x[0])
            for _, processed_frame in results:
                # Check for frame validity
                if processed_frame is None or processed_frame.size == 0:
                    print("Skipping invalid frame.")
                    continue

                # Resize frame to 1920x1080
                processed_frame = cv2.resize(processed_frame, target_size)

                # Ensure BGR format
                if len(processed_frame.shape) == 2:  # Grayscale image
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

                # Write frame to video
                out.write(processed_frame)

        # Finalize video writer
        out.release()
        print(f"Processing complete! Video saved to {output_video_path}.")
        print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

    def play_video(self, video_path):
        """
        Play a video using OpenCV.
        :param video_path: Path to the video file to be played.
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        while True:
            # Read each frame
            ret, frame = cap.read()

            # If the frame was read correctly
            if not ret:
                print("End of video or error in reading the video.")
                break

            # Display the frame
            cv2.imshow("Video Playback", frame)

            # Wait for a key press and check if it is the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Video playback stopped by user.")
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
