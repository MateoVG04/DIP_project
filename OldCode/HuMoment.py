from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import time


class HuMomentClass:
    def __init__(self):
        self.reference_area_image_1 = None
        self.reference_hu_1 = None
        self.reference_image_1 = None

    @staticmethod
    def calculate_hu_moments(image):
        """Calculate Hu moments for a grayscale binary image."""
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments)
        for i in range(7):
            hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-6)
        return hu_moments

    @staticmethod
    def compare_hu_moments(hu1, hu2, threshold=0.1):
        """
        Compare two sets of Hu moments using a threshold.
        """
        diff = np.abs(hu1 - hu2)
        return np.all(diff < threshold)

    def process_single_frame(self, frame):
        """
        Process a single frame: detect objects, match Hu moments, and draw bounding boxes.
        """
        # Ensure frame is 3-channel color
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        """# Convert to grayscale and find contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive thresholding for images with varying lighting
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)"""
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Loop through each contour to match objects
        for contour in contours:
            # Ignore very small contours for performance
            area = cv2.contourArea(contour)
            if area > 500:
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                hu_moments = self.calculate_hu_moments(mask)
                # Compare hu moments with first reference image
                if self.compare_hu_moments(hu_moments, self.reference_hu_1, threshold=0.5):
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame

    def process_and_write_frame(self, index, frame, hu_moment_obj):
        """Process a frame and return the processed frame with its index."""
        processed_frame = hu_moment_obj.process_single_frame(frame)
        return index, processed_frame

    def process_frames_and_save(self, input_npy, output_video_path, reference_image_1):
        """
        Process frames in parallel using ThreadPoolExecutor, match objects with reference images,
        and save the processed video to an output file.
        """
        self.reference_image_1 = cv2.imread(reference_image_1)
        # Load frames
        print("Loading input frames...")
        frames = np.load(input_npy)
        total_frames = len(frames)
        print(f"Total frames: {total_frames}")
        # Calculate Hu moments for the reference images
        print("Processing reference images...")
        # If its color mode change it to gray scale
        if len(self.reference_image_1.shape) == 2:
            reference_gray_1 = cv2.cvtColor(self.reference_image_1, cv2.COLOR_GRAY2BGR)
        else:
            reference_gray_1 = self.reference_image_1

        _, reference_binary_1 = cv2.threshold(reference_gray_1, 128, 255, cv2.THRESH_BINARY)
        self.reference_hu_1 = self.calculate_hu_moments(reference_binary_1)
        print("Reference Hu moments calculated.")
        # Initialize Video Writer
        print("Initializing video writer...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (2240, 1726))
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
                out.write(processed_frame)  # Write each processed frame sequentially
        # Finalize video writer
        out.release()
        print(f"Processing complete! Video saved to {output_video_path}.")
        print(f"Total processing time: {time.time() - start_time:.2f} seconds.")
