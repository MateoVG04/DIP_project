from itertools import pairwise

import cv2.typing

from OldCode.PlayVideo import PlayVideo
from object_tracking.test_tracked_objects import TEST_TRACKED_RECTS


class CalculateDeformation:
    def __init__(self, tracked_objects: list[list[cv2.typing.Rect]]):  # List of list of tracked objects NOT PER FRAME
        self.tracked_objects = tracked_objects
        self.tracked_objects_centers: list[list[tuple[int, int]]] = [list(map(self.middle, tracked_object_list)) for tracked_object_list in tracked_objects]

        # PER FRAME
        self.objects_per_frame: list[tuple[cv2.typing.Rect, cv2.typing.Rect, cv2.typing.Rect]] = self.linear_to_per_frame(tracked_objects)

    @classmethod
    def linear_to_per_frame(cls, input_list: list[list[cv2.typing.Rect]]) -> list[tuple[cv2.typing.Rect, cv2.typing.Rect, cv2.typing.Rect]]:
        frame_count = len(input_list[0])

        result: list[tuple[cv2.typing.Rect, cv2.typing.Rect, cv2.typing.Rect]] = []
        for i in range(frame_count):
            frame = []
            for object_track in input_list:  # For each tracked object, add to the currently selected frame
                frame.append(object_track[i])
            result.append(tuple(frame))
        return result

    @classmethod
    def per_frame_to_linear(cls, input_list: list[list[cv2.typing.Rect]]) -> list[list[cv2.typing.Rect]]:
        vector_count = len(input_list[0])
        result: list[list] = []

        # Initializing
        for _ in range(vector_count):
            result.append([])

        # Placing values
        for frame in input_list:
            for i, value in enumerate(frame):
                result[i].append(value)

        return result

    @classmethod
    def middle(cls, rect: cv2.typing.Rect) -> tuple[int, int]:
        return rect[0] + rect[2] // 2, rect[1] + rect[3] // 2

    @classmethod
    def length(cls, lhs: cv2.typing.Rect, rhs: cv2.typing.Rect):
        """
        Returns length of line formed by connecting the middle of lhs and rhs
        Uses Pythagoras Theorem
        :param lhs: Rectangle
        :param rhs: Rectangle
        :return: Integer
        """
        lhs_x, lhs_y = cls.middle(lhs)
        rhs_x, rhs_y = cls.middle(rhs)
        return pow(pow(abs(lhs_x - rhs_x), 2) + pow(abs(lhs_y - rhs_y), 2), 0.5)

    @classmethod
    def calculate_internal_length(cls, objects_in_frame: tuple[cv2.typing.Rect,cv2.typing.Rect, cv2.typing.Rect]) -> list[int]:
        """
        CALCULATED PER FRAME

        Given x number of points which from a framework as input
        Calculate each length of line connecting n and n+1
        :param objects_in_frame:
        :return: List of lengths
        """
        return [cls.length(lhs, rhs) for lhs, rhs in pairwise(objects_in_frame)] + [cls.length(lhs=objects_in_frame[0], rhs=objects_in_frame[-1])]

    @classmethod
    def summarise_length_list(cls, internal_lengths: list[int]):
        # Distance between minimum and maximum length in the vector
        min_max_distance = max(internal_lengths) - min(internal_lengths)

        # Calculate the deviation
        deviation = list(map(lambda val: abs(val - internal_lengths[0]), internal_lengths[1:]))
        min_deviation = min(deviation)
        max_deviation = max(deviation)

        return min_deviation, max_deviation, min_max_distance

    def calc_variation_of_length(self):
        """
        Calculates the difference between lengths on every frame
        :return:
        """
        internal_lengths: list[list[int]] = []
        for frame in self.objects_per_frame:
            internal_length = self.calculate_internal_length(frame)
            internal_lengths.append(internal_length)

        for internal_length in self.per_frame_to_linear(internal_lengths):
            min_deviation, max_deviation, min_max_distance = self.summarise_length_list(internal_length)
            print(min_deviation, max_deviation, min_max_distance)


if __name__ == '__main__':
    to_track_objects = [
        (193, 1193, 210, 203),  # Left bucket
        (736, 1142, 57, 54),  # Rotor attachment point
        (1180, 941, 67, 62),  # Lever attachment point
        (1437, 517, 243, 240)  # Right bucket
    ]
    playVideo = PlayVideo(file_path="../Data/Ballenwerper_sync_380fps_006.npy")

    tracked_objects_result: list[list[cv2.typing.Rect]] = TEST_TRACKED_RECTS
    # for to_track in to_track_objects:
    #     tracker = TrackRectangles(video=playVideo, initial_bounding_box=to_track)
    #     tracked_objects.append(tracker.track_rectangle())

    deformation_calculator = CalculateDeformation(tracked_objects_result)

    deformation_calculator.calc_variation_of_length()
