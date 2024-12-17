import math
from itertools import pairwise
import cv2
import matplotlib.pyplot as plt



class TrajectoryAnalysis:
    def __init__(self,boxes_list,playVideo):
        self.boxList = boxes_list
        self.playVideo = playVideo

    @classmethod
    def calc_middle(cls, input_array: list[cv2.typing.Rect]) -> tuple[int, int]:
        # The rectangle class is a tuple[x: int, y: int, int, int]
        # We calculate the min x and min y per index
        min_x = min(input_array, key=lambda rect: rect[0])
        min_y = min(input_array, key=lambda rect: rect[1])
        return min_x[0], min_y[1]

    @classmethod
    def align_with_zero(cls, input_array: list[cv2.typing.Rect]) -> list[cv2.typing.Rect]:
        # Calculate middle points
        min_x, min_y = cls.calc_middle(input_array)

        # We subtract these values off every rectangle in the list and return
        return list(map(lambda rect: (rect[0] - min_x, rect[1] - min_y, rect[2], rect[3]), input_array))

    @classmethod
    def calculate_angle(cls, input_array: list[cv2.typing.Rect]) -> list[float]:
        """
        More information for calculating the angle
        https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-vectors
        https://www.tutorialgateway.org/python-atan2/
        :param input_array:
        :return: list of angles represented by floating point
        """

        # First align the boxes with 0 and calculate their middle
        zero_aligned_list = cls.align_with_zero(input_array)
        middle_x, middle_y = cls.calc_middle(zero_aligned_list)

        return list(map(lambda rect: math.atan2(rect[0] - middle_x, rect[1] - middle_y), zero_aligned_list))

    @classmethod
    def calculate_angle_speed(cls, input_angles: list[float]) -> list[float]:
        """
        Iterates pair wise (per two elements) and calculates the angle
            pairwise(input_angles) returns a list[tuple[float, float], ...] for every n, n+1 element
            lambda operation calculates the difference in angle.
            Could be negative!
        :param input_angles:
        :return: list of difference in angle represented by floating point
        """
        return list(map(lambda pair: abs(pair[1] - pair[0]), pairwise(input_angles)))

    def displayPath(self):
        x_list: list[int] = []
        y_list: list[int] = []
        for box in self.boxList:
            xCord = box[0]
            yCord = box[1]
            width = box[2]
            height = box[3]
            print("("+str(xCord)+","+str(yCord)+","+str(width)+","+str(height)+"),")
            centerXCord = (xCord + width) / 2
            x_list.append(centerXCord)

            centerYCord = (yCord + height) / 2
            y_list.append(centerYCord)
        plt.plot(x_list, y_list)
        plt.title('position')
        plt.xlabel('x-coordinaat')
        plt.ylabel('y-coordinaat')
        #plt.show()

    def calculate_degree(self,angles_rad):
        angles_degree = []
        for angle in angles_rad:
            angle_calculate = angle * (180/math.pi)
            angles_degree.append(angle_calculate)
        return angles_degree

    def display_angle(self):
        angles_rad = self.calculate_angle(self.boxList)
        angles_degree = self.calculate_degree(angles_rad)
        x_axis = list(range(len(self.boxList)))
        plt.plot(x_axis, angles_degree)
        plt.title('angles')
        plt.xlabel('[°]')
        plt.ylabel('frame')
        #plt.show()

    def display_speed(self):
        angles = self.calculate_angle(self.boxList)
        angles_degree = self.calculate_degree(angles)
        speed = self.calculate_angle_speed(angles_degree)
        x_axis = list(range(len(self.boxList)-1))
        plt.plot(x_axis,speed)
        plt.title('Speed')
        plt.title('[°/frame]')
        plt.ylabel('frame')
        #plt.show()
