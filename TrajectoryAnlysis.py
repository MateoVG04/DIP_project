import matplotlib.pyplot as plt
import cv2
import numpy as np


class TrajectoryAnalysis:
    def __init__(self,boxes_list,playVideo):
        self.boxList = boxes_list
        self.playVideo = playVideo


    def displayPath(self):
        plt.plot([1, 2, 3, 4])
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
        plt.show()
