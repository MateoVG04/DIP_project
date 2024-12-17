import cv2

from humoment import HuMoment

input_npy = "Data/Ballenwerper_sync_380fps_006.npy"
output_video_path = "Data/output_video.avi"

left_cup_reference_image = cv2.imread("Data/LeftCup.png")
right_cup_reference_image = cv2.imread("Data/RightCup.png")
humoment = HuMoment.HuMomentClass()
humoment.process_frames_and_save(input_npy=input_npy, output_video_path=output_video_path, reference_image_1=left_cup_reference_image, reference_image_2=right_cup_reference_image)
#HuMoment.play_video("Data/output_video.avi")
