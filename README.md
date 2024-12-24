# DIP_project
## Prerequisites
The next files need to be put in the Data directory
```
Ballenwerper_sync_380fps_006.npy
Ballenwerper_sync_380fps_006.npy_output_video.mp4
```
## Deformation
Our method:
1. Tracking 4 points
2. Calculating the distance between these 4 points
3. Plot the distance for each frame, if there is a difference then there is deformation
```bash
python main.py deformation
```
## Angle, speed and vibration
Vibrations:
* These are only calculated where the speed is 0. Done by taking the difference between the first frame of a segment of 0 speed and the frames with 0 speed after that.
* Probably inaccurate
### Pre created data
```bash
python main.py object_tracking_pre_created_data
```
### Own data
The way this works is that you draw a box around the part that you want to follow.
After you have selected it you need to press enter.
```bash
python main.py object_tracking_own_data
```
### Circle Tracking
One way we tried tracking the movement of the arm was by tracking the bearings. We thought this would be a good approach since we wouldn't have to track a complex shape but just a circle. This proved to be more difficult than expected, the program had trouble locking on to the correct circle. Even if it did lock on to the bearing, the tracking wasn't accurate enough to perform the necessary measurements

Using the following command you can click a circle to be tracked, notice how it only sometimes picks the bearing.

```bash
python main.py circle_tracking
```
### Frame difference
We also made a way where you can see the difference between each frame.
```bash
python main.py play_frame_difference
```

## Hu moment
This was one of the methods we used for tracking, but it wasn't very reliable and only seemed to be working for the right cup.

Potential improvements:
1. Comparing the area size between the contours with similar hue values and the reference image.
2. Using the built-in cv2.matchShapes function that uses hu moments instead of calculating the hu moments ourselves.

Useful links:
```
https://learnopencv.com/shape-matching-using-hu-moments-c-python/
```
### Pre created data
Left cup:
```bash
python main.py hu_moment_pre_created_data LeftCup
```
Righ cup:
```bash
python main.py hu_moment_pre_created_data RightCup
```
### Own data
If you want to use your own reference image you need to save it under the directory Data as ReferenceImage.png

WARNING: This takes a lot of processing power and time so while its running using the computer is not feasible!
```bash
python main.py hu_moment_own_data
```