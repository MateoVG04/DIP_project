# DIP_project
## Opgave
* Do all the 7 recordings film the same movement? What is the deviation?
* What frequencies do the rods have when moving? What is the amplitude?
* What is vibrating, why is it vibrating?
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

## Hu moment
Useful links:
```
https://learnopencv.com/shape-matching-using-hu-moments-c-python/
```
This was one of the methods we used for tracking, but it wasn't very reliable and only seemed to be working for the right cup.
You can still see the result if you want by running the next commands.

Left cup:
```bash
python main.py humoment LeftCup
```
Righ cup:
```bash
python main.py humoment RightCup
```