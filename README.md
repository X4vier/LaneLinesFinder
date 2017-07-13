# Lane Line Finding
---

This is a project I completed as part of Udacity's self-driving car nanodegree program in July 2017.

(https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

The code in this project looks at RGB images taken from a car's dashboard camera and works out where the lane lines are in the image. It then draws the lane lines over the original image. To see the final result, take a look at the video file `output.mp4`

This project doesn't use any artificial intelligence techniques, just old-school computer vision, it serves to demonstrate my ability to write effective, readable python code.

The rest of this document is devoted to explaining how the code works.

---

## Camera Calibration

The code for calibrating the camera and correcting for distortion is found in the file `camera_callibration.py`. The `cv2.findChessboardCorners()` method is used to find the chessboard corners in the calibration images, then the distortion matrix and distortion coefficients are calculated using the `cv2.calibrateCamera()` method, using chessboard corners as image points, and points on a 2D plane as object points. The distortion coefficients and distortion matrix are saved to the file `distortion.p`. I wrote a function called `undistort()` which uses the contents of this file to undistort camera images.

![Distortion Image](https://github.com/X4vier/LaneLinesFinder/blob/master/example_outputs/distortion_example.jpg)

## Perspective Transform

To find lane lines, it is useful to transform images to a �bird�s-eye view�. The code for calculating the perspective transform matrix is found in `perspective_transform.py`. The perspective transform matrix is calculated using an image of straight road lines. Four points which sit approximately on the corners of a rectangle when viewed from above are specified, then the `cv2.getPerspectiveTransform()` function uses these points to calculate the perspective transform matrix. The transformation matrix is saved to the file `perspective_transform.p`. I wrote a function called `perspective_transform()` which uses the contents of this file to transform camera images.

![Perspective Transofrm Image](https://github.com/X4vier/LaneLinesFinder/blob/master/example_outputs/perspective_example.jpg)


## Binary Thresholding

The code for converting the camera images to binary images is in the file  `binary_threshold.py`. In order to determine which pixels to include in the binary image, three filters are applied- a filter which checks whether the pixel�s red component is within a certain range of values, a filter which checks whether the pixel�s HLS saturation is within a certain range of values, and a filter which checks whether the value obtained by applying the sobel operator in the x direction is within a certain range of values. The thresholds for these filters were determined by experimenting on the test set images. A mask is also applied to the image, so that pixels in a region of the image where lane lines don�t usually appear are removed. If a pixel passes through any of the three filters and isn�t masked, then it will be included in the output of the `binary_threshold()` function.

![Threshold Image](https://github.com/X4vier/LaneLinesFinder/blob/master/example_outputs/threshold_example.jpg)

![Mask Image](https://github.com/X4vier/LaneLinesFinder/blob/master/example_outputs/mask_example.jpg)


--**Xavier O'Rourke**