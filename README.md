# Self_Driving_Car_Lane_Detection_on_Video


The script [LaneFInding.py](https://github.com/hamza9305/Self_Driving_Car_Lane_Detection_on_Video/blob/main/LaneFInding.py) detects the lane markings in a video using the following pipeline
- Convert the image to grayscale
- Apply Gaussian blur
- Apply Canny Edge
- Finding region of interest
- Apply Hough Transform
- Merge the original image with Hough Lines

I used a function which takes in the image and detected line from the hough function. For every line, the function finds the gradient and intercept, and then append it to two different lists (right_lines, left_lines) based on the value of gradient. Now using the mean values of gradient and intercept, a line is predicted which would roughly cover 60 percent of the image height. This way the function is able to extrapolate the detected lines.


<p align="center">
  <img width="480" height="270" src="https://github.com/hamza9305/Self_Driving_Car_Lane_Detection_on_Video/blob/main/resource/output.gif">
</p>
