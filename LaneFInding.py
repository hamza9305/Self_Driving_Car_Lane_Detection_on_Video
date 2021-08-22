from moviepy.editor import VideoFileClip
import math
import cv2
import numpy as np

input_folder = r"C:\Users\Hamza Ahmed\PycharmProjects\self_Driving_cars_degree\2)Lane_Findiing_Project\CarND-LaneLines-P1\test_videos/solidWhiteRight.mp4"
output_folder = r'F:\GitHub_Live_Projects\Self_Driving_Cars\8. Lane_Detection_on_Video\resource\output video/solidWhiteRight.mp4'


def grayscale(img):
    """Applies the Grayscale transform
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    line segments are separated by their  slope ((y2-y1)/(x2-x1)) to decide which
    segments are part of the left line vs. the right line.  Then, you can average
    the position of each of the lines and extrapolate to the top and bottom of the lane.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    initial_img * α + img * β + γ
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def slope_lines(image, lines):
    img = image.copy()
    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # checking for vetical lines
            if x1 == x2:
                pass
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m, c))
                elif m >= 0:
                    right_lines.append((m, c))

    left_line = np.mean(left_lines, axis=0)
    right_line = np.mean(right_lines, axis=0)

    for slope, intercept in [left_line, right_line]:
        # getting complete height of image in y1
        rows, cols = image.shape[:2]
        y1 = int(rows)

        # taking y2 upto 60% of actual height or 60% of y1
        y2 = int(rows * 0.6)

        # we know that equation of line is y=mx +c so we can write it x=(y-c)/m
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        draw_lines(img, np.array([[[x1, y1, x2, y2]]]))

    return cv2.addWeighted(image, 0.7, img, 0.4, 0.)


def vertices(image):
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.15, rows]
    top_left = [cols * 0.45, rows * 0.6]
    bottom_right = [cols * 0.95, rows]
    top_right = [cols * 0.55, rows * 0.6]

    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver


def hough_lines_extrapolation(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # edited
    # draw_lines(line_img, lines)
    line_img = slope_lines(line_img, lines)
    return line_img


def pipeline_for_finding_lane(image):
    # Convert the image to grayscale
    grayScale = grayscale(image)

    # Apply Gaussian blur
    gaussianBlur = gaussian_blur(grayScale, 3)

    # Apply Canny Edge
    cannyEdge = canny(gaussianBlur, low_threshold=50, high_threshold=150)

    # Finding region of interest
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    vertices = np.array([[(131, 538), (443, 324), (540, 324), (imshape[1], imshape[0])]], dtype=np.int32)
    roiImage = region_of_interest(cannyEdge, vertices)

    # Apply Hough Transform
    houghLines = hough_lines(roiImage, rho=2, theta=np.pi / 180, threshold=1, min_line_len=1, max_line_gap=5)

    # Extend the lines
    weightImage = weighted_img(image, houghLines, 0.8, β=1., γ=0.)

    return weightImage


def main():
    clip1 = VideoFileClip(input_folder)
    white_clip = clip1.fl_image(pipeline_for_finding_lane)
    white_clip.write_videofile(output_folder, audio=False)

if __name__ == '__main__':
    main()