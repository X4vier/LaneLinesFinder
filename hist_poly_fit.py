import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from binary_threshold import binary_threshold
from perspective_transform import perspective_transform
from camera_calibration import undistort
import glob

def hist_poly_fit(img):
    """
        Takes in a binary, birds-eye-view image and fits a second-order polynomial
        to the lane lines.
    """

    # Create color image to store the results
    out_img = np.dstack((img, img, img))

    # See which columns in the bottom half the preprocessed image contain the most pixels
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)

    # Assume left/right lane lines are to the left/right of image center
    midpoint = histogram.shape[0]//2

    # Set the starting position of the lane lines to be the max of the histogram
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = img.shape[0]//nwindows

    nonzero = img.nonzero()

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    # Width of windows
    margin = 100

    # Minimum number of pixels needed to recenter window
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - window_height*(window+1)
        win_y_high = img.shape[0] - window_height*(window)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # When the main method is calling this function we want to see the windows, but
        # not when this function's being called from elsewhere.
        if __name__ == '__main__':
            cv2.rectangle(img=out_img, pt1=(win_xleft_low, win_y_low), pt2=(win_xleft_high, win_y_high), color=(0, 255, 0), thickness=2)
            cv2.rectangle(img=out_img, pt1=(win_xright_low, win_y_low), pt2=(win_xright_high, win_y_high), color=(0, 255, 0), thickness=2)


        # Identify the nonzero pixels within the window
        good_left_inds = \
            ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)) \
            .nonzero()[0]

        good_right_inds = \
            ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)) \
        .nonzero()[0]


        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] # red
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 225] # blue

    return left_fit, right_fit, leftx_base, rightx_base, out_img


if __name__ == "__main__":

    # See what is happening to the test images (useful in debugging)
    image_urls = glob.glob('test_images/*.jpg')

    for url in image_urls:
        img = mpimg.imread(url)
        undistorted = undistort(img)
        thresholded = binary_threshold(undistorted)
        transformed = perspective_transform(thresholded)
        left_fit, right_fit, leftx_base, rightx_base, out_img = hist_poly_fit(transformed)

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    # Save an example image:
    img = mpimg.imread("test_images/test3.jpg")
    undistorted = undistort(img)
    thresholded = binary_threshold(undistorted)
    transformed = perspective_transform(thresholded)
    left_fit, right_fit, leftx_base, rightx_base, out_img = hist_poly_fit(transformed)

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(transformed, cmap='gray')
    ax1.set_title('Input Image', fontsize=30)


    ax2.imshow(out_img)
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    ax2.set_xlim(0, 1280)
    ax2.set_ylim(720, 0)
    ax2.set_title('Lane Lines Identified', fontsize=30)
    plt.savefig("output_images/line_fit_example.jpg", bbox_inches='tight')
