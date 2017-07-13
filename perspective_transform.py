# Xavier O'Rourke 29 June 2017

# This script uses image of straight lines on the road to find a perspective
# transform matrix. This matrix can be used to transform images of the road to
# "bird's eye view"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from camera_calibration import undistort
import os


def get_perspective_transform():
    """  Calculate the perspective transorm matrix to convert camera images to
        'birds-eye view' and save this matrix to the file 'perspective_transform.p'
    """
    img = mpimg.imread("test_images/straight_lines1.jpg")
    img = undistort(img)
    img_size = (img.shape[1], img.shape[0])

    # Specify source points to use in perspective transform
    src = np.float32(
        [[195, 720], # Bottom left
         [1120, 720], # Bottom right
         [687, 450],  # Top right
         [594, 450]] # Top left
    )

    x_offset_l = 200
    x_offset_r = 300
    y_offset = 0

    # Specify desitnation points to use in perspective transform
    dst = np.float32([
        [x_offset_l, img_size[1] - y_offset],  # Bottom Left
        [img_size[0] - x_offset_r, img_size[1] - y_offset],  # Bottom Right
        [img_size[0] - x_offset_r, y_offset],  # Top Right
        [x_offset_l, y_offset] # Top Left
        ])


    # Save the perspective transform matrix for later use
    M = cv2.getPerspectiveTransform(src, dst)
    pickle.dump(M, open("perspective_transform.p", 'wb'))

    if __name__ == '__main__':
        # Plot the original and perspective-transformed images, along with points
        # used to define a rectangle

        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        x_cords = np.hstack((src[:, 0], src[0, 0]))
        y_cords = np.hstack((src[:, 1], src[0, 1]))
        ax1.plot(x_cords, y_cords, 'g-', linewidth=3)
        ax1.set_title('Original Image', fontsize=30)

        ax2.imshow(warped)
        x_cords = np.hstack((dst[:, 0], dst[0, 0]))
        y_cords = np.hstack((dst[:, 1], dst[0, 1]))
        ax2.plot(x_cords, y_cords, 'g-', linewidth=3)
        ax2.set_title('Perspective Transform', fontsize=30)
        plt.savefig("output_images/perspective_example.jpg", bbox_inches='tight')
        plt.show()



def perspective_transform(img):
    """Transform the image to birds-eye view using the perspective transform matrix"""

    if not os.path.isfile("perspective_transform.p"):
        get_perspective_transform() # Calibrate camera if we haven't already

    M = pickle.load(open("perspective_transform.p", 'rb'))
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

if __name__ == '__main__':
    get_perspective_transform()

