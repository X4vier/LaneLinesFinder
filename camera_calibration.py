import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os

# Calibrating camera using images of 9x6 chess boards

def callibrate_camera():
    """
        Calibrate the camera using images of 9x6 chessboards and save the distortion
        matrix and coefficients for later use.
    """

    # Prepare object (real world) points, like (0,0,0), (1,0,0), ... (6,9,0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store the object points and image points from all the images
    objpoints = [] # 3d points
    imgpoints = [] # 2d points

    image_urls = glob.glob('camera_cal/*.jpg')

    for idx, url in enumerate(image_urls):
        img = cv2.imread(url)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add points to lists to use for calibration
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    img = cv2.imread(image_urls[0])
    img_size = img.shape[0], img.shape[1]

    # Calibrate the camera given these object points and image points
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration values for later use
    distortion_dict = {"dist": dist, "mtx": mtx}
    pickle.dump(distortion_dict, open("distortion.p", 'wb'))

def undistort(img):
    """ Undistorts the image using saved matrix and distortion coefficient """

    if not os.path.isfile("distortion.p"):
        callibrate_camera() # Calibrate camera if we haven't already

    distortion_dict = pickle.load(open("distortion.p", 'rb'))
    img = cv2.undistort(img, distortion_dict['mtx'], distortion_dict['dist'], None, None)
    return img

if __name__ == '__main__':
    callibrate_camera()

    # Plot an undistorted image to see that the correction is working properly
    img = cv2.imread("camera_cal/calibration4.jpg")
    distortion_dict = pickle.load(open("distortion.p", 'rb'))
    undist = cv2.undistort(img, distortion_dict['mtx'], distortion_dict['dist'], None, None)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.savefig("output_images/distortion_example.jpg", bbox_inches='tight')
    plt.show()
