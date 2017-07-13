import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from binary_threshold import binary_threshold
from camera_calibration import undistort
from perspective_transform import perspective_transform
from hist_poly_fit import hist_poly_fit
import glob


def draw_lanes(img, left_fit, right_fit, lanes_img):

    M = pickle.load(open("perspective_transform.p", 'rb'))
    Minv = np.linalg.inv(M)

    undistorted = undistort(img)
    thresholded = binary_threshold(undistorted)
    transformed = perspective_transform(thresholded)

    ploty = np.linspace(0, transformed.shape[0] - 1, transformed.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(transformed).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))



    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped = cv2.warpPerspective(color_warp, Minv, (transformed.shape[1], transformed.shape[0]))
    # Combine the result with the original image


    result = cv2.addWeighted(undistorted, 1, unwarped, 0.3, 0)

    lanes_img_unwarped = cv2.warpPerspective(lanes_img, Minv, (transformed.shape[1], transformed.shape[0]))
    result = cv2.addWeighted(lanes_img_unwarped, 1, result, 0.8, 0)

    return result



def draw_text(img, left_fit, right_fit, leftx_base, rightx_base):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    y_eval = img.shape[0]

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])

    # Now our radius of curvature is in meters
    average_curve_rad = 0.5*(left_curverad + right_curverad)
    rad_string = "Radius of Curvature = {0:.0f}m".format(average_curve_rad)
    cv2.putText(img, rad_string, org=(50, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255), thickness=3)

    # Calculate distance from center:
    distance_right_of_center = xm_per_pix * 0.5*(rightx_base + leftx_base - img.shape[1])
    dist_string = "Veichle is {0:.1f}m right of center".format(distance_right_of_center)
    cv2.putText(img, dist_string, org=(50, 160), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)

    return img

def process_image(img):
    undistorted = undistort(img)
    thresholded = binary_threshold(undistorted)
    transformed = perspective_transform(thresholded)
    left_fit, right_fit, leftx_base, rightx_base, out_img = hist_poly_fit(transformed)
    processed_image = draw_lanes(img, left_fit, right_fit, out_img)
    draw_text(processed_image, left_fit, right_fit, leftx_base, rightx_base)
    return processed_image

if __name__ == "__main__":
    # See what is happening to the test images (useful in debugging)
    image_urls = glob.glob('test_images/*.jpg')

    for url in image_urls:
        img = mpimg.imread(url)
        plt.imshow(process_image(img))
        plt.show()

    img = mpimg.imread("test_images/test5.jpg")
    plt.imshow(process_image(img))
    plt.title("Image with lane lines, radius of curvature and distance from center")
    plt.savefig("output_images/pipeline_example.jpg", bbox_inches='tight')