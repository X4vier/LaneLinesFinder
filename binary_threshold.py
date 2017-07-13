import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """Selects pixels whose gradient direction is within a certain range"""

    # Calculate gradient direction
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    # Apply threshold
    dir_binary = np.zeros_like(direction)
    dir_binary[(thresh[0] <= direction) & (direction <= thresh[1])] = 1
    return dir_binary

def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """Selects pixels whose gradient in one direction is within a certain range"""

    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    abs_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    grad_binary = np.zeros_like(sobel)
    grad_binary[(thresh[0] <= abs_sobel) & (abs_sobel <= thresh[1])] = 1
    return grad_binary

def saturation_threshold(img, thresh=(0,255)):
    """Selects pixels whose HLS saturation is within a certain range"""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_chanel = hls[:, :, 2]
    s_binary = np.zeros_like(s_chanel)
    s_binary[(thresh[0] < s_chanel) & (s_chanel <= thresh[1])] = 1
    return s_binary

def red_threshold(img, thresh=(0,255)):
    """Selects pixels whose r value is within a certain range"""
    r_chanel = img[:, :, 0]
    r_binary = np.zeros_like(r_chanel)
    r_binary[(thresh[0] <= r_chanel) & (r_chanel <= thresh[1])] = 1
    return r_binary

def mask_binary(img, points):
    """Selects pixels within the polygon described by 'points'"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.ones_like(gray)
    vertices = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 0)
    return mask

def binary_threshold(img, color = False):

    # Apply each of the thresholding functions
    s_binary = saturation_threshold(img, thresh=(170, 190))
    r_binary = red_threshold(img, thresh=(230, 255))
    xgrad_binary = abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(20, 255))

    # Apply mask
    points = np.array([[370, 720], [910, 720], [640, 500]])
    mask = mask_binary(img, points)

    if color:
        color_binary = np.dstack((r_binary, xgrad_binary, s_binary))
        return color_binary

    combined_binary = np.zeros_like(r_binary)
    combined_binary[((r_binary == 1) | (xgrad_binary == 1) | (s_binary == 1)) & (mask != 0)] = 1

    return combined_binary

if __name__ == "__main__":
    # See what is happening to the test images (useful in debugging)
    image_urls = glob.glob('test_images/*.jpg')

    for url in image_urls:
        img = mpimg.imread(url)

        plt.imshow(binary_threshold(img), cmap='gray')
        plt.show()

    # Save an example color image
    img = mpimg.imread("test_images/test5.jpg")
    color_binary = binary_threshold(img, color=True)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Original Image')
    ax1.imshow(img)
    ax2.set_title('Binary Thresholds')
    ax2.imshow(color_binary)
    plt.savefig("output_images/threshold_example.jpg", bbox_inches='tight')
    plt.close()


    # Apply mask
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    points = np.array([[370, 720], [910, 720], [640, 500]])
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Original Image')
    ax1.imshow(img)

    ax2.set_title('Masked Image')
    mask = mask_binary(img, points)
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    ax2.imshow(masked_image)
    plt.savefig("output_images/mask_example.jpg", bbox_inches='tight')
    plt.close()
