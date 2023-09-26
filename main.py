import cv2
import numpy as np


def nearest_neighbour_interpolation(image, new_height, new_width):
    if input_image is None:
        print("Error: Failed to load the input image.")
        return None
    # channels := the number of color channels in the image
    height, width, channels = image.shape

    # initialized as a NumPy array filled with zeros.
    # This line of code creates a blank image with the same number of channels
    output = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    # each pixel value in the image should be represented as an 8-bit unsigned integer
    for i in range(new_height):
        for j in range(new_width):
            y, x = i * height / new_height, j * width / new_width
            y, x = int(round(y)), int(round(x))
            output[i, j] = image[y, x]

    return output


def bi_linear_interpolation(image, new_height, new_width):
    if input_image is None:
        print("Error: Failed to load the input image.")
        return None
    height, width, channels = image.shape
    output = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    # x2,y2 are neighboring pixels
    # x1 + 1 := next pixel column to the right   y1+1:= next pixel row below
    # min(int(y) + 1, width - 1):=to make sure that we don't go beyond the boundaries of the image
    for i in range(new_height):
        for j in range(new_width):
            y, x = i * height / new_height, j * width / new_width
            y1, y2 = int(y), min(int(y) + 1, height - 1)
            x1, x2 = int(x), min(int(x) + 1, width - 1)

            dy, dx = y - y1, x - x1

            for c in range(channels):
                output[i, j, c] = (1 - dx) * (1 - dy) * image[y1, x1, c] + \
                                  dx * (1 - dy) * image[y1, x2, c] + \
                                  (1 - dx) * dy * image[y2, x1, c] + \
                                  dx * dy * image[y2, x2, c]
                # (1 - dx) * (1 - dy) * image[y1, x1, c]:= the contribution of the top-left neighboring pixel
                # dx * (1 - dy) * image[y1, x2, c] := contribution of the top-right neighboring pixel
                # (1 - dx) * dy * image[y2, x1, c] := contribution of the bottom-left neighboring pixel
                # dx * dy * image[y2, x2, c] :=  contribution of the bottom-right neighboring pixel
    return output


# read the image
input_image = cv2.imread('C:\\Users\\User\\Downloads\\Hibiscus.jpg')

# Define new dimensions which need to resample
new_height = 200
new_width = 400

#  nearest neighbor interpolation
nearst_neighbour_output = nearest_neighbour_interpolation(input_image, new_height, new_width)

# bi linear interpolation
bi_linear_output = bi_linear_interpolation(input_image, new_height, new_width)

# Save results in the folder
cv2.imwrite('nearest_neighbor_output.jpg', nearst_neighbour_output)
cv2.imwrite('bi_linear_output.jpg', bi_linear_output)
