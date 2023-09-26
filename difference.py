from skimage import io, color, metrics

# Load the output images
nearest_neighbour_image = io.imread('nearest_neighbor_output.jpg')
bi_linear_image = io.imread('bi_linear_output.jpg')

# Convert images to grayscale (SSIM works on grayscale images)
nearest_neighbour_gray = color.rgb2gray(nearest_neighbour_image)
bi_linear_gray = color.rgb2gray(bi_linear_image)

# Calculate SSIM with data_range specified (range is [0, 1])
ssim_score = metrics.structural_similarity(nearest_neighbour_gray, bi_linear_gray, data_range=1)

print(f"SSIM Score: {ssim_score}")

# Compare the SSIM score
if ssim_score > 0.5:
    print("Bi linear interpolation produces better results.")
else:
    print("Nearest neighbour interpolation produces better results.")
