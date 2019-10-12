from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import shutil, os
import argparse


# Run K-Means algorithm
def quantize(img: Image, color_level: int, iteration_level: int):
    # Get the raw data and attributes of image
    img_raw_data = np.array(img)
    img_raw_height, img_raw_width, img_raw_depth = tuple(img_raw_data.shape)

    # Make raw data processable
    # img_data_arr represents the image, rgb_center_arr represents the centroids
    img_data_arr = np.reshape(img_raw_data, (img_raw_height * img_raw_width, img_raw_depth))
    rgb_center_arr = np.array(centers)
    rgb_diff_min = np.array([0] * img_raw_width * img_raw_height)

    # K-Means iterations
    for iteration in range(iteration_level):
        rgb_diff_arr = []

        # For every centroid, find the distance of every pixel to its RGB color
        for rgb_color in rgb_center_arr:
            rgb_diff_arr.append(euclidean_distance(img_data_arr, rgb_color))

        # Turn the list into an np-array
        rgb_diff_arr = np.array(rgb_diff_arr)

        # For every pixel, find the minimum centroid-distance
        rgb_diff_min = np.argmin(rgb_diff_arr, axis=0)

        # Reassign the centers, calculating the means of clusters
        for level in range(color_level):
            # Detect the clusters
            cluster = img_data_arr[level == rgb_diff_min]
            cluster_size = len(cluster)

            # If the cluster size is not zero, find new centers
            if not cluster_size == 0:
                rgb_center_arr[level] = np.average(cluster, axis=0)

    # Create the result image using the centroids we have
    # Since we will have only the colors of centroids, we can create the new image out of it
    result_img_arr = np.reshape(rgb_center_arr[rgb_diff_min], (img_raw_height, img_raw_width, -1)).astype('uint8')

    # Convert the array into Image file
    result_img = Image.fromarray(result_img_arr, 'RGB')

    # If the output folder does not exist, create it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Display the image
    if input_mode == 1:
        plt.imshow(result_img)

    # Save the image into output path
    result_img.save(output_name, output_format)


# Calculate the euclidean distance to given color for every pixel
def euclidean_distance(data, color):
    arr_sub = np.subtract(data, color)
    arr_pow = np.power(arr_sub, 2)
    euclidean_sum = np.sum(arr_pow, axis=1)

    return euclidean_sum


# Set command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', required=True, help="Input image path")
ap.add_argument('-m', '--mode', required=True, help="Run-mode, explained as in README")
ap.add_argument('-s', '--size', required=True, help="Size of k-means centers")
ap.add_argument('-i', '--iteration', required=False, default=10, help="Number of iterations, default is 10")

# Get terminal args, parse it
args = vars(ap.parse_args())
input_image = args['file']
input_mode = int(args['mode'])
input_size = int(args['size'])
input_iteration = int(args['iteration'])

# Define output path and format
output_dir = './output/'
output_name = "./output/output_image_" + str(input_size) + "_centers" + ".jpg"
output_format = "JPEG"

# Read and open input image
im = Image.open(input_image)
pix = im.load()
plt.imshow(im)

# Select centers based on run-mode
if input_mode == 1:
    # Centers by selection
    points = plt.ginput(input_size, show_clicks=True)
    centers = [pix[point_x, point_y] for point_x, point_y in points]
elif input_mode == 2:
    # Centers by uniform random
    centers = np.random.uniform(0, 256, (input_size, 3))
else:
    # Wrong mode selection, exit the script
    print("Erroneous mode selection")
    exit(1)

# Run the algorithm
quantize(im, input_size, input_iteration)
