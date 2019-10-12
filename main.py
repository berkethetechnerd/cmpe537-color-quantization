from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


def quantize(img: Image, color_level: int, iteration_level: int):
    img_raw_data = np.array(img)
    img_raw_height, img_raw_width, img_raw_depth = tuple(img_raw_data.shape)

    img_data_arr = np.reshape(img_raw_data, (img_raw_height * img_raw_width, img_raw_depth))
    rgb_center_arr = np.array([pix[point_x, point_y] for point_x, point_y in points])
    rgb_diff_min = np.array([0] * img_raw_width * img_raw_height)

    for iteration in range(iteration_level):
        rgb_diff_arr = []
        for rgb_color in rgb_center_arr:
            rgb_diff_arr.append(euclidean_distance(img_data_arr, rgb_color))

        rgb_diff_arr = np.array(rgb_diff_arr)
        rgb_diff_min = np.argmin(rgb_diff_arr, axis=0)

        for level in range(color_level):
            rgb_center_arr[level] = np.mean(img_data_arr[level == rgb_diff_min], axis=0)

    result_img_arr = np.reshape(rgb_center_arr[rgb_diff_min], (img_raw_height, img_raw_width, -1)).astype('uint8')
    result_img = Image.fromarray(result_img_arr, 'RGB')

    output_name = "./image1_" + str(color_level) + ".jpg"
    output_format = "JPEG"

    plt.imshow(result_img)
    result_img.save(output_name, output_format)


def euclidean_distance(data, color):
    arr_sub = np.subtract(data, color)
    arr_pow = np.power(arr_sub, 2)
    euclidean_sum = np.sum(arr_pow, axis=1)

    return euclidean_sum


im = Image.open("./data/1.jpg")
pix = im.load()
plt.imshow(im)

points = plt.ginput(2, show_clicks=True)
quantize(im, 2, 10)
