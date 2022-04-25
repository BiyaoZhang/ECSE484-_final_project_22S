from PIL import Image, ImageDraw, ImageFont
from random import choice, randint, shuffle, randrange

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

from xml.dom.minidom import Document
import glob

import time

import sys

import os

chrs = ["0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9",
        "a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y",
        "z", "A", "B", "C", "D",
        "E", "F", "G", "H", "I",
        "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "X",
        "Y", "Z",
        "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9",
        "a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y",
        "z", "A", "B", "C", "D",
        "E", "F", "G", "H", "I",
        "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "X",
        "Y", "Z",
        "0", "1", "2", "3", "4",
        "5", "6", "7", "8", "9",
        "a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y",
        "z", "A", "B", "C", "D",
        "E", "F", "G", "H", "I",
        "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "X",
        "Y", "Z"
        ]

# chrs = ["0", "1", "2"]

chrs_len = len(chrs)
chrs_index_set = [i for i in range(chrs_len)]
shuffle(chrs_index_set)
chr_index = 0

font_paths = glob.glob('fonts/*.ttf')
# font_paths = font_paths[1:2]
shuffle(font_paths)
font_paths_len = len(font_paths)
font_paths_index = 0


def get_font_path():
    global font_paths_index, font_paths_len
    if font_paths_index == font_paths_len:
        font_paths_index = 0
        shuffle(font_paths)
    result = font_paths[font_paths_index]
    font_paths_index += 1
    # print(result)
    return result


def selected_chrs(length):
    result = []
    global chr_index, font_paths_index
    for i in range(length):
        if chr_index == chrs_len:
            chr_index = 0
            shuffle(chrs_index_set)
            # if font_paths_index == font_paths_len:
            #     font_paths_index = 0
            #     shuffle(font_paths)
            # else:
            #     font_paths_index += 1
        result.append(chrs[chrs_index_set[chr_index]])
        chr_index += 1
    return result


# def selected_chrs(length):
#     """
#     返回length个随机字符串
#     :param length:
#     :return:
#     """
#     result = ''.join(choice(chrs) for _ in range(length))
#     return result


def get_color():
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return (r, g, b)


def get_new_color(number_of_colors):
    color_set = []
    for i in range(number_of_colors):
        color = get_color()
        while color in color_set:
            color = get_color()
        color_set.append(color)
    return color_set


def get_unzero_pixels_box(image, font_color, background_color):
    if image is None:
        return None, None
    pixels = []
    pixels_temp = image.load()
    size = image.size
    for y in range(size[1]):
        for x in range(size[0]):
            if pixels_temp[x, y] != background_color:
                color = pixels_temp[x, y]
                # color = font_color
                pixels.append([x, y, color[0], color[1], color[2]])

    if len(pixels):
        pixels = np.array(pixels)
        xmin = np.min(pixels[:, 0])
        xmax = np.max(pixels[:, 0])

        ymin = np.min(pixels[:, 1])
        ymax = np.max(pixels[:, 1])
        box = [xmin, ymin, xmax, ymax]
        # box = [xmin - 1, ymin - 1, xmax, ymax]

        # pixels = [x, y, r, g, b]
        # box = [xmin, ymin, xmax, ymax]
        return pixels, box

    return None, None


def get_unzero_margin_pixels_box(image, font_color, background_color):
    if image is None:
        return None, None
    image_temp = cv2.Canny(np.array(image), 50, 80)
    # image_temp = Image.fromarray(image_temp)
    pixels = []
    # pixels_temp = image_temp.load()
    size = image.size
    for y in range(size[1]):
        for x in range(size[0]):
            if image_temp[x, y] != 0:
                color = font_color
                pixels.append([y, x, color[0], color[1], color[2]])

    if len(pixels):
        pixels = np.array(pixels)
        xmin = np.min(pixels[:, 0])
        xmax = np.max(pixels[:, 0])

        ymin = np.min(pixels[:, 1])
        ymax = np.max(pixels[:, 1])
        box = [xmin, ymin, xmax, ymax]
        # box = [xmin - 1, ymin - 1, xmax, ymax]

        # pixels = [x, y, r, g, b]
        # box = [xmin, ymin, xmax, ymax]
        return pixels, box

    return None, None


def cut_image(image, box):
    if image is None:
        return None
    size = (box[2] - box[0] + 1, box[3] - box[1] + 1)
    image_temp = Image.new('RGB', size, (0, 0, 0))
    pixels_temp = image_temp.load()
    pixels_image = image.load()

    for y in range(box[1], box[3] + 1):
        for x in range(box[0], box[2] + 1):
            pixels_temp[int(x - box[0]), int(y - box[1])] = pixels_image[x, y]

    # for y in range(size[1]):
    #     for x in range(size[0]):
    #         pixels_temp[x, y] = pixels_image[int(x + box[1]), int(y + box[2])]

    return image_temp


def get_pixels_image_box(chr_string, font, font_color, background_color, flag_margin, flag_twits, flag_twist_type):
    image_temp = Image.new('RGB', (160, 160), background_color)

    size = image_temp.size

    draw_temp = ImageDraw.Draw(image_temp)

    width, height = draw_temp.textsize(chr_string, font)

    position = ((size[0] - width) // 2, (size[1] - height) // 2)

    draw_temp.text(xy=position, text=chr_string, font=font, fill=font_color, stroke_width=0)

    random_rotate_angle = randint(-20, 20)

    image_temp = image_temp.rotate(random_rotate_angle, expand=0)


    cut_box = [int(size[0] // 4), int(size[1] // 4), int(size[0] // 4 * 3), int(size[1] // 4 * 3)]
    image_temp = cut_image(image_temp, cut_box)


    # flag_margin = randint(0, 1)
    if flag_margin <= 0:
        pixels, box = get_unzero_pixels_box(image_temp, font_color, background_color)
    if flag_margin > 0:
        pixels, box = get_unzero_margin_pixels_box(image_temp, font_color, background_color)

    if pixels is None:
        return None, None

    width = box[2] - box[0]
    height = box[3] - box[1]


    pixels = pixels - [box[0], box[1], 0, 0, 0]
    box = [0, 0, width, height]

    # 扭曲字符
    # flag_twits = randint(0, 1)
    period = randint(3, 5)
    if flag_twits > 0:
        pixels = twist_pixels(image_temp.size, pixels, 2, (period, period), flag_twist_type)

    # pixels = [x, y, r, g, b]
    # box = [xmin, ymin, xmax, ymax]

    return pixels, box


def draw_pxiels_on_image(image, pixels, position):
    if image is None:
        return None
    pixels_image = image.load()
    size = image.size
    if pixels is not None:
        for pixel in pixels:
            x = int(pixel[0] + position[0])
            y = int(pixel[1] + position[1])
            if 0 <= x < size[0] and 0 <= y < size[1]:
                pixels_image[x, y] = (int(pixel[2]), int(pixel[3]), int(pixel[4]))
    return image


def draw_string(numbers_of_chars, font_paths, colors, flag_margin, flag_twits, flag_twist_type):
    pixels_set = []
    box_set = []

    bgcolor = colors[0]

    color_set = colors[1:]

    texts = selected_chrs(numbers_of_chars)

    font_path = get_font_path()
    # font_path = choice(font_paths)
    # print(font_path)
    font = ImageFont.truetype(font_path, randint(35, 45))

    texts_set = []
    d_x_set = []

    for i in range(numbers_of_chars):
        pixels, box = get_pixels_image_box(texts[i], font, choice(color_set), bgcolor, flag_margin, flag_twits,
                                           flag_twist_type)

        if pixels is None:
            return None, None

        pixels_set.append(pixels)
        box_set.append(box)
        texts_set.append(texts[i])
        if i > 0:
            d_x_set.append(randint(-7, 0))
    d_x_set.append(0)

    box_set = np.array(box_set)
    texts_width = np.sum(box_set[:, 2]) + np.sum(d_x_set)
    texts_height = np.max(box_set[:, 3])

    size = (texts_width + 80, texts_height + 30)
    image_temp = Image.new('RGB', size, bgcolor)

    position_x = (size[0] - texts_width) // 2

    for i in range(len(pixels_set)):
        box = box_set[i]
        box_width = box[2]
        position_y = (size[1] - box[3]) // 2
        position = (position_x, position_y + randint(-5, 5))
        image_temp = draw_pxiels_on_image(image_temp, pixels_set[i], position)
        box_set[i] = box_set[i] + [position[0], position[1], position[0], position[1]]
        position_x = position[0] + box_width + d_x_set[i]

    # texts_box = np.array([np.min(box_set[:, 0]), np.min(box_set[:, 1]), np.max(box_set[:, 2]), np.max(box_set[:, 3])])
    texts_box = np.array(
        [np.min(box_set[:, 0]) - 20, np.min(box_set[:, 1]) - 5, np.max(box_set[:, 2]) + 20, np.max(box_set[:, 3]) + 5])
    # width_offset = randint(-2, 2)
    # height_offset = randint(-5, 2)
    # texts_box = texts_box - [width_offset, height_offset, -1 * width_offset, -1 * height_offset]

    image_temp = cut_image(image_temp, texts_box)

    box_set = box_set - [texts_box[0], texts_box[1], texts_box[0], texts_box[1]]

    width_sub_image = texts_box[2] - texts_box[0] + 1
    height_sub_image = texts_box[3] - texts_box[1] + 1

    for box in box_set:
        if box[0] <= 0:
            box[0] = 1
        if box[1] <= 0:
            box[1] = 1
        if box[2] >= width_sub_image:
            box[0] = width_sub_image - 1
        if box[3] >= height_sub_image:
            box[0] = height_sub_image - 1

    boxes = []
    for i in range(len(box_set)):
        box = box_set[i]
        boxes.append([str(box[0]), str(box[1]), str(box[2]), str(box[3]), texts[i]])

    return image_temp, boxes


def get_horizontal_line_pixels(image, background_color, color_set):
    if image is None:
        return None
    size = image.size
    image_temp = Image.new('RGB', size, background_color)
    draw = ImageDraw.Draw(image_temp)

    start = (0, randint(0, size[1]))
    end = (size[0], randint(0, size[1]))

    line_color = choice(color_set)
    draw.line([start, end], fill=line_color, width=randint(2, 3))

    pixels, box = get_unzero_pixels_box(image_temp, line_color, background_color)

    if pixels is None:
        return None

    return pixels


def draw_horizontal_line(image, background_color, color_set):
    if image is None:
        return None
    image_temp = image

    pixels = get_horizontal_line_pixels(image, background_color, color_set)

    if pixels is None:
        return None

    size = image_temp.size

    flag_twits = randint(0, 7)
    if flag_twits > 0:
        flag_twist_type = randint(0, 2)
        period = (int(size[0] // 20), int(size[1] // 20))
        pixels = twist_pixels(size, pixels, 2, period, flag_twist_type)

    image_temp = draw_pxiels_on_image(image_temp, pixels, (0, 0))

    return image_temp


def draw_horizontal_lines(image, unmbers_of_lines, colors):
    if image is None:
        return None

    background_color = colors[0]
    color_set = colors[1:]
    image_temp = image
    for i in range(unmbers_of_lines):
        image_temp = draw_horizontal_line(image_temp, background_color, color_set)
    return image_temp


def draw_vertical_lines(image, number_of_lines, colors):
    if image is None:
        return None

    color_set = colors[1:]
    size = image.size
    image_temp = image
    draw = ImageDraw.Draw(image_temp)
    for i in range(number_of_lines):
        start = (randint(0 + 10, size[0] - 10), randint(0, size[1]))

        end = (start[0] + randint(-10, 10), randint(0, size[1]))

        draw.line([start, end], fill=choice(color_set), width=randint(2, 3))
    return image_temp


def draw_lines(image, colors, flag_draw_lines_type):
    if image is None:
        return None

    flag_draw_lines_type = 2
    if flag_draw_lines_type == 0:
        return_image = draw_horizontal_lines(image, randint(2, 3), colors)
        return return_image

    if flag_draw_lines_type == 1:
        return_image = draw_vertical_lines(image, randint(2, 3), colors)
        return return_image

    if flag_draw_lines_type == 2:
        return_image = draw_horizontal_lines(image, randint(1, 3), colors)
        return_image = draw_vertical_lines(return_image, randint(2, 3), colors)
        return return_image


def draw_nois(image, colors):
    if image is None:
        return None

    color_set = colors[1:]
    size = image.size
    image_temp = image
    draw = ImageDraw.Draw(image_temp)
    for i in range(int(size[0] * size[1] * 0.01 * randint(1, 3))):  # 1~2%密度的干扰像素
        draw.point((randrange(size[0]), randrange(size[1])), fill=choice(color_set))  # randrange取值范围是左开右闭
    return image_temp


def twist_pixels(size, pixels, bais, period, flag_twist_type):
    # period = (p1, p2)
    # flag_twist_type = randint(0, 2)
    if flag_twist_type == 0:
        return_pixels = vertical_twist_pixels(size, pixels, bais, period[0])
        return return_pixels

    if flag_twist_type == 1:
        return_pixels = horizontal_twist_pixels(size, pixels, bais, period[1])
        return return_pixels

    if flag_twist_type == 2:
        return_pixels = vertical_twist_pixels(size, pixels, bais, period[0])
        return_pixels = horizontal_twist_pixels(size, return_pixels, bais, period[1])
        return return_pixels


def vertical_twist_pixels(size, pixels, bais, period):
    return_pixels = []
    if pixels is not None:
        for pixel in pixels:
            x = pixel[0]
            y = pixel[1]
            cor_x = float(x / size[0])
            new_y = y + int(bais * np.sin(cor_x * period * 2 * np.pi))
            return_pixels.append([x, new_y, pixel[2], pixel[3], pixel[4]])
    return return_pixels


def horizontal_twist_pixels(size, pixels, bais, period):
    return_pixels = []
    if pixels is not None:
        for pixel in pixels:
            x = pixel[0]
            y = pixel[1]
            cor_y = float(y / size[1])
            new_x = x + int(bais * np.sin(cor_y * period * 2 * np.pi))
            return_pixels.append([new_x, y, pixel[2], pixel[3], pixel[4]])
    return return_pixels


def show_image_and_boxes(image, boxes):
    plt.imshow(np.array(image))
    for box in boxes:
        rect = Rectangle((int(box[0]), int(box[1])), (int(box[2]) - int(box[0])), (int(box[3]) - int(box[1])),
                         fill=False, color='red')
        ax = plt.gca()
        ax.axes.add_patch(rect)
    plt.show()


def creat_captcha(numbers_of_chr, flag_add_interference):

    flag_colors = randint(0, 9)
    if flag_colors == 0:
        colors = [(0, 0, 0), (255, 255, 255)]
    if flag_colors == 1:
        colors = [(255, 255, 255), (0, 0, 0)]
    if flag_colors in [2, 3, 4]:
        colors = get_new_color(numbers_of_chr + 1)
    if flag_colors in [5, 6, 7, 8, 9]:
        colors = get_new_color(2)

    flag_margin = randint(-3, 1)
    flag_twits = randint(-1, 1)

    if flag_add_interference > 0:
        flag_add_lines = 1
        flag_add_noise = 1
    else:
        flag_add_lines = 0
        flag_add_noise = 0

    image, boxes = draw_string(numbers_of_chr, font_paths, colors, flag_margin, flag_twits, randint(0, 2))

    if image is None:
        return None, None

    if flag_add_lines > 0:
        image = draw_lines(image, colors, randint(0, 2))
        if image is None:
            return None, None

    if flag_add_noise > 0:
        image = draw_nois(image, colors)
        if image is None:
            return None, None

    return image, boxes


def creat_xml(fname, boxes, image):
    img_size = image.size
    doc = Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)

    folder = doc.createElement("folder")
    folder_text = doc.createTextNode("")
    folder.appendChild(folder_text)
    annotation.appendChild(folder)

    filename = doc.createElement("filename")
    filename_text = doc.createTextNode(fname)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    path = doc.createElement("path")
    path_text = doc.createTextNode("")
    path.appendChild(path_text)
    annotation.appendChild(path)

    source = doc.createElement("source")
    database = doc.createElement("database")
    database_text = doc.createTextNode("Unknown")
    database.appendChild(database_text)
    source.appendChild(database)
    annotation.appendChild(source)

    size = doc.createElement("size")

    width = doc.createElement("width")
    width_text = doc.createTextNode(str(img_size[0]))
    width.appendChild(width_text)
    size.appendChild(width)

    height = doc.createElement("height")
    height_text = doc.createTextNode(str(img_size[1]))
    height.appendChild(height_text)
    size.appendChild(height)

    depth = doc.createElement("depth")
    depth_text = doc.createTextNode("3")
    depth.appendChild(depth_text)
    size.appendChild(depth)

    annotation.appendChild(size)

    segmented = doc.createElement("segmented")
    segmented_text = doc.createTextNode("0")
    segmented.appendChild(segmented_text)
    annotation.appendChild(segmented)

    for box in boxes:
        object_box = doc.createElement("object")

        name = doc.createElement("name")
        name_text = doc.createTextNode(box[4])
        name.appendChild(name_text)
        object_box.appendChild(name)

        pose = doc.createElement("pose")
        pose_text = doc.createTextNode("Unspecified")
        pose.appendChild(pose_text)
        object_box.appendChild(pose)

        truncated = doc.createElement("truncated")
        truncated_text = doc.createTextNode("0")
        truncated.appendChild(truncated_text)
        object_box.appendChild(truncated)

        difficult = doc.createElement("difficult")
        difficult_text = doc.createTextNode("0")
        difficult.appendChild(difficult_text)
        object_box.appendChild(difficult)

        bndbox = doc.createElement("bndbox")

        xmin = doc.createElement("xmin")
        xmin_text = doc.createTextNode(box[0])
        xmin.appendChild(xmin_text)
        bndbox.appendChild(xmin)

        ymin = doc.createElement("ymin")
        ymin_text = doc.createTextNode(box[1])
        ymin.appendChild(ymin_text)
        bndbox.appendChild(ymin)

        xmax = doc.createElement("xmax")
        xmax_text = doc.createTextNode(box[2])
        xmax.appendChild(xmax_text)
        bndbox.appendChild(xmax)

        ymax = doc.createElement("ymax")
        ymax_text = doc.createTextNode(box[3])
        ymax.appendChild(ymax_text)
        bndbox.appendChild(ymax)

        object_box.appendChild(bndbox)
        annotation.appendChild(object_box)

    return doc


def save_image_and_xml(save_path, save_name, boxes, image):
    image_save_path = save_path + save_name + ".jpg"
    xml_save_path = save_path + save_name + ".xml"
    image.save(image_save_path)
    doc = creat_xml(save_name + ".jpg", boxes, image)
    with open(xml_save_path, "w", encoding="utf-8") as f:
        doc.writexml(f, addindent='\t', newl='\n', encoding='utf-8')


def creat_samples(save_path, num_of_samples, flag_add_interference):
    print("\ncreat " + str(num_of_samples) + " samples to folder: " + save_path)
    i = 0
    while i < num_of_samples:
        image, boxes = creat_captcha(randint(4, 7), flag_add_interference)
        if image is not None:
            t = time.time()
            save_name = str(round(t * 1000000)) + str(randint(0, 9))
            save_image_and_xml(save_path, save_name, boxes, image)
            sys.stdout.write("\r progress: %d / %d" % (i + 1, num_of_samples))
            sys.stdout.flush()
            i = i + 1
        else:
            continue


if __name__ == '__main__':
    # SAVE_PATH_TRAIN = "samples/train/"
    # creat_samples(SAVE_PATH_TRAIN, 100000)
    #
    # SAVE_PATH_TRAIN = "samples/test/"
    # creat_samples(SAVE_PATH_TRAIN, 4000)

    # image, boxes = creat_captcha(32, 0)
    # show_image_and_boxes(image, boxes)

    ratio_of_interference = 0.6

    # generate samples to folder samples/train/
    number_of_samples_for_train = 10
    SAVE_PATH_TRAIN = "samples/train/"
    if not os.path.exists(SAVE_PATH_TRAIN):
        os.makedirs(SAVE_PATH_TRAIN)
    creat_samples(SAVE_PATH_TRAIN, int(number_of_samples_for_train - (number_of_samples_for_train * ratio_of_interference)), 0)
    creat_samples(SAVE_PATH_TRAIN, int(number_of_samples_for_train * ratio_of_interference), 1)

    # generate samples to folder samples/test/
    number_of_samples_for_test = 10
    SAVE_PATH_TEST = "samples/test/"
    if not os.path.exists(SAVE_PATH_TEST):
        os.makedirs(SAVE_PATH_TEST)
    creat_samples(SAVE_PATH_TEST, int(number_of_samples_for_test - (number_of_samples_for_test * ratio_of_interference)), 0)
    creat_samples(SAVE_PATH_TEST, int(number_of_samples_for_test * ratio_of_interference), 1)
