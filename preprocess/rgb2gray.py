import os
import numpy as np
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

png2rgb_dict = {
    0: (0, 0, 0),  # background
    1: (0, 128, 255),  # hat
    2: (255, 255, 0),  # hair      rgb
    5: (255, 0, 128),  # upper clothes
    6: (51, 51, 204),  # dress
    7: (255, 0, 0),  # coat      rgb
    9: (0, 255, 0),  # pants
    12: (255, 64, 191),  # skirt
    13: (128, 128, 204),  # face
    14: (128, 64, 0),  # left_arm
    15: (64, 0, 128),  # right_arm
    16: (0, 64, 194),  # left_leg
    17: (192, 128, 64),  # right_leg
    18: (255, 0, 255),  # left_shoe
    19: (0, 0, 255),  # right_shoe
    20: (0, 255, 255),  # shorts    rgb
}


# 通过比较颜色的曼哈顿距离，将杂色转成较近的类别
def map_to_category(label: np.ndarray):
    def manhattan_dis(p1: tuple, p2: tuple):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
        # return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2

    color_map = png2rgb_dict.values()
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            color = tuple(label[i][j].tolist())
            # print(color)
            if color not in color_map:
                min_dis = 255 * 3
                idea_color = color
                for c in color_map:
                    d = manhattan_dis(color, c)
                    if d < min_dis:
                        min_dis = d
                        idea_color = c
                # if idea_color == (0, 64, 128) or idea_color == (128, 0, 64) or idea_color == (128, 0, 255):
                #     print(f"{color} change to{idea_color}")
                #     idea_color = (255, 255, 255)
                label[i][j] = np.array(idea_color)
    return label


# png转为rgb
def png2rgb(mask: np.ndarray):
    # 统计一下有多少类
    category = set()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] not in category:
                category.add(mask[i][j])

    # 将png灰度标签图转成rgb图
    visual_label = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color = png2rgb_dict[mask[i, j]]

            visual_label[i, j, 0] = color[0]
            visual_label[i, j, 1] = color[1]
            visual_label[i, j, 2] = color[2]
    return visual_label, category


# 将rgb图转为png灰度图
def rgb2png(label: np.ndarray):
    label_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.int8)
    for key, value in png2rgb_dict.items():
        # print(key, value)
        # 注意这里value是元组，需要转成ndarray数组
        locations = np.all(label == np.array(value), axis=-1)
        # print(locations)
        # print(value)
        label_mask[locations] = key
    # print(label_mask[100])
    return label_mask


def transform_label(labels_path: str, target_path: str, labels_name: list):
    start_t = time.time()

    if not os.path.exists(labels_path):
        print("invalid path")
        return

    # print(label_names)
    for label_name in tqdm(labels_name):
        label = cv2.imread(os.path.join(labels_path, label_name), cv2.IMREAD_UNCHANGED)

        # 去除一些杂色，很少了已经，还有将颜色调整好，因为渲染出来的颜色有些许偏差
        map_to_category(label)

        label_mask = rgb2png(label)

        label_true_name = label_name.split('.')[0] + '.png'
        cv2.imwrite(os.path.join(target_path, label_true_name), label_mask)

    return round(time.time() - start_t, 2)


def transform_rgb2png():
    label_path = r'F:\Datasets\SinglePerson\final\original_labels'
    target_path = r'F:\Datasets\SinglePerson\final\labels_mapping'

    values = list(png2rgb_dict.values())
    for i in range(len(values)):
        values[i] = list(values[i])
    print(values)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    labels_name = os.listdir(label_path)

    # ==================多进程版本==================
    process_num = 12
    pool = Pool(12)
    results = []
    each_process_pnum = []
    num_per_process = len(labels_name) // process_num
    for i in range(process_num):
        print('process ' + str(i) + ' start work...')
        s = i * num_per_process
        if i == process_num - 1:
            e = len(labels_name)
        else:
            e = s + num_per_process
        each_process_pnum.append(e - s)
        res = pool.apply_async(func=transform_label, args=(label_path, target_path, labels_name[s:e]))
        results.append(res)

    pool.close()
    pool.join()
    for i in range(len(results)):
        print('process {} processed num: {}, processed time: {}s'.format(i, each_process_pnum[i],
                                                                         round(results[i].get(), 2)))



if __name__ == '__main__':
    transform_rgb2png()
