import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import random

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


def png2rgb(mask: np.ndarray):
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


if __name__ == '__main__':
    real_img_path = r'F:\Datasets\SinglePerson\final\images'
    real_label_path = r'F:\Datasets\SinglePerson\final\labels_mapping_shorts2pants'

    syn_img_path = r'F:\Datasets\genBySd\src_img'
    syn_label_path = r'F:\Datasets\genBySd\labels_mapping_shorts2pants'
    syn_filter_label_path = r'F:\Datasets\genBySd\filter_masks'

    save_path = r'F:\Datasets\filter_compare'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    k = 0

    syn_img_list = random.sample(os.listdir(syn_img_path), 8)
    print(syn_img_list)
    for syn_img_name in syn_img_list:
        syn_label_name = syn_img_name.replace('.jpg', '.png')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        syn_img = cv2.imread(os.path.join(syn_img_path, syn_img_name))
        syn_img = cv2.cvtColor(syn_img, cv2.COLOR_BGR2RGB)
        syn_label = cv2.imread(os.path.join(syn_label_path, syn_label_name), cv2.IMREAD_GRAYSCALE)
        syn_filter_label = cv2.imread(os.path.join(syn_filter_label_path, syn_label_name), cv2.IMREAD_GRAYSCALE)

        syn_label, _ = png2rgb(syn_label)
        syn_filter_label, _ = png2rgb(syn_filter_label)

        # print(syn_img.shape)
        # print(syn_label.shape)
        # print(syn_filter_label.shape)

        syn_filter_label = cv2.resize(syn_filter_label, dsize=(200,350))

        axes[0].imshow(syn_img)
        axes[1].imshow(syn_label)
        axes[2].imshow(syn_filter_label)

        for ax in axes:
            ax.axis('off')
        plt.savefig(os.path.join(save_path, str(k)+'.png'))
        k += 1
    plt.show()




