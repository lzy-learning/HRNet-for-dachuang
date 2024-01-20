import os
import random


def generate_img_list():
    # train_img_list = r'F:/Datasets/SinglePerson/final/trainList.txt'
    # val_img_list = r'F:/Datasets/SinglePerson/final/valList.txt'
    # train_root = r'F:/Datasets/SinglePerson/final/images'
    # train_label_root = r'F:/Datasets/SinglePerson/final/labels_mapping_shorts2pants'

    train_img_list = r'F:/Datasets/genBySd/trainList.txt'
    val_img_list = r'F:/Datasets/genBySd/valList.txt'
    train_root = r'F:/Datasets/genBySd/src_img'
    train_label_root = r'F:/Datasets/genBySd/labels_mapping_shorts2pants'
    img_list = os.listdir(train_root)
    with open(train_img_list, mode='w') as fp:
        for img_name in img_list:
            train_path = os.path.join(train_root, img_name)
            train_label_path = os.path.join(train_label_root, img_name).replace('.jpg', '.png')
            fp.write(train_path + ' ' + train_label_path + ' ' + img_name.replace('.jpg', '') + '\n')

    img_list = random.sample(img_list, int(len(img_list) * 0.3))

    with open(val_img_list, mode='w') as fp:
        for img_name in img_list:
            val_path = os.path.join(train_root, img_name)
            val_label_path = os.path.join(train_label_root, img_name).replace('.jpg', '.png')
            fp.write(val_path + ' ' + val_label_path + ' ' + img_name.replace('.jpg', '') + '\n')

    img_list = [line.strip().split() for line in open(train_img_list)]
    for item in img_list:
        if 'train' in 'trainList.txt':
            image_path, label_path, _ = item
            print(image_path, label_path)
        break


if __name__ == '__main__':
    generate_img_list()