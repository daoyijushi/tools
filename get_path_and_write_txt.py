# -*-encoding: utf-8 -*-
import os
import random
import glob


# 创建
def txt(path, label_list):
    if not os.path.exists(path):
        os.makedirs(path)
    txt_name = path + '/' + 'nametum.txt'
    f = open(txt_name, 'w')
    for line in label_list:
        f.write(line + '\n')
    f.close()

# 获取路径
def get_path(input_path):
    # 创建文件夹
    name_list = list()
    for root, dirs, files in os.walk(input_path):
        for dir in dirs:
            path = input_path + '/' + dir
            imgs = glob.glob(path + "/*.dcm")
            for img in imgs:
                name_list.append(img)
    return name_list


if __name__ == '__main__':
    input = '/Users/jingyi/Desktop/data_60'
    path = '/Users/jingyi/Desktop/'
    label_list = get_path(input)
    txt(path, label_list)
    num = 1
    print num



