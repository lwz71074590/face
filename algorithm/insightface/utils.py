"""Some utils functions
"""
import os
import glob
import numpy as np
import cv2
import mxnet as mx
import sys


def load_dataset(path, model, force_reload=False, save_intermediate_result=True, suffix='jpg'):
    """从数据文件夹中加载数据

    Args:
        path (str): 人脸库文件夹的绝对路径。路径下每个人的照片保存到以名字命名的文件夹中, 图片以jpg的格式存储
        model (FaceModel): deploy.face_model.FaceModel object
        force_reload (bool, optional): Defaults to False. 是否强制重新加载
        save_intermediate_result (bool, optional): Defaults to False. 是否保存生成的向量到embedding_matrix.npy。保存生成的向量可以提高下次加载模型的速度

    Returns:
        [feature_matrix]: 数据库的特征矩阵
        [index2name]: index 到 name 的映射
        [error_list]: 加载错误图片路径的列表
    """

    database = {}
    error_list = []  # 记录出错的图片路径

    categories = os.listdir(path)
    for category in categories:
        intermediate_result = os.path.join(
            path, category, 'embedding_matrix.npy')
        error_file = os.path.join(path, category, 'error_list.txt')

        # 从中间结果加载
        if os.path.exists(intermediate_result) and not force_reload and os.path.exists(error_file):
            embedding_matrix = np.load(intermediate_result)
            with open(error_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    error_list.append(os.path.join(path, category, line))
            # end if
        # 从原始图片加载
        elif os.path.exists(error_file) and not force_reload:
            with open(error_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    error_list.append(os.path.join(path, category, line))
            continue
        else:
            image_files = glob.iglob(
                os.path.join(path, category, '*.' + suffix))
            image_list = []
            error_this_category = []
            for image_file in image_files:  # 遍历类别下的所有照片
                img = cv2.imread(image_file)
                img = cv2.resize(img, (720, 720))
                img, boxes, _, flag = model.get_input(img)

                # 照片中没有检测到人脸，或者检测到两张人脸，跳过该照片并记录错误信息
                if flag == False:
                    print("Image has no face detected. %s." % image_file)
                    error_this_category.append(os.path.basename(image_file))
                    error_list.append(image_file)
                    continue
                
                # 检测到多张人脸
                if img.shape[0] > 1:
                    valid_boxes = list(filter(lambda x: x[1][3] - x[1][1] > 50 and x[1][2] - x[1][0] > 50, enumerate(boxes)))
                    if len(valid_boxes) == 1:
                        img = np.expand_dims(img[valid_boxes[0][0]], axis=0)
                    else:
                        print("Detect more than one face in this image %s." % image_file)
                        error_this_category.append(os.path.basename(image_file))
                        error_list.append(image_file)
                        continue
                
                image_list.append(img)
                # end for loop

            # 将出错图片记录下来
            with open(error_file, 'w') as f:
                for error in error_this_category:
                    f.write(error + '\n')

            if len(image_list) == 0:  # 如果整个文件夹都没检测到人脸
                print("Invalid category %s, no face has been detect." % category)
                continue

            batch_image = np.concatenate(image_list)
            embedding_matrix = model.get_feature(batch_image)

            if save_intermediate_result:
                np.save(intermediate_result, embedding_matrix)
            # end else

        database[category] = embedding_matrix

    index2name = list()
    # 建立index 到 name 的映射
    feature_list = list()
    for name, images in database.items():
        feature_list.append(images)
        index2name.extend([name] * images.shape[0])

    if len(feature_list) == 0:
        raise Exception("There is no face in this folder. Please check your datasets.")
        
    # 将特征矩阵库加载到MXnet
    feature_matrix = np.concatenate(feature_list)

    return feature_matrix, index2name, error_list