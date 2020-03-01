import time
from numpy import *
import numpy as np


def cos_sim(vector_a, vector_b):

    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def distEclud(box_center, cluster_center):

    dist = sqrt(power((box_center[0] - cluster_center[0]), 2) + power((box_center[1] - cluster_center[1]), 2))
    return dist

def get_cluster_center(cluster):

    box_x_axis = list()
    box_y_axis = list()
    for index in range(len(cluster)):
        box_x_axis.append(cluster[index][3][0])
        box_y_axis.append(cluster[index][3][1])

    cluster_x_center = sum(box_x_axis)/len(box_x_axis)
    cluster_y_center = sum(box_y_axis)/len(box_y_axis)

    cluster_center = list()
    cluster_center.append(cluster_x_center)
    cluster_center.append(cluster_y_center)

    return cluster_center

def get_box_center(box_list):

    x_center = abs((box_list[0] + box_list[2])/2)
    y_center = abs((box_list[1] + box_list[3])/2)

    box_center = list()
    box_center.append(x_center)
    box_center.append(y_center)

    return box_center

def box_cluster(cameraImg, before_last_time_cluster, all_people, cameraKey):

    """
    异常聚集检测
    :param cameraImg: 摄像头读取的当前帧的图片
    :param before_last_time_cluster: 上一帧图片检测到的人脸信息
    :param all_people: 当前帧图片检测到的人脸信息
    :param cameraKey: 摄像头编号
    :returns [
              clusters_info: 当前帧人脸信息更新起止时间后的信息
              flag: 是否报警的标志位，False代表未报警，True代表报警
              cameraImg: 摄像头读取的当前帧的图片
              image_id: 图片编号，如果报警（flag为True）, 生成以时间戳编码的图片编号, 否则为0
              cameraKey: 摄像头编号
            ]
    """

    flag = False
    image_id = 0

    clusters = list()
    if before_last_time_cluster == []:
        for people in all_people:
            embs = people[0]
            boxs = people[3]
            box_list = list()
            for index in range(len(boxs)):
                if index <= 3:
                    box_list.append(boxs[index])

            box_center = get_box_center(box_list)

            min_dist = 100000
            threshold = 500
            targetCluster = list()

            for cluster in clusters:
                dist = distEclud(box_center, get_cluster_center(cluster))
                if dist < min_dist:
                    min_dist = dist
                    targetCluster = cluster

            dist = min_dist
            if dist < threshold:
                for cluster in clusters:
                    if cluster == targetCluster:
                        first_appear_time = time.time()
                        final_disappear_time = 0
                        cluster.append([first_appear_time, final_disappear_time, embs, box_center])
                        break
            else:
                c = list()
                first_appear_time = time.time()
                final_disappear_time = 0
                c.append([first_appear_time, final_disappear_time, embs, box_center])
                clusters.append(c)

        return clusters, False, None, None, None
    else:

        for people in all_people:
            embs = people[0]
            boxs = people[3]
            box_list = list()
            for index in range(len(boxs)):
                if index <= 3:
                    box_list.append(boxs[index])

            box_center = get_box_center(box_list)

            min_dist = 100000
            threshold = 500
            targetCluster = list()

            for cluster in clusters:
                dist = distEclud(box_center, get_cluster_center(cluster))
                if dist < min_dist:
                    min_dist = dist
                    targetCluster = cluster

            dist = min_dist
            if dist < threshold:
                for cluster in clusters:
                    if cluster == targetCluster:
                        first_appear_time = time.time()
                        final_disappear_time = 0
                        cluster.append([first_appear_time, final_disappear_time, embs, box_center])
                        break
            else:
                c = list()
                first_appear_time = time.time()
                final_disappear_time = 0
                c.append([first_appear_time, final_disappear_time, embs, box_center])
                clusters.append(c)


        clusters_info = list()
        for m in range(len(clusters)):

            cluster = clusters[m]
            before_count = 0
            for n in range(len(before_last_time_cluster)):
                before_cluster = before_last_time_cluster[n]
                embs_max = 0
                threshold = 0.9
                for i in range(len(cluster)):
                    embs = cluster[i][2]
                    for j in range(len(before_cluster)):
                        before_embs = before_cluster[j][2]
                        dist = cos_sim(embs, before_embs)
                        if dist > embs_max:
                            embs_max = dist

                if embs_max >= threshold:
                    stay_time = []

                    for x in range(len(cluster)):
                        distance = []
                        for y in range(len(before_cluster)):
                            embs = cluster[x][2]
                            before_embs = before_cluster[y][2]
                            distance.append(cos_sim(embs, before_embs))
                        if max(distance) >= threshold:
                            stay_time.append(cluster[x][0] - before_cluster[distance.index(max(distance))][0])
                        else:
                            stay_time.append(0)

                    if min(stay_time) > 0.2 and len(cluster) >= 2:
                        print('聚众报警')

                        # 报警，标志位变True
                        flag = True

                        # image_id: 时间戳到秒
                        image_id = str(int(time.time()))

                        for k in range(len(cluster)):
                            first_appear_time = time.time()
                            cluster[k][0] = first_appear_time
                            cluster[k][1] = 0
                        break

                    else:

                        final_disappear_time = time.time()
                        for x in range(len(cluster)):
                            for y in range(len(before_cluster)):
                                embs = cluster[x][2]
                                before_embs = before_cluster[y][2]
                                distance = cos_sim(embs, before_embs)
                                if distance >= threshold:
                                    cluster[x][0] = before_cluster[y][0]
                                    cluster[x][1] = final_disappear_time
                                else:
                                    cluster[x][1] = final_disappear_time
                        clusters_info.append(cluster)
                        break

                elif embs_max < threshold:
                    before_count += 1
            if before_count == len(before_last_time_cluster):
                clusters_info.append(cluster)

        return clusters_info, flag, cameraImg, image_id, cameraKey