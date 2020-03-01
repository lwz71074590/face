# coding: utf-8
import os
import paramiko
ip = '47.97.185.63'
port = 9527
username = 'root'
password = 'HZbigdata-cs'
transport = paramiko.Transport((ip, port))
transport.connect(username=username, password=password)
# TODO 这里没有close


def upload_image_to_remote(local_image_path, remote_save_path):
    '''
    从算法节点把识别到的人脸图片上传到web节点
    :param local_image_path: 需要上传的算法节点上的图片路径
    :param remote_save_path: 需要上传到web节点的保存路径
    :return:
    '''
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(local_image_path, remote_save_path)
    except:
        import traceback
        traceback.print_exc()


def batch_stranger_upload(paths, stranger_ids):
    '''批量上传陌生人，远程存储路径/docker_data/nginx/web/file/school_storge/stranger

    Arguments:
            paths {list} -- 本地路径
            stranger_ids {list} -- 陌生人编号
    '''
    remote_save_path = '/docker_data/nginx/web/file/school_storge/stranger'

    for index in range(len(stranger_ids)):
        remote_path = os.path.join(
            remote_save_path, str(stranger_ids[index])+'.jpg')
        upload_image_to_remote(paths[index], remote_path)


def batch_people_upload(paths, camera_id, ids, stamp):
    '''批量上传type为1的情况

    Arguments:
            paths {list} -- 本地照片的路径
            camera_id {str} -- 摄像头编号
            ids {str} -- 摄像头的编号
            stamp {str} -- 时间戳
    '''

    # remote_save_path = '/raid/home/wangning/1.png'
    remote_save_path = '/docker_data/nginx/web/file/school_storge/man'
    for index in range(len(ids)):
        remote_path = os.path.join(
            remote_save_path, ids[index], str(camera_id), stamp+'.jpg')
        upload_image_to_remote(paths[index], remote_save_path)


def batch_type2_upload(path, camera_id, stamp):
    '''批量上传type为2的情况，远程存储路径：/excep_scene/设备编号/时间戳.jpg

    Arguments:
            path {list} -- 本地存储的照片
            camera_id {str} -- 摄像头编号
            stamp {str} -- 时间戳
    '''
    remote_save_path = '/docker_data/nginx/web/file/school_storge/excep_scene'
    remote_save_path = os.path.join(
        remote_save_path, str(camera_id), stamp + '.jpg')
    upload_image_to_remote(path, remote_save_path)
