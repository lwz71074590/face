"""Face Recognization threads with hook
"""
import time
import os
import traceback
import multiprocessing

from utils.image_base64 import str_to_base64
from algorithm import insightface
from algorithm import abnormal_detection
from algorithm import focus
from processes.message import RecognizerMessage, CameraMessage, AbnormalDetectionMessage, StrangerMessage, AttentionMessage, HumanDetectionMessage

from . import BaseNode


class BaseRecognizer(BaseNode):

    def init_node(self,
                  engine_name,  # 人脸识别引擎名称
                  face_database_path,  # 人脸库路径
                  minsize,  # 最小人脸像素
                  threshold,  # 人脸识别相似度阈值
                  tag,  # tag
                  gpu_ids,  # 可用的gpu编号
                  stranger_threshold=None,  # 陌生人识别阈值
                  ):

        self.engine_name = engine_name
        self.face_database_path = face_database_path
        self.minsize = minsize
        self.threshold = threshold
        self.tag = tag
        self.gpu_ids = gpu_ids
        self.stranger_threshold = stranger_threshold

    def on_detect(self, channel_id, name):
        """
        当某个人脸符合条件时调用
        """
        raise NotImplementedError

    def _run_sigle_process(self, i):
        print("Recognization node has been started.")

        gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
        engine = getattr(insightface, self.engine_name)(gpu_id=gpu_id)
        engine.load_database(self.face_database_path)

        while True:
            # 如果当前模式为单元测试模式并且队列为空则程序返回， 此处不影响程序正常运行
            if self.get_test_option() and self.q_in.qsize() == 0:
                break

            # Get the message from Queue
            msg = self.q_in.get()

            frame, channel_id, img_time, tag = msg.image, msg.channel_id, msg.record_time, msg.tag

            # TODO 这里运行时间长汇出错，这里判断一下
            try:
                if self.stranger_threshold is None:
                    original_face_image, names, probabilities, _, _ = engine.detect_recognize(
                        frame, p_threshold=self.threshold, min_size=self.minsize)
                else:
                    acquaintance, stranger = engine.detect_recognize_stranger(
                        frame, self.threshold, self.stranger_threshold, self.minsize, stranger_id=str_to_base64(img_time))

                    original_face_image, names, probabilities = acquaintance[
                        'image_matrix'], acquaintance['names'], acquaintance['probabilities']
                    stranger_face_image, stranger_names = stranger[
                        'image_matrix'], stranger['names']

            except Exception:
                print("Recogize error. Camera id: %s." % channel_id)
                traceback.print_exc()
                continue

            for _, name, _ in zip(original_face_image, names, probabilities):

                self.on_detect(channel_id, name)

            # 照片中没有人脸的时候不往队列里存储
            if len(names) > 0 and self.stranger_threshold is None:
                msg = RecognizerMessage(
                    original_face_image, names, img_time, channel_id, tag)
                self.q_out.put(msg)

            if self.stranger_threshold is not None and (len(names) > 0 or len(stranger_names) > 0):
                msg = StrangerMessage(stranger_face_image, stranger_names,
                                      original_face_image, names, img_time, channel_id, tag)
                self.q_out.put(msg)


source_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..')


class RealTimeRecognizer(BaseRecognizer):

    TOP = CameraMessage  # 上游节点需要传递的消息类
    BOTTOM = RecognizerMessage  # 下游节点需要传递的消息类

    default_params = {
        'engine_name': 'CosineSimilarityEngine',
        'face_database_path': os.path.join(source_root, 'database/origin'),
        'minsize': 40,
        'threshold': 0.5,
        'tag': "RealTimeRecognizer",
        'gpu_ids': [0]
    }

    def init_node(self, **kwargs):
        params = self.default_params.copy()
        params.update(kwargs)
        super(RealTimeRecognizer, self).init_node(**params)

    def on_detect(self, channel_id, name):
        # 测试状况下不打印
        if not self.get_test_option():
            print("摄像头%s检测到%s" % (channel_id, name))


class RealTimeStrangerRecognizer(BaseRecognizer):

    TOP = CameraMessage  # 上游节点需要传递的消息类
    BOTTOM = StrangerMessage  # 下游节点需要传递的消息类

    default_params = {
        'engine_name': 'CosineSimilarityEngine',
        'face_database_path': os.path.join(source_root, 'database/origin'),
        'minsize': 40,
        'threshold': 0.5,
        'tag': "RealTimeRecognizer",
        'gpu_ids': [0],
        'stranger_threshold': 0.3
    }

    def init_node(self, **kwargs):
        params = self.default_params.copy()
        params.update(kwargs)
        super(RealTimeStrangerRecognizer, self).init_node(**params)

    def on_detect(self, channel_id, name):
        # 测试状况下不打印
        if not self.get_test_option():
            print("摄像头%s检测到%s" % (channel_id, name))

# TODO merge with BaseRecognizer


class AttentionRecognizer(BaseRecognizer):
    TOP = CameraMessage  # 上游节点需要传递的消息类
    BOTTOM = AttentionMessage  # 下游节点需要传递的消息类

    default_params = {
        'engine_name': 'CosineSimilarityEngine',
        'face_database_path': os.path.join(source_root, 'database/origin'),
        'minsize': 40,
        'threshold': 0.5,
        'tag': "RealTimeRecognizer",
        'gpu_ids': [0]
    }

    def __init__(self, process_size=1, queue_type="ProcessingQueue"):
        super(AttentionRecognizer, self).__init__(process_size, queue_type)
        self.mp_dict = multiprocessing.Manager().dict()

    def init_node(self, **kwargs):
        params = self.default_params.copy()
        params.update(kwargs)
        super(AttentionRecognizer, self).init_node(**params)

    def _run_sigle_process(self, i):
        print("Recognization node has been started.")

        gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
        engine = getattr(insightface, self.engine_name)(gpu_id=gpu_id)
        engine.load_database(self.face_database_path)

        attn_detect = focus.Forcus(self.mp_dict)

        while True:
            # 如果当前模式为单元测试模式并且队列为空则程序返回， 此处不影响程序正常运行
            if self.get_test_option() and self.q_in.qsize() == 0:
                break

            # Get the message from Queue
            msg = self.q_in.get()

            frame, channel_id, img_time, tag = msg.image, msg.channel_id, msg.record_time, msg.tag

            # TODO 这里运行时间长汇出错，这里判断一下
            try:

                original_face_image, names, _, _, points = engine.detect_recognize(
                    frame, p_threshold=self.threshold, min_size=self.minsize)

            except Exception:
                print("Recogize error. Camera id: %s." % channel_id)
                traceback.print_exc()
                continue

            # TODO algorithm are not completed, create fake data here
            score = attn_detect.get_forcus(channel_id, names, points)

            # 照片中没有人脸的时候不往队列里存储
            if len(names) > 0:
                msg = self.BOTTOM(
                    score, original_face_image, names, img_time, channel_id, tag)
                self.q_out.put(msg)


class AbnormalDetectionRecognizer(BaseNode):

    TOP = CameraMessage  # 上游节点需要传递的消息类
    BOTTOM = RecognizerMessage  # 下游节点需要传递的消息类

    def __init__(self, queue_type="ProcessingQueue"):
        """该节点每个场景暂时只能开启一个
        """

        super(AbnormalDetectionRecognizer, self).__init__(
            process_size=1, queue_type=queue_type)

    default_params = {
        'tag': "RealTimeRecognizer",
        'gpu_ids': [0]
    }

    def init_node(self, **kwargs):
        params = self.default_params.copy()
        self.tag = params.get('tag')
        self.gpu_ids = params.get('gpu_ids')
        self.engine_name = "BaseEngine"

    def _run_sigle_process(self, i):
        print("Recognization node has been started.")

        self.before_last_time = list()
        self.before_last_time_cluster = list()

        def detect_abnormal(cameraImg, box, emb_array, cameraKey):
            # 异常检测代码

            all_people = []
            for mini_index, mini_box in enumerate(box):
                each_person = list()
                first_appear_time = time.time()
                final_disappear_time = 0
                each_person.append(emb_array[mini_index])
                each_person.append(first_appear_time)
                each_person.append(final_disappear_time)
                each_person.append(
                    [mini_box[0], mini_box[1], mini_box[2], mini_box[3]])
                all_people.append(each_person)

            self.before_last_time, stay_flag, stay_base64_data, stay_image_id, stay_cameraKey = abnormal_detection.stay_detect(
                cameraImg, self.before_last_time, all_people, cameraKey)
            self.before_last_time_cluster, box_flag, box_base64_data, box_image_id, box_cameraKey = abnormal_detection.box_cluster(
                cameraImg, self.before_last_time_cluster, all_people, cameraKey)

            stay_msg = AbnormalDetectionMessage(
                'stay_too_long', stay_flag, stay_base64_data, stay_image_id, stay_cameraKey)
            box_msg = AbnormalDetectionMessage(
                'cluster', box_flag, box_base64_data, box_image_id, box_cameraKey)

            return stay_msg, box_msg

        gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
        engine = getattr(insightface, self.engine_name)(gpu_id=gpu_id)

        while True:
            if self.get_test_option() and self.q_in.qsize() == 0:
                break

            msg = self.q_in.get()
            frame, channel_id, _ = msg.image, msg.channel_id, msg.record_time
            try:
                scaled_images, boxes, _, flag = engine.model.get_input(frame)
                if not flag:
                    continue
                mx_image_tensor = engine.model.get_feature_tensor(
                    scaled_images)
                stay_msg, box_msg = detect_abnormal(
                    frame, boxes, mx_image_tensor.asnumpy(), channel_id)

                if stay_msg.flag:
                    self.q_out.put(stay_msg)

                if box_msg.flag:
                    self.q_out.put(box_msg)

            except:
                traceback.print_exc()


class PeopleRecognizer(BaseRecognizer):

    TOP = HumanDetectionMessage  # 上游节点需要传递的消息类
    BOTTOM = StrangerMessage  # 下游节点需要传递的消息类

    default_params = {
        'engine_name': 'CosineSimilarityEngine',
        'face_database_path': os.path.join(source_root, 'database/origin'),
        'minsize': 40,
        'threshold': 0.5,
        'tag': "RealTimeRecognizer",
        'gpu_ids': [0]
    }

    def __init__(self, process_size=1, queue_type="ProcessingQueue"):
        super(PeopleRecognizer, self).__init__(process_size, queue_type)

    def init_node(self, **kwargs):
        params = self.default_params.copy()
        params.update(kwargs)
        super(PeopleRecognizer, self).init_node(**params)

    def _run_sigle_process(self, i):
        print("Recognization node has been started.")

        gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
        engine = getattr(insightface, self.engine_name)(gpu_id=gpu_id)
        engine.load_database(self.face_database_path)

        while True:
            # 如果当前模式为单元测试模式并且队列为空则程序返回， 此处不影响程序正常运行
            if self.get_test_option() and self.q_in.qsize() == 0:
                break

            # Get the message from Queue
            msg = self.q_in.get()

            flag, image_matrix, image_id, channel_id = msg.flag, msg.image_matrix, msg.image_id, msg.camera_key

            # TODO 这里运行时间长汇出错，这里判断一下
            try:
                acquaintance_names = []
                stranger_names = []
                acquaintance_images = []
                stranger_images = []
                for frame in image_matrix:
                    _, names, _, _, _ = engine.detect_recognize(
                        frame, p_threshold=self.threshold, min_size=self.minsize)
                    if len(names) > 0:
                        acquaintance_names.append(names[0])
                        acquaintance_images.append(frame)
                    else:
                        stranger_names.append(str_to_base64(time.time()))
                        stranger_images.append(frame)
            except Exception:
                print("Recogize error. Camera id: %s." % channel_id)
                traceback.print_exc()
                continue

            msg = self.BOTTOM(stranger_images, stranger_names, acquaintance_images, acquaintance_names, time.time(), channel_id, self.__class__.__name__)
            self.q_out.put(msg)
