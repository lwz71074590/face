import tensorflow as tf


from algorithm import fire_detection
from processes.message import AbnormalDetectionMessage, CameraMessage

from . import BaseNode


class SmokeDetection(BaseNode):

    TOP = CameraMessage  # 上游节点需要传递的消息类
    BOTTOM = AbnormalDetectionMessage  # 下游节点需要传递的消息类


    def init_node(self, block_threshold):
        """Init SmokeDetection

        Args:
            block_threshold (int): 如果烟雾块大于block_threshold, 报警
        """
        self.block_threshold = block_threshold

    def _run_sigle_process(self, i):

        # 加载参数和模型
        sess, model = fire_detection.get_default_model()

        while True:
            # 如果是测试状态队列为空时跳出循环
            if self.get_test_option() and self.q_in.qsize() == 0:
                break
            # Get the message from Queue
            msg = self.q_in.get()

            frame, channel_id, img_time, tag = msg.image, msg.channel_id, msg.record_time, msg.tag

            height = frame.shape[0]
            width = frame.shape[1]
            # 烟雾检测返回结果
            flag, frame, channel_id, image_id = fire_detection.detect(sess=sess,
                                                                      model=model,
                                                                      frame=frame,
                                                                      frame_height=height,
                                                                      frame_width=width,
                                                                      block_threshold=self.block_threshold,
                                                                      cameraKey=channel_id)
            # 如果flag为True, 表示检测到烟雾, 将结果存入队列
            if flag:
                print('as;dkfjak;dslfjakldfkal;sdfjkl;adsjfds')
                msg = AbnormalDetectionMessage(abnormal_type='smoke',
                                               flag=flag,
                                               base64_data=frame,
                                               image_id=image_id,
                                               camera_key=channel_id)
                self.q_out.put(msg)