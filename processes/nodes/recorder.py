import json
import os
import cv2
import time

from processes.message import CameraMessage

from . import BaseNode


class CameraReader(BaseNode):

    BOTTOM = CameraMessage

    def __init__(self, *args, **kwargs):
        kwargs['is_in'] = False
        super(CameraReader, self).__init__(*args, **kwargs)  # 　摄像头节点没有入口只有出口

    def init_node(self, channel_ips, channel_ids, read_fps_interval, tag):
        """Init CameraReader

        Args:
            channel_ips (list): IP address to acess camera
            channel_ids (list): Camera identity number
            read_fps_interval (float): How often do you read the camera.
            tag (str): tag for this thread
        """
        self.channel_ips = channel_ips
        self.read_fps_interval = read_fps_interval
        self.channel_ids = channel_ids
        self.tag = tag

        # 确认id 和 ip数量相等
        assert len(channel_ids) == len(channel_ips)
        self.process_size = len(channel_ids)

    def _run_sigle_process(self, i):
        # 测试计数变量
        if self.get_test_option():
            count = 0

        channel_id = self.channel_ids[i]
        channel_ip = self.channel_ips[i]

        cap = cv2.VideoCapture(channel_ip)
        i = 0
        print("Camera Thread %s has been started" % channel_id)
        while True:
            if not cap.isOpened():
                cap.release()
                time.sleep(10)
                print('摄像头%s断开，正在重连。' % (channel_id))
                cap = cv2.VideoCapture(channel_ip)
            res, image = cap.read()
            if res == False:
                continue

            # Push frame in queue every "read_fps_interval" times.
            i += 1
            if i == self.read_fps_interval:
                if self.q_out.qsize() > 3 and not self.get_test_option():

                    print("Queue Name %s" % self.tag,
                          self.q_out.qsize())

                else:
                    message = CameraMessage(
                        image=image,
                        channel_id=channel_id,
                        tag=self.tag
                    )
                    self.q_out.put(message)

                i = 0

                if self.get_test_option():
                    count += 1
            # 在测试的情况下跳出程序
            if self.get_test_option() and count >= 10:
                break

    def run(self):
        # 在新开的进程中使用线程运行
        super(CameraReader, self).run(type='threading')
