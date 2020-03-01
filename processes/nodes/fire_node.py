'''
@Author: TangZhiFeng
@Data: 2019-01-05
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-08 11:33:22
@Description: 火焰检测——NODE
'''

import time
import os

from processes.message import CameraMessage
from processes.message import AbnormalDetectionMessage
from algorithm.fire_discover.interface import FireEngine
from . import BaseNode


class FlameDiffNode(BaseNode):
    TOP = CameraMessage
    BOTTOM = AbnormalDetectionMessage

    def __init__(self, process_size=1, queue_type="ProcessingQueue"):
        super(FlameDiffNode, self).__init__(
            process_size, queue_type)

    def init_node(self):
        pass
        

    def _run_sigle_process(self, i):
        
        fire = FireEngine()

        while(True):

            if self.get_test_option() and self.q_in.qsize() == 0:
                break
            msg_in = self.q_in.get()

            image = msg_in.image

            if self.q_out.qsize() > 4:
                print("%s bottom queue size is greater than 4." % self.__class__.__name__)
                continue
            result = fire.predict(image)
            if result:
                data = AbnormalDetectionMessage(
                    'flame',
                    True,
                    image,
                    str(time.time()),
                    msg_in.channel_id
                )
                self.q_out.put(data)