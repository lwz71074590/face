import time
import os

from processes.message import CameraMessage
from algorithm import interframe_difference

from . import BaseNode


class FrameDiffNode(BaseNode):
    TOP = CameraMessage
    BOTTOM = CameraMessage

    def __init__(self, process_size=1, queue_type="ProcessingQueue"):
        super(FrameDiffNode, self).__init__(
            process_size, queue_type)

    def init_node(self, pixel_threshold=20):

        self.diff = interframe_difference.FrameDiff(
            pixel_threshold=pixel_threshold)

    def _run_sigle_process(self, i):

        while(True):
            if self.get_test_option() and self.q_in.qsize() == 0:
                break
            msg_in = self.q_in.get()

            image = msg_in.image

            if self.q_out.qsize() > 4:
                print("%s bottom queue size is greater than 4." % self.__class__.__name__)
                continue

            if self.diff.compute_diff(image):
                self.q_out.put(msg_in)
