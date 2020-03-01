from processes.message import HumanDetectionMessage, CameraMessage

from algorithm.object_detection.person_detect import detect_person, load_model

from . import BaseNode


class HumanDetection(BaseNode):
    TOP = CameraMessage
    BOTTOM = HumanDetectionMessage

    def init_node(self):
        pass

    def _run_sigle_process(self, i):

        model, classes, colors = load_model()

        while True:
            
            if self.get_test_option() and self.q_in.qsize() == 0:
                break
            # Get the message from Queue
            msg = self.q_in.get()

            frame, channel_id, img_time, tag = msg.image, msg.channel_id, msg.record_time, msg.tag

            # 烟雾检测返回结果
            flag, frame, channel_id, image_id = detect_person(model=model,
                                                              frame=frame,
                                                              cameraKey=channel_id,
                                                              classes=classes,
                                                              colors=colors)
            # 如果flag为True, 表示检测到人体, 将结果存入队列
            if flag:
                msg = HumanDetectionMessage(flag=flag,
                                            image_matrix=frame,
                                            image_id=image_id,
                                            camera_key=channel_id)
                self.q_out.put(msg)
            else:
                continue
