'''
@Author: TangZhiFeng
@Data: 2019-01-06
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-08 16:35:24
@Description: 宿舍的异常报警，主要是消防的
'''


import os
import sys
import json

current = os.path.dirname(__name__)
project = os.path.dirname(os.path.dirname(current))
sys.path.append(project)


from utils.image_base64 import array_to_file
from processes.nodes.diff_node import FrameDiffNode
from processes.nodes.recorder import CameraReader
from scenes import BaseEngineering, BaseScenesManage
from processes.nodes.recognizer import RealTimeRecognizer
from utils.time_utils import genera_stamp
from utils.socket_client import client
from utils.upload_image import batch_people_upload,batch_type2_upload
from processes.nodes.recognizer import AbnormalDetectionRecognizer
from utils.keymap import camera_ip2camera_id, abnormal_type_to_rule_id
class AbnormalDormitorEngineering(BaseEngineering):
    '''实时检测宿舍异常情况
    '''

    def __init__(self):
        real_time = True
        super(AbnormalDormitorEngineering, self).__init__(real_time)

    def build_data(self, i, data):
        return data

    def generater(self, data):
        '''返回示例{
            type:x
                //识别数据类型,1 为人员检测结果,非 1 为场景异常结果
                data:{
                    camera_id:xxxx,
                    //识别摄像头 id
                    scenario_id:xxx,
                    //场所 id
                    image_id:xxxxx ,
                    //图片识别时间戳
                    rule_id:xxxx
                    //匹配的异常规则
                }
            }

        Arguments:
            data {[obj]} -- 对象中含有我需要的数据,stay_too_long逗留，cluster聚集
        '''
        result = {'data': {
            'type': 2,
            'data': {}
        }}
        stamp = genera_stamp()
        rule_id = abnormal_type_to_rule_id[data.abnormal_type]
        result['data']['data']['camera_id'] = data.camera_key
        result['data']['data']['scenario_id'] = data.camera_key
        result['data']['stamp'] = stamp
        result['data']['rule_id'] = rule_id

        # 这里后面一个参数只是取这个list的长度作为生成paths的list长度，前者需要一个list。
        path = array_to_file([data.base64_data], [1])
        assert len(path) == 1
        batch_type2_upload(path[0], data.camera_key, result['data']['stamp'])
        client.send(json.dumps(result, ensure_ascii=False))


class AbnormalDormitoryScene(BaseScenesManage):
    '''实时宿舍异常检测
    '''

    def init_node(self):
        camera = CameraReader()
        realtime_abnormal = AbnormalDetectionRecognizer()
        diff = FrameDiffNode(1)
        diff.init_node()
        camera_ip = [
            'rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101']
        camera_id = [camera_ip2camera_id[ip] for ip in camera_ip]
        camera.init_node(camera_ip, camera_id, 10, "123")
        realtime_abnormal.init_node()

        self.nodes.append(camera)
        self.nodes.append(diff)
        self.nodes.append(realtime_abnormal)
        self.manage = AbnormalDormitorEngineering()


if __name__ == "__main__":
    rt_scene = AbnormalDormitoryScene()
    rt_scene.start()
