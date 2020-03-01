'''
@Author: TangZhiFeng
@Data: 2019-01-05
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-08 16:36:18
@Description: 一些对应的映射关系  
'''

# 摄像头ip对应的id
camera_ip2camera_id = {
    'rtsp://admin:sdkj123456@192.168.0.64:554/Streaming/Channels/101' : '1',
    'rtsp://admin:admin12345@192.168.0.141:554/Streaming/Channels/101' : '2',
    'rtsp://admin:admin12345@192.168.0.52:554/Streaming/Channels/101' : '3',
    'rtsp://admin:admin12345@192.168.0.51:554/Streaming/Channels/101' : '4',
}

# 异常对应的rule_id
abnormal_type_to_rule_id = {
    'stay_too_long': 90002,
    'cluster': 90001,
    'flame': 90003
}
