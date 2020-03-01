from scenes.realtime.school_gate import RealTimeSchoolGateScene
from scenes.realtime.dormitory import RealTimeDormitoryScene
from scenes.abnormal.school_gate import AbnormalSchoolGateScene

if __name__ == "__main__":
    real_time_school_gate = RealTimeSchoolGateScene()
    real_time_dormitory = RealTimeDormitoryScene()
    abnormal_school_gate = AbnormalSchoolGateScene()
    real_time_dormitory.start()
    real_time_school_gate.start()
    abnormal_school_gate.start()