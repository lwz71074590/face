import cv2

class FrameDiff:

    def __init__(self, cur_frame=None, next_frame=None, pixel_threshold=25):

        self.cur_frame = cur_frame
        self.next_frame = next_frame
        self.pixel_threshold = pixel_threshold

    def compute_diff(self, cur_frame):
        flag = False

        if self.next_frame is None:
            self.next_frame = cur_frame
            return True
        else:
            self.cur_frame = cur_frame

            gray_prev_frame = cv2.cvtColor(self.next_frame, cv2.COLOR_BGR2GRAY)
            gray_prev_frame = cv2.GaussianBlur(gray_prev_frame, (3, 3), 0)
            gray_cur_frame = cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2GRAY)
            gray_cur_frame = cv2.GaussianBlur(gray_cur_frame, (3, 3), 0)

            frame_diff = cv2.absdiff(gray_prev_frame, gray_cur_frame)
            frame_diff = cv2.threshold(frame_diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)[1]

            binary, cnts, hierarchy = cv2.findContours(frame_diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            self.next_frame = cur_frame

            # 遍历轮廓
            for c in cnts:
                # 忽略小轮廓，排除误差
                if cv2.contourArea(c) < 1000:
                    continue
                else:
                    flag = True
                    break

            if flag:
                return True
            else:
                return False

