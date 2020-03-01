'''
@Author: TangZhiFeng
@Data: 火情检测——单元测试
@LastEditors: TangZhiFeng
@LastEditTime: 2019-01-05 09:35:57
@Description: 
'''


import os
import sys
import unittest

project = os.path.dirname(os.path.dirname(__name__))
sys.path.append(os.path.join(project))

from algorithm.fire_discover.interface import FireEngine

class FireTest(unittest.TestCase):
    
    def setUp(self):
        self.fire = FireEngine()
        self.test_image = cv2.imread(os.path.join(
            project, 'database/cache/test_fire/test_fire.jpg'))
        self.test_image_nomal = cv2.imread(os.path.join(
            project, 'database/cache/test_fire/test_nonfire.jpg'))
        
    def discover(self):
        result = self.fire.predict(self.test_image)
        self.assertTrue(result)
        nomal = self.fire.predict(self.test_image_nomal)
        self.assertFalse(nomal)
        
if __name__ == "__main__":
    unittest.main()


