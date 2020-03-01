import os
import sys
import unittest

project = os.path.dirname(os.path.dirname(__name__))
sys.path.append(os.path.join(project))

from utils.upload_image import batch_upload

class ScpUpload(unittest.TestCase):
    
    def upload(self):
        batch_upload(None)
    
if __name__ == "__main__":
    unittest.main()
