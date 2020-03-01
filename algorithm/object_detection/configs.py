

dataset = 'pascal'  # Dataset on which the network has been trained
confidence = 0.5  # Object Confidence to filter predictions
nms_thresh = 0.4  # NMS Threshhold
cfg = './cfg/yolov3.cfg'  # Config file
weights = './yolov3.weights'  # weightsfile
reso = '416'  # Input resolution of the network. Increase to increase accuracy. Decrease to increase speed
classes_model = './data/coco.names'
pallete_model = './pallete'