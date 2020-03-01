import os
import sys
import cv2
import numpy as np
import argparse

src_root = os.path.join(os.path.dirname(__file__), '..')
if src_root not in sys.path:
    sys.path.append(src_root)

from algorithm import insightface

here = os.path.abspath(os.path.dirname(__file__))
parser = argparse.ArgumentParser(description='this is a description')
parser.add_argument('--video_path', type=str,
                    default=None, help="Read from video.")
parser.add_argument('--saved_path', type=str, default=None,
                    help="If set, Save as video. Or show it on screen.")

args = parser.parse_args()

engine = getattr(insightface, 'BaseEngine')()


fourcc = cv2.VideoWriter_fourcc(*"XVID")

cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

if args.saved_path is not None:
    out = cv2.VideoWriter(args.saved_path, fourcc, fps, size)

while True:

    res, image = cap.read()
    if not res:
        break
    try:
        _, boxes, points, _ = engine.get_detection(image)
    except Exception as e:
        continue
    if boxes is not None:
        for box in boxes:
            box = box.astype(np.int32)
            image = cv2.rectangle(image, (box[0], box[1]),
                                (box[2], box[3]), (255, 0, 0), 2)
        for p in points:
            for i in range(5):
                cv2.circle(image, (p[i], p[i+5]), 1, (0, 255, 0), -1)

    if args.saved_path is None:
        cv2.imshow("asdfas", image)
        cv2.waitKey(1)
    else:
        out.write(image)

out.release()
