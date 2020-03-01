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
parser.add_argument('--camera_ip', type=str,
                    default=None, help="Read from video.")
parser.add_argument('--saved_path', type=str, default=None,
                    help="If set, Save as video. Or show it on screen.")

args = parser.parse_args()


fourcc = cv2.VideoWriter_fourcc(*"XVID")

cap = cv2.VideoCapture(args.camera_ip)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps > 100:
    fps = 40
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


if args.saved_path is not None:
    out = cv2.VideoWriter(args.saved_path, fourcc, fps, size)

while True:

    res, image = cap.read()

    if args.saved_path is None:
        cv2.imshow("asdfas", image)
        cv2.waitKey(1)
    else:
        out.write(image)

out.release()
