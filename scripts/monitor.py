"""
Monitor a camera
"""
import os
import sys
import time
import cv2

import argparse
import numpy as np

src_root = os.path.join(os.path.dirname(__file__), '..')
if src_root not in sys.path:
    sys.path.append(src_root)

from algorithm import insightface

parser = argparse.ArgumentParser(description='this is a description')
parser.add_argument('--video_path', type=str,
                    default=None, help="Read from video.")
parser.add_argument('--database_path', type=str, default=os.path.join(src_root, 'database/origin'))
parser.add_argument('--saved_path', type=str, default=None,
                    help="If set, Save as video. Or show it on screen.")
parser.add_argument("--p_threshold", type=float, default=0.5)
parser.add_argument("--min_size", type=int, default=20)
parser.add_argument("--stranger_pthreshold", type=float, default=0.28)
args = parser.parse_args()

here = os.path.abspath(os.path.dirname(__file__))

engine = getattr(insightface, "CosineSimilarityEngine")()
engine.load_database(args.database_path)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(args.saved_path, fourcc, fps, size)

while True:
    res, image = cap.read()
    if not res:
        continue

    try:
        ac, st = engine.detect_recognize_stranger(
            image, p_threshold=args.p_threshold, p_threshold_stranger=args.stranger_pthreshold, min_size=args.min_size)
    except:
        continue

    processed_image = engine.visualize(image, ac['names'], ac['probabilities'], ac['boxes'])
    processed_image = engine.visualize(image, st['names'], st['probabilities'], st['boxes'])

    if args.saved_path is None:
        cv2.imshow('Recognization', processed_image)
        cv2.waitKey(1)
    else:
        out.write(processed_image)
