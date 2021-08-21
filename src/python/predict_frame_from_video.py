#!/usr/bin/env python3
# -- coding: utf-8 --

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import tqdm
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata

import time

import sys
import os

DIR = os.path.dirname(os.path.abspath(__file__))

model = sys.argv[1]
source = sys.argv[2]
target = sys.argv[3]
frameNumber = int(sys.argv[4])
with open(f"{DIR}/models/{model}/labels") as f:
    labels = f.read().splitlines()

# Extract video properties
video = cv2.VideoCapture(source)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if frameNumber <=0 or frameNumber > num_frames:
    print(f"Frame number must be 0<N<={num_frames}")
    exit()

# Initialize predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
cfg.MODEL.WEIGHTS = f"{DIR}/models/{model}/{model}.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

my_metadata = Metadata()
my_metadata.set(thing_classes = labels)
# Initialize visualizer
v = VideoVisualizer(my_metadata, ColorMode.IMAGE)


video.set(0,frameNumber)

_, frame = video.read()
# Get prediction results for this frame
outputs = predictor(frame)
# Make sure the frame is colored
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# Draw a visualization of the predictions using the video visualizer
visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
# Convert Matplotlib RGB format to OpenCV BGR format
visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
cv2.imwrite(target, visualization)
# Release resources
video.release()
cv2.destroyAllWindows()
