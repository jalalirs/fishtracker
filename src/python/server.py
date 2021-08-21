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
from flask import Flask, flash, request, redirect, url_for, send_file
import time
import sys
import os
import io
from PIL import Image

app = Flask(__name__)


DIR = os.path.dirname(os.path.abspath(__file__))

model = sys.argv[1]
with open(f"{DIR}/models/{model}/labels") as f:
    labels = f.read().splitlines()
# Initialize predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.20  # set threshold for this model
cfg.MODEL.WEIGHTS = f"{DIR}/models/{model}/{model}.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

my_metadata = Metadata()
my_metadata.set(thing_classes = labels)
# Initialize visualizer
v = VideoVisualizer(my_metadata, ColorMode.IMAGE)

# define a predict function as an endpoint
@app.route("/predict", methods=["GET","POST"])
def predict():
    # print(request.files , file=sys.stderr)
    npimg = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    ######### Do preprocessing here ################
    # Make sure the frame is colored
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    outputs = predictor(img)

    # Draw a visualization of the predictions using the video visualizer
    visualization = v.draw_instance_predictions(img, outputs["instances"])

    # Convert Matplotlib RGB format to OpenCV BGR format
    visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_BGR2RGB)
    ################################################
    img = Image.fromarray(visualization.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    return send_file(rawBytes, mimetype='image/jpeg')

if __name__ == '__main__':
	app.run(host='127.0.0.1',port=8080)