import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import boto3
from flask import Flask, request
from flask_restful import Resource, Api
import flask
from json import dumps
import io

app = Flask(__name__)
api = Api(app)

class Process(Resource):
    def get(self, picture_name):

        s3 = boto3.resource('s3')
        obj = s3.Object(bucket_name='ndpainteddogs', key=picture_name)
        response = obj.get()
        data = response['Body'].read()

        nparr = np.fromstring(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        imor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Root directory of the project
        ROOT_DIR = os.path.abspath(".")

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        from mrcnn import utils
        from mrcnn import visualize
        from mrcnn.visualize import display_images
        import mrcnn.model as modellib
        from mrcnn.model import log 

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        # MS COCO Dataset
        import coco
        config = coco.CocoConfig()
        COCO_DIR = "./coco"  # TODO: enter value here

        # Override the training configurations with a few
        # changes for inferencing.
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Device to load the neural network on.
        # Useful if you're training a model on the same 
        # machine, in which case use CPU and leave the
        # GPU for training.
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        # TODO: code for 'training' test mode not ready yet
        TEST_MODE = "inference"

        # Build validation dataset
        dataset = coco.CocoDataset()
        dataset.load_coco(COCO_DIR, "minival")

        # Must call before using the dataset
        dataset.prepare()

        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)

        # Set weights file path
        weights_path = COCO_MODEL_PATH

        # Load weights
        print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True)

        from skimage.transform import resize
        im = resize(imor, (1024,1024), mode="constant", preserve_range=True)
        # Run object detection
        results = model.detect([im], verbose=1)

        def apply_mask(image, mask, color, alpha=1):
            """Apply the given mask to the image.
            """
            for c in range(3):
                image[:, :, c] = np.where(mask == 0,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * color[c] * 255,
                                          image[:, :, c])
            return image

        # Display results
        r = results[0]
        unique_class_ids = np.unique(r['class_ids'])
        mask_area = [np.sum(r['masks'][:, :, np.where(r['class_ids'] == i)[0]]) for i in unique_class_ids]
        top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area), key=lambda r: r[1], reverse=True) if v[1] > 0]
        # Generate images and titles
        for i in range(1):
            class_id = top_ids[i] if i < len(top_ids) else -1
            # Pull masks of instances belonging to the same class.
            m = r['masks'][:, :, np.where(r['class_ids'] == class_id)[0]]
            m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        image_cropped = im.astype(np.uint32).copy()
        image_cropped = apply_mask(image_cropped, m, (0,0,0))
        TARGET_PIXEL_AREA=160000.0
        ratio = float(imor.shape[1]) / float(imor.shape[0])
        """Calculate the new height"""
        new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
        """Calculate the new width"""
        new_w = int((new_h * ratio) + 0.5)
        imnew = image_cropped.astype(np.uint8)
        imnew = cv2.resize(imnew, (new_w,new_h))
        almost = cv2.imencode('.jpg',cv2.cvtColor(imnew, cv2.COLOR_RGB2BGR))[1].tostring()
        #return flask.send_file(io.BytesIO(almost), mimetype='image/jpeg')
        s3.Bucket('ndprocessedimages').put_object(Key=picture_name, Body=almost)
        response = {
          "statusCode": 200
        }
        return response



api.add_resource(Process, '/process/<path:picture_name>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
