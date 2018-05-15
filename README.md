# README

## Backend Code

This repository represents a year long work on back-end code for the Painted Dog Classification Project

It is split into three parts:

1. The Lambda Functions (Using the serverless platform for deployment)
*   Add-image-to-database 
    -   This function takes any image uploaded to the s3 bucket and adds it to the database automatically.
*   image-processor-lambda-event
    -   This function operates on an aws database stream, if an image is added to the database it is processed by called the correct RESTFUL API to trigger the image processing (#TODO: THE SERVER RUNNING THE IMAGE PROCESSING WITH THE MASKRCNN MUST STILL BE IMPLEMENTED AND THIS UPDATED LAMBDA FUNCTION DEPLOYED TO AWS)
2.  maskRCNN
*   This folder contains all of the code necessary to run a flask server capable of running the MaskRCNN image processing step (the coco dataset weights must be downloaded from [here](https://github.com/matterport/Mask_RCNN))
3.  Finally processor.py, this is the old depreciated code that used the mobile single shot detector paired with opencv cutgrab for image segmentation and background removal.
4.  The actual classifier with trained weights is located at this address: [Trained Painted Dog Image Feature Extractor](https://drive.google.com/file/d/19x_JcRjDs1d0BwO1dK_8RHNIlv0jnejh/view?usp=sharing)
*   There is still some work to be done here currently a dummy service sits at http://ec2-18-217-204-181.us-east-2.compute.amazonaws.com:8080/classify/packname/dogname/filename
    -   This dummy service always returns the same three dogs due to a poorly trained classifier, the new updated trained classifier must be dropped into this service. This should be fairly simple as soon as the dog classifier code is ready just wrap it a Flask API code and drop it onto the EC2 server.