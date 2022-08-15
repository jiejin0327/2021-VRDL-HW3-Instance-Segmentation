# instance_segmentation

HW3 Introduction: Nuclei segmentation

The proposed challenge is a nuclei segmentation, which contains two parts:

1. Transform the original train data format to coco dataset format for detectron2 training Segment the nuclei of bounding boxes
2. Nuclear segmentation dataset contains 24 training images with 14,598 nuclear and 6 test images with 2,360 nuclear Train an instance segmentation model to detect and segment all the nuclei in the image

This project uses the detectron2 pre-trained model to fix this challenge.

model : Mask-RCNN on detectron2

1. [inference.ipynb](https://drive.google.com/file/d/1xXLVqnBvuqr3Qdib3MuDLGYjeivvtwZK/view?usp=sharing)  
2. [Download weight data](https://drive.google.com/file/d/1--TCck58yvNZc2zTGyeTCn8z7OQMcPtl/view?usp=sharing)
3. [Download dataset](https://drive.google.com/file/d/1f3y6Wbessm-qC3WvaretMf46UL8MLPfU/view?usp=sharing)

reference code from [detectron2 official github](https://github.com/facebookresearch/detectron2)
