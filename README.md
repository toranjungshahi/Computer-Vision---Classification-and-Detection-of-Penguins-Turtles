# Computer-Vision---Classification-and-Detection-of-Penguins & Turtles

This is a final project for UNSW COMP9517 2023 Term 2. The task involves experimenting with various data-preprocessing techniques to obtain highest accuracies in classification and detection of marine species, penguins and turtles. By adopting deep learning technology and utilizing the provided labeled dataset, we aim to develop a robust and accurate model capable of distinguishing between penguins and turtles in wildlife images. To obtain robust and accurate model we have experimented with SOTA models such as Yolov5 and Faster-RCNN.


## Data and Data Exploration


**Data Structure**
- Dataset used are supplied by the course lecturer. Similar dataset can be found on Kaggle.
- Contains 500 images of 640x640 resolution as training set and 71 images as validation set.
- Exploring the dataset shows that penguins have predominantly white belly and black back.
- Thus, penguin images predominantly has three regions; white belly, black back and image background.

**Preprocess – Thresholding and Augmentation**

To highlight the key features that differentiate penguins from other objects in the image and 
enhance the model's ability to accurately classify them, thresholding technique is used. The specific implementation involves 
iterating over the grayscale intensity values from the lowest to the highest and accumulating the pixel count until finding a 
threshold that evenly splits the pixels into two halves. Once this threshold is identified, we proceed to apply a similar 
triangular thresholding technique to each side of the threshold. The two thresholds obtained in this way, we can use two threshold values to 
segment the image into three regions: the black back, the white belly, and the background. Pixels with intensity values below 
the first threshold are classified as the black back region, pixels between the first and second thresholds belong to the 
background, and pixels above the second threshold represent the white belly region.

For image augmentation, we need to consider how to preserve the bbox information, crucial for object detection 
tasks. Horizontal and vertical flips were chosen as the primary augmentation methods. Unlike image rotation, flips allow us 
to maintain the relative positions of objects within the image, including the bbox coordinates. By using flips, we can 
effectively update the bbox information while augmenting the dataset. Regarding image size, we considered the trade-off 
between computational efficiency and information preservation. As the project involved deep learning for image 
prediction, larger image sizes would result in higher computational costs during training and inference. Therefore, 
we explored the impact of resizing images to different dimensions. So, for the choice of image augmentation techniques, 
including horizontal and vertical flips, and the decision to resize images to 160x160, were driven by their compatibility 
with object detection requirements and their ability to improve training efficiency without significant loss of information.  
As a result of data augmentation, our initial training dataset, which consisted of 500 images at 640x640 resolution, 
has been substantially expanded to a more diverse and robust set of 2000 images at 160x160 resolution. This augmentation process has 
enriched the training data, providing the model with a wider variety of samples to learn from.

## Usage

- Install the required dependencies listed in `requirements.txt` using the following command:

```
pip install -r requirements.txt
``` 
if in .ipynb file run the:
``` 
pip install -r ../requirements.txt
```
- Perform image preprocessing. Run all the file image_preprocess.ipynb in preprocess folder.
- Perform image Augmentation. Run all the file image_Augmentation.ipynb in folder preprocess folder.
- If you want to restart preprocessing or delete the preprocessed dataset for some reason, run the file delete_process_image.ipynb in preprocess folder.

**Faster R-CNN**
- Colored_160_Detector_FasterRCNN.ipyn in preprocess folder is used to apply FasterRCNN model on resized 160*160 images to detect and classify penguins and   turtles.

**Yolov5**
- Use the yolov5s model structure from yolov5's repo https://github.com/ultralytics/yolov5?tab=readme-ov-file 
- Change the parameter number of classes (nc) to 2 in yolov5s.yml file.
- Anchor, backbone and head can be used as it is.


## Frontend
- The frontend is build using Faster RCNN and Yolo v5 with pretrained state in the frontendModel folder. User can use their own pretrained state. In order to change the weight, the following will need to be done.
1. Change the state in frontendModel folder, make sure the name remain FasterRCNN.pth for faster R-CNN or Yolov5_best.pt for Yolo v5
2. Change the function detect_and_label_image_Faster_rcnn/detect_and_label_image_Yolov5 in UseFasterRCNN.py or UseYolov5 according to the model you used.


