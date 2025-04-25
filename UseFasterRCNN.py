import torch
from torchvision import transforms
import os
import cv2
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.models as models
import torchvision.ops as ops
import numpy as np

Path = os.getcwd()
# Green for class 1(penguin) and Orange for class 2(Turtle)
class_color = {
    1: (0, 255, 0),
    2: (0, 165, 255),
}

class_names = ['BG', 'penguin', 'turtle']

def detect_and_label_image_Faster_rcnn(img):
    backbone = models.vgg16(pretrained=True).features
    backbone.out_channels = 512
    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0', '1'], output_size=5, sampling_ratio=2)
    model = FasterRCNN(backbone=backbone,rpn_nms_thresh=0.6, rpn_fg_iou_thresh=0.6, rpn_bg_iou_thresh=0.2, num_classes=3, rpn_positive_fraction=1,rpn_anchor_generator=AnchorGenerator(sizes=((5,10,32,64,128,256,512,800,1024,2000,2500,3000,3500,4000,4500,5000,6000,10000,14000,16000,20000,25000),), aspect_ratios=((0.1,0.3,0.5,0.8,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5,5.5,6),)),box_detections_per_img=300,box_roi_pool=roi_pooler)
    modelPath = os.path.join(Path, 'frontendModel', 'FasterRCNN.pth')
    state_dict = torch.load(modelPath)
    
    # print(state_dict.keys())
    
    model.load_state_dict(state_dict)
    model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(img)
    
    with torch.no_grad():
        pred = model([image_tensor])
        print(f"pred = {pred}")
        # Draw the predicted bounding boxes on the image
        for box, label, score in zip(pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']):
            if score > 0.5:
                color = class_color[int(label)]
                box = box.numpy().astype(np.int32)
                img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                img = cv2.putText(img, class_names[label], (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return img
    
if __name__ == '__main__':
    # Load an image
    imgPath = os.path.join(Path, 'frontendModel','testImages', 'Test2.jpg')
    img = cv2.imread(imgPath)
    # Show the decorated image
    img = detect_and_label_image_Faster_rcnn(img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
