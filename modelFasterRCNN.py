from config import *
import os
import sys
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class modelFasterRCNN:
    def __init__(self):
        # Set device to CPU (you can add GPU support if needed)
        self.device = torch.device('cpu')
        self.threshold = CFG_THRESHOLD
        # Initialize the Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)  # Adjust num_classes as per dataset
        
        # Load the checkpoint
        checkpoint = torch.load(CFG_PATH_FASTERRCNN_MODEL, map_location=self.device)
        self.model.load_state_dict(checkpoint)        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        print("Loaded: FASTER RCNN")
        
    def predict(self, image):
        image_transformed = self.transform(image)
        # image_transformed = image_transformed.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            # Perform inference
            predictions = self.model([image_transformed.to(self.device)])
        end_time = time.time()
        print(f'time to detect{end_time - start_time}')
        # Extract and rescale bounding boxes
        boxes = []
        for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
            if score > self.threshold:  # Confidence threshold
                # Scale bounding box back to original image size
                x_min, y_min, x_max, y_max = box.tolist()                
                
                x_min = int(x_min )
                y_min = int(y_min )
                x_max = int(x_max )
                y_max = int(y_max )
                
                # Append rescaled bounding box
                boxes.append([x_min, y_min, x_max, y_max])
        
        return boxes
