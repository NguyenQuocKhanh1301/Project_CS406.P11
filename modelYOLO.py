from config import *
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO

class modelYOLO:
    def __init__(self):
        self.model = YOLO('./best_7.pt')
        self.model.classes = [0] #Class: LP 
        self.model.conf = CFG_THRESHOLD
        if torch.cuda.is_available():
            cudnn.benchmark = True
        print("Loaded: YOLOv8")
            
    def predict(self, image, ):
        list_coors = []
        results = self.model.predict(image)
        
        for result in results:
            boxes = result.boxes  # Get all bounding boxes
            for box in boxes:
                # Extract coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Extract confidence score and class ID
                confidence = box.conf[0].item()  # Confid~ence score
                
                # Apply thresholds
                if confidence >= CFG_THRESHOLD :
                    list_coors.append([int(x1.item()),int(y1.item()),int(x2.item()),int(y2.item())])
        
        return list_coors
