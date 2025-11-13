import torch
from ultralytics import YOLO
from torchvision import transforms

# Load YOLO v11 model
model = YOLO("yolo11n.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.model.to(device)
model.model.eval()

# YOLO v11 uses 80 COCO classes
num_classes = 80  # COCO dataset has 80 classes
dfl_channels = 64  # Distribution Focal Loss channels for bbox

# Transform for preprocessing (YOLO expects [0, 1] range)
transform = transforms.Compose([
    transforms.ToTensor(),
])

def adv_loss_calc(batch_frames, orig_clases):
        logits = predict_raw(batch_frames)
        if (logits != logits).any():
            raise
        probs = torch.softmax(logits, dim=1)
        orig_class_probs = probs[:, orig_clases]
        orig_loss = 5.0 * torch.log(orig_class_probs.sum(dim=1) + 1e-10).mean()
        return orig_loss

def predict_raw(img_tensor):
    """
    Extract class logits from YOLO model (assumes model is in .eval() mode).
    
    Args:
        model: YOLO model (should already be in eval mode)
        img_tensor: Input image tensor [batch, 3, H, W] with requires_grad=True
    
    Returns:
        logits: Tensor of shape [80] with maximum score for each class
    """
    with torch.set_grad_enabled(True):
        output = model.model(img_tensor)
        
        # In eval mode: output is tuple, first element is [batch, 84, num_detections]
        # where 84 = 4 (bbox) + 80 (class scores)
        predictions = output[0]  # [batch, 84, num_detections]
        class_scores = predictions[:, 4:, :]  # [batch, 80, num_detections]
        
        # Get max score for each class across all detections
        logits, _ = class_scores.max(dim=2)  # [batch, 80]
    
    return logits



# COCO class names for reference
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
