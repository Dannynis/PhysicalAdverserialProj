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


def get_bboxes_with_class_probs(model, img_tensor):
    """
    Extract bounding boxes with class probabilities from YOLO model.
    This function is fully differentiable.
    
    Args:
        model: YOLO model (should be in .eval() mode)
        img_tensor: Input image tensor [batch, 3, H, W] with requires_grad=True
    
    Returns:
        dict with:
            'bboxes': Tensor of shape [batch, num_detections, 4] - bounding boxes (x, y, w, h)
            'class_probs': Tensor of shape [batch, num_detections, 80] - class probabilities for each bbox
            'num_detections': int - number of detections
            
    Note: All tensors maintain gradient flow for adversarial attacks
    """
    with torch.set_grad_enabled(True):
        output = model.model(img_tensor)
        
        # In eval mode: output is tuple, first element is [batch, 84, num_detections]
        # where 84 = 4 (bbox coords) + 80 (class scores)
        predictions = output[0]  # [batch, 84, num_detections]
        
        batch_size = predictions.shape[0]
        num_detections = predictions.shape[2]
        
        # Split into bounding boxes and class scores
        bboxes = predictions[:, :4, :]  # [batch, 4, num_detections]
        class_scores = predictions[:, 4:, :]  # [batch, 80, num_detections]
        
        # Transpose to more intuitive shape: [batch, num_detections, features]
        bboxes = bboxes.permute(0, 2, 1)  # [batch, num_detections, 4]
        class_probs = class_scores.permute(0, 2, 1)  # [batch, num_detections, 80]
        
    return {
        'bboxes': bboxes,
        'class_probs': class_probs,
        'num_detections': num_detections
    }



def xywh2xyxy(boxes_xywh):
    """
    Convert boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2).
    
    Args:
        boxes_xywh: Tensor [..., 4] with (cx, cy, w, h)
    
    Returns:
        boxes_xyxy: Tensor [..., 4] with (x1, y1, x2, y2)
    """
    boxes_xyxy = boxes_xywh.clone()
    boxes_xyxy[..., 0] = boxes_xywh[..., 0] - boxes_xywh[..., 2] / 2  # x1
    boxes_xyxy[..., 1] = boxes_xywh[..., 1] - boxes_xywh[..., 3] / 2  # y1
    boxes_xyxy[..., 2] = boxes_xywh[..., 0] + boxes_xywh[..., 2] / 2  # x2
    boxes_xyxy[..., 3] = boxes_xywh[..., 1] + boxes_xywh[..., 3] / 2  # y2
    return boxes_xyxy


def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor [N, 4] in xyxy format
        boxes2: Tensor [M, 4] in xyxy format
    
    Returns:
        iou: Tensor [N, M] with IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def adv_loss_calc(batch_frames, orig_clases, iou_thresh=0.5):
    """
    Calculate adversarial loss using IoU-based detection scoring approach.
    
    This loss uses raw YOLO11 predictions and combines objectness + class scores
    for detections of the original classes. It uses log-sum-exp for smooth max.
    
    FULLY DIFFERENTIABLE - uses logits directly for better gradient flow.
    
    Args:
        batch_frames: Input images [batch, 3, H, W]
        orig_clases: Tensor of original class indices to suppress
        iou_thresh: IoU threshold for considering high-overlap detections (default: 0.5)
        
    Returns:
        total_loss: Scalar loss value to minimize
    """
    # Get raw predictions from YOLO model - gradients flow through this
    # Use torch.set_grad_enabled to ensure gradients flow even in eval mode
    with torch.set_grad_enabled(True):
        output = model.model(batch_frames)
        predictions = output[0]  # [batch, 84, num_detections]
    
    B, no_minus_nc, N = predictions.shape
    nc = 80  # Number of classes
    
    # Reshape to [B, N, 84] for easier manipulation
    preds = predictions.permute(0, 2, 1)  # [B, N, 84]
    
    # Split into components
    boxes_xywh = preds[..., :4]      # [B, N, 4]
    obj_logits = preds[..., 4]       # [B, N]
    cls_logits = preds[..., 5:5+nc]  # [B, N, 80]
    
    # Extract logits for original classes BEFORE sigmoid for better gradients
    orig_cls_logits = cls_logits[:, :, orig_clases]  # [B, N, len(orig_clases)]
    
    # Get max logit across original classes for each detection
    max_orig_logits, _ = orig_cls_logits.max(dim=2)  # [B, N]
    
    # Combined detection score using logits (better gradients than probabilities)
    # Use logsumexp for numerical stability
    # obj_logits + max_orig_logits approximates log(sigmoid(obj) * sigmoid(cls))
    combined_logits = obj_logits + max_orig_logits  # [B, N]
    
    # Multi-component loss for stronger signal
    # IMPORTANT: We want to MINIMIZE loss, so we NEGATE to suppress detections
    
    # Component 1: Suppress max detection (strongest signal)
    max_detection = combined_logits.view(B, N).max(dim=1)[0]  # [B]
    loss_max = -10.0 * max_detection.mean()  # NEGATIVE to minimize
    
    # Component 2: Suppress all high-confidence detections
    # Use soft threshold with sigmoid
    threshold_logit = 0.0  # corresponds to ~0.5 probability
    high_conf_scores = torch.sigmoid(combined_logits - threshold_logit)  # [B, N]
    loss_count = -5.0 * high_conf_scores.sum() / (B * N)  # NEGATIVE to minimize
    
    # Component 3: Global suppression via log-sum-exp (smooth max)
    loss_logsumexp = -2.0 * torch.logsumexp(combined_logits.view(-1), dim=0) / (B * N)  # NEGATIVE to minimize
    
    # Total loss (all negative components)
    total_loss = loss_max + loss_count + loss_logsumexp
    
    return total_loss


def check_orig_class_detected(img_tensor, orig_clases, conf_threshold=0.25):
    """
    Check if original class is detected in the image (based on bounding boxes).
    
    Args:
        img_tensor: Input image tensor [batch, 3, H, W]
        orig_clases: Tensor of original class indices to check
        conf_threshold: Minimum confidence to consider a detection valid (default 0.25)
        
    Returns:
        detected: Boolean tensor [batch] - True if orig class detected, False otherwise
    """
    with torch.no_grad():
        result = get_bboxes_with_class_probs(model, img_tensor)
        probs = result['class_probs']  # [batch, num_detections, 80]
        
        # Get probabilities for original classes
        orig_cls_probs = probs[:, :, orig_clases]  # [batch, num_detections, len(orig_clases)]
        
        # Find maximum original class probability for each detection
        max_orig_probs, _ = orig_cls_probs.max(dim=2)  # [batch, num_detections]
        
        # Check if any detection has original class above threshold
        detected = (max_orig_probs > conf_threshold).any(dim=1)  # [batch]
        
    return detected

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
        # logits, _ = class_scores.max(dim=2)  # [batch, 80]
        logits = class_scores.mean(dim=2)  # [batch, 80]
    
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
