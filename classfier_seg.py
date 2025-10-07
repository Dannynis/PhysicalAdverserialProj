from IPython.core.display import Image
from torchvision.io import read_image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
import torchvision
import torch

weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_mobilenet_v3_large(weights=weights)
model = model.eval().cuda()

preprocess = weights.transforms()
#input image w x h x c

def deeplabv3_predict(image):
    with torch.no_grad():
        segmentation = predict_raw(image)
        # DeepLabV3 outputs segmentation maps, we'll get the most common class
        if segmentation.shape[0] == 1:
            seg_map = segmentation.squeeze(0)  # Remove batch dimension
            # Get the most frequent class in the segmentation map
            unique_classes, counts = torch.unique(seg_map.argmax(0), return_counts=True)
            dominant_class = unique_classes[counts.argmax()].item()
            category_name = weights.meta["categories"][dominant_class]
            return f"{category_name}: segmentation"
        res_lst = []
        for seg in segmentation:
            seg_map = seg.argmax(0)
            unique_classes, counts = torch.unique(seg_map, return_counts=True)
            dominant_class = unique_classes[counts.argmax()].item()
            category_name = weights.meta["categories"][dominant_class]
            res_lst.append(category_name)
        return res_lst


def predict_raw(image):
    # Apply inference preprocessing transforms
    batch = preprocess(image)

    # DeepLabV3 returns a dictionary with 'out' key containing segmentation logits
    output = model(batch)
    segmentation_logits = output['out']
    
    # For compatibility with adversarial loss functions, we'll convert to classification-like format
    # by averaging the segmentation logits across spatial dimensions
    # Average across spatial dimensions to get class scores
    class_scores = segmentation_logits.mean(dim=[2, 3])  # Shape: [batch_size, num_classes]
    
    return class_scores.softmax(-1)


def predict_raw_segmentation(image):
    """
    Returns the raw segmentation output without spatial averaging.
    This preserves the spatial structure for segmentation-specific attacks.
    """
    # Apply inference preprocessing transforms
    batch = preprocess(image)

    # DeepLabV3 returns a dictionary with 'out' key containing segmentation logits
    output = model(batch)
    segmentation_logits = output['out']
    
    return segmentation_logits


def get_dominant_classes_per_pixel(image):
    """
    Returns the predicted class for each pixel in the segmentation map.
    """
    with torch.no_grad():
        segmentation_logits = predict_raw_segmentation(image)
        predicted_classes = segmentation_logits.argmax(dim=1)  # [batch_size, H, W]
        return predicted_classes


batch_size = 1
# DeepLabV3 has 21 classes (VOC+COCO), map to relevant segmentation classes
# VOC classes: background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, 
#              cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
# Using indices for: person, car, bus, train, motorbike, bicycle, etc.
orig_clases = torch.tensor([817, 705, 609, 586, 436, 627, 468, 621, 803, 407, 408, 751, 717,866, 661]).cuda()
total_clases_without_orig = torch.tensor([x for x in list(range(0, 21)) if x not in orig_clases]).cuda()


def adv_loss_calc(image):
    """
    Adversarial loss calculation for segmentation model.
    Uses spatially averaged class scores for compatibility.
    """
    assert len(image.shape) == 4, "Image should be of shape (batch_size, 3, h, w)"
    adv_loss = []
    pred = predict_raw(image)
    for p in pred:
        adv_loss.append(p[orig_clases].mean())
    return torch.stack(adv_loss)


def adv_loss_calc2(image):
    """
    Alternative adversarial loss calculation for segmentation model.
    Uses max strategy instead of mean.
    """
    adv_loss = []
    pred = predict_raw(image)
    for p in pred:
        forbiden = p[orig_clases].max()
        allowed = p[total_clases_without_orig].max()
        adv_loss.append(forbiden - allowed)
    return torch.stack(adv_loss)


def adv_loss_calc_segmentation(image):
    """
    Segmentation-specific adversarial loss that considers spatial structure.
    Penalizes pixels that are classified as forbidden classes.
    """
    assert len(image.shape) == 4, "Image should be of shape (batch_size, 3, h, w)"
    
    segmentation_logits = predict_raw_segmentation(image)
    batch_size, num_classes, height, width = segmentation_logits.shape
    
    # Convert to probabilities
    seg_probs = torch.softmax(segmentation_logits, dim=1)
    
    adv_loss = []
    for b in range(batch_size):
        # Get probabilities for forbidden classes at each pixel
        forbidden_probs = seg_probs[b, orig_clases, :, :].sum(dim=0)  # [H, W]
        
        # Calculate loss as mean probability of forbidden classes across all pixels
        pixel_loss = forbidden_probs.mean()
        adv_loss.append(pixel_loss)
    
    return torch.stack(adv_loss)


def adv_loss_calc_segmentation_weighted(image, target_region_mask=None):
    """
    Weighted segmentation adversarial loss that can focus on specific regions.
    
    Args:
        image: Input image tensor [batch_size, 3, H, W]
        target_region_mask: Optional mask [batch_size, 1, H, W] to weight different regions
                           If None, uniform weighting is used
    """
    assert len(image.shape) == 4, "Image should be of shape (batch_size, 3, h, w)"
    
    segmentation_logits = predict_raw_segmentation(image)
    batch_size, num_classes, height, width = segmentation_logits.shape
    
    # Convert to probabilities
    seg_probs = torch.softmax(segmentation_logits, dim=1)
    
    adv_loss = []
    for b in range(batch_size):
        # Get probabilities for forbidden classes at each pixel
        forbidden_probs = seg_probs[b, orig_clases, :, :].sum(dim=0)  # [H, W]
        
        if target_region_mask is not None:
            # Apply regional weighting
            region_mask = target_region_mask[b, 0, :, :]  # [H, W]
            weighted_loss = (forbidden_probs * region_mask).sum() / region_mask.sum()
        else:
            # Uniform weighting
            weighted_loss = forbidden_probs.mean()
            
        adv_loss.append(weighted_loss)
    
    return torch.stack(adv_loss)