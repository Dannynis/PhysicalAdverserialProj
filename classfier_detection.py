import torch
from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.3)
model = model.cuda().eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

orig_clases = torch.tensor([3,8]).cuda()
total_clases_without_orig = torch.tensor([x for x in list(range(0, 1000)) if x not in orig_clases]).cuda()


def predict_raw(image):
    # Apply inference preprocessing transforms
    preds_lst = []
    pr = model(image.cuda())
    for pred in pr:
        clases_scores = torch.zeros(len(weights.meta["categories"])).cuda()
        for idx in range(len(pred['labels'])):
            label,score = pred['labels'][idx], pred['scores'][idx]
            clases_scores[label] = score
        preds_lst.append(clases_scores)
    preds_lst = torch.stack(preds_lst)

    return preds_lst