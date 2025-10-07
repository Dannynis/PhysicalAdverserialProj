import torch
import clip
from PIL import Image
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import cv2
import torchvision

orig_clases = torch.tensor([817, 705, 609, 586, 436, 627, 468, 621, 803, 407, 408, 751, 717,866, 661]).cuda()
total_clases_without_orig = torch.tensor([x for x in list(range(0, 1000)) if x not in orig_clases]).cuda()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
weights = EfficientNet_B4_Weights.IMAGENET1K_V1
categories = ['a photo of ' + c for c in weights.meta["categories"]]

del preprocess.transforms[2]
del preprocess.transforms[2]

with torch.no_grad():
    cat_tokens = clip.tokenize(categories).to(device)
    categories_vecs = model.encode_text(cat_tokens)

def predict_raw(x):
    vec = model.encode_image(preprocess(x).to(device))
    logits = categories_vecs @ vec.T 
    return logits.T 