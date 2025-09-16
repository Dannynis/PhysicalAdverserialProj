import torch
import torch.nn as nn
import torchvision.models as models

import glob
import tqdm
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import time
from tqdm import tqdm
import copy
from IPython.display import display, Image, clear_output
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import kornia
import tqdm

import torchvision

tt = torchvision.transforms.ToTensor()

import cv2
import cv2.aruco as aruco
import numpy as np

import pickle as pkl

with open("photometric_calibration.pkl", "rb") as f:
    data = pkl.load(f)

height = data['height']
width = data['width']

resizer = torchvision.transforms.Resize((400, we))


# Define parameters for ArUco marker detection
aruco_dict_type = cv2.aruco.DICT_6X6_250 # Change dictionary type if needed
marker_length = 0.05  # Marker length in meters (adjust as needed)
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

marker_id = 40
marker_size = 500  # Size in pixels
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)


aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
parameters = aruco.DetectorParameters()

# Detect ArUco markers
detector = aruco.ArucoDetector(aruco_dict, parameters)


from diffusers import StableDiffusionPipeline
import torch

# Load stable diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae=  pipe.vae.to(device).eval()

def decode_latents_grad(latents):
    # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
    latents = 1 / 0.18215 * latents

    imgs = vae.decode(latents).sample

    imgs = (imgs / 2 + 0.5).clamp(0, 1)

    return imgs

def decode_latents(latents):
    # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
    with torch.no_grad():
        with torch.amp.autocast(device):
            latents = 1 / 0.18215 * latents

            with torch.no_grad():
                imgs = vae.decode(latents).sample

            imgs = (imgs / 2 + 0.5).clamp(0, 1)

    return imgs

def encode_imgs(imgs):
    # imgs: [B, 3, H, W]
    with torch.no_grad():
        with torch.amp.autocast(device):
            imgs = 2 * imgs - 1

            posterior = vae.encode(imgs).latent_dist
            latents = posterior.sample() * 0.18215

    return latents


latent = (torch.rand((1,4, 4, 4), device=device) - 0.5) * 2
with torch.no_grad():
    decoded_latents = resizer(decode_latents(latent))


l_size_h = decoded_latents.shape[-2]
l_size_w = decoded_latents.shape[-1]

orig_img_corners = np.array([[0,0],[l_size_w,0],[l_size_w,l_size_h],[0,l_size_h]], dtype=np.float32)


import cv2
import os
import glob

border_size = 0

ls = os.listdir('.')
captures = [f for f in ls if f.startswith('captures_frames_multiview_') ]
cap_dir = f'captures_frames_multiview_{len(captures)}'

valid_frame_paths = glob.glob(f'{cap_dir}/*.png')

valid_frames = []

Hs = []

for path in tqdm.tqdm_notebook(valid_frame_paths):
    img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    with torch.no_grad():
        pr = resnet_predict_raw(tt(img).cuda().unsqueeze(0))

    if ids is not None and pr.argmax(1) in orig_clases:

        c = corners[0][0]
        unbordred_corners = np.array([[c[0][0]-border_size, c[0][1]+border_size],
                              [c[1][0]-border_size, c[1][1]-border_size],
                              [c[2][0]+border_size, c[2][1]-border_size],
                              [c[3][0]+border_size, c[3][1]+border_size]])

        dst_pts = unbordred_corners
        H, _ = cv2.findHomography(orig_img_corners, dst_pts, cv2.RANSAC)

        Hs.append(H)
        valid_frames.append(img)

print(f"Found {len(valid_frames)} valid frames with ArUco markers and original classes.")


class framesDataset(Dataset):
    def __init__(self, frames, Hs):
        self.frames = frames
        self.Hs = Hs

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        H = self.Hs[idx]

        # Convert to tensor and normalize
        frame_tensor = tt(frame)

        return frame_tensor, H.astype(np.float32)
    





def warp(decoded_latents,H_t):
    warped_imgs = []
    for decoded_latent in decoded_latents:
        img = decoded_latent.unsqueeze(0).float().repeat(H_t.shape[0], 1, 1, 1)
        w=  kornia.geometry.transform.warp_perspective(img, H_t, (data['height'], data['width']))
        warped_imgs.append(w)
    return torch.stack(warped_imgs, dim=0)#.squeeze(1)



def optimize_patch():
    ds = framesDataset(valid_frames, Hs)


    train,test = torch.utils.data.random_split(ds, [int(len(ds)*0.8), len(ds)-int(len(ds)*0.8)])
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=10, shuffle=False, num_workers=2)

        
    latent.requires_grad = True


    latent_opt = torch.optim.Adam([latent], lr=0.1)


    


    jitter = T.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1)
    jitter_total_photo = T.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1)
    jitter_with_hue = T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.1)

    augmentor = data['augmentor'].to(device).eval()
    mapper = lambda x: jitter(augmentor(x))
    # mapper = lambda x: jitter(x)