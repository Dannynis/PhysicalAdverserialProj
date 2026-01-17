#!/usr/bin/env python3
"""
Ablation Study Experiment Runner

This script runs a single ablation experiment and saves all results to a directory.
Multiple experiments can run in parallel by specifying different experiment names.

Usage:
    python run_ablation_experiment.py --experiment latent_4x4
    python run_ablation_experiment.py --experiment latent_8x8
    python run_ablation_experiment.py --experiment latent_16x16
    python run_ablation_experiment.py --experiment latent_32x32
    python run_ablation_experiment.py --experiment latent_16x16_with_rejuv

    # Run with custom parameters:
    python run_ablation_experiment.py --experiment latent_8x8 --epochs 50 --batch-size 16

    # Run all experiments sequentially:
    python run_ablation_experiment.py --all

    # Specify GPU device for parallel runs:
    python run_ablation_experiment.py --experiment latent_4x4 --gpu 0
    python run_ablation_experiment.py --experiment latent_8x8 --gpu 1
"""

import argparse
import os
import sys
import datetime
import pickle
import gc
import json
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Change to project root
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import kornia
from tqdm import tqdm
import pickle as pkl
from functools import partial
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENT_CONFIGS = {
    "latent_4x4": {"latent_size": 4, "rejuvenate": False},
    "latent_8x8": {"latent_size": 8, "rejuvenate": False},
    "latent_16x16": {"latent_size": 16, "rejuvenate": False},
    "latent_32x32": {"latent_size": 32, "rejuvenate": False},
    "latent_16x16_with_rejuv": {"latent_size": 16, "rejuvenate": True},
}

# Default training parameters
DEFAULT_NUM_EPOCHS = 30
DEFAULT_BATCH_SIZE = 10
DEFAULT_BLEND_RATIO = 1.0
DEFAULT_CLASSIFIER_WEIGHTS = {
    'inception': 0.25,
    'resnet': 0.25,
    'vgg': 0.25,
    'vit': 0.25,
    'dino': 0.0  # Disabled to save GPU memory
}

# ============================================================================
# HELPER CLASSES AND FUNCTIONS
# ============================================================================

class framesDataset(Dataset):
    """Dataset for captured frames with homography matrices."""
    def __init__(self, frames, Hs):
        self.frames = frames
        self.Hs = Hs
        self.tt = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        H = self.Hs[idx]
        frame_tensor = self.tt(frame)
        return frame_tensor, H.astype(np.float32)


def find_border_drop_point(gray, c):
    """Find border drop point for ArUco corner refinement."""
    sub = np.subtract
    add = np.add
    borders_drop_points = []
    for idx, operators in enumerate(([sub, sub], [add, sub], [add, add], [sub, add])):
        margin = 1
        a, b = int(c[idx][0]), int(c[idx][1])
        diag_idxs = np.arange(5)
        nca = operators[0](a, diag_idxs)
        ncb = operators[1](b, diag_idxs)
        nc = np.stack([nca, ncb], axis=1)
        diag_line_vals = gray[nc[:, 1], nc[:, 0]].astype(np.float32)
        diag_line_vals_diff = np.diff(diag_line_vals)
        if np.all(diag_line_vals_diff >= 0):
            borders_drop_points.append((nca[0], ncb[0]))
            continue
        diag_line_vals_diff_first_neg = min(np.where(diag_line_vals_diff < 0)[0][0] + margin, len(diag_line_vals_diff)-1)
        new_a = nca[diag_line_vals_diff_first_neg]
        new_b = ncb[diag_line_vals_diff_first_neg]
        borders_drop_points.append((new_a, new_b))
    return np.array(borders_drop_points)


class AblationExperiment:
    """
    Main class for running a single ablation experiment.
    Designed to be self-contained and parallelizable.
    """
    
    def __init__(self, exp_name, config, args, output_dir):
        self.exp_name = exp_name
        self.config = config
        self.args = args
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if args.gpu is not None:
            self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.vae = None
        self.predict_raw = None
        self.predict_raw_dev = None
        self.orig_clases = None
        self.augmentor_model = None
        self.height = None
        self.width = None
        self.valid_frames = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Results tracking
        self.results = {
            "exp_name": exp_name,
            "config": config,
            "losses": [],
            "success_rates": [],
            "aug_success_rates": [],
            "best_success_rate": 0,
            "final_aug_rate": 0,
            "epochs_to_50_percent": None,
            "total_epochs_run": 0,
        }
        
        # Comet ML experiment (optional)
        self.comet_experiment = None
        
    def setup(self):
        """Initialize all components needed for the experiment."""
        print("=" * 60)
        print(f"Setting up experiment: {self.exp_name}")
        print("=" * 60)
        
        self._load_classifier()
        self._load_vae()
        self._load_photometric_calibration()
        self._load_frames()
        self._setup_augmentations()
        self._setup_comet()
        
        print(f"Setup complete. Ready to train.")
        
    def _load_classifier(self):
        """Load classifier ensemble."""
        print("Loading classifier ensemble...")
        from classfier_ensemble import predict_raw, orig_clases, model, weights
        from classfier_ensemble import predict_raw as predict_raw_dev
        
        self.orig_clases = orig_clases
        self.weights = weights
        self.model_name = model if type(model) == str else model.__class__.__name__
        
        # Apply classifier weights
        if 'weight_dict' in predict_raw.__code__.co_varnames:
            self.predict_raw = partial(predict_raw, weights_dict=DEFAULT_CLASSIFIER_WEIGHTS)
            self.predict_raw_dev = partial(predict_raw_dev, weights_dict=DEFAULT_CLASSIFIER_WEIGHTS)
        else:
            self.predict_raw = predict_raw
            self.predict_raw_dev = predict_raw_dev
            
        print(f"Classifier: {self.model_name}")
        print(f"Original classes: {self.orig_clases}")
        
    def _load_vae(self):
        """Load Stable Diffusion VAE for latent decoding."""
        print("Loading Stable Diffusion VAE...")
        from diffusers import StableDiffusionPipeline
        
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        self.vae = pipe.vae.to(self.device).eval()
        
        # Clean up pipeline to save memory
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        
    def _load_photometric_calibration(self):
        """Load photometric calibration data."""
        print("Loading photometric calibration...")
        photometric_calibrations_dir = './photometric_calibrations'
        photometric_calibration_files = glob.glob(
            os.path.join(photometric_calibrations_dir, 'photometric_calibration_*.pkl')
        )
        photometric_calibration_files.sort(key=os.path.getmtime, reverse=True)
        photometric_calibration_path = photometric_calibration_files[0]
        
        with open(photometric_calibration_path, "rb") as f:
            data = pkl.load(f)
        
        self.height = data['height']
        self.width = data['width']
        self.augmentor_model = data['augmentor'].to(self.device).eval()
        
        print(f"Loaded calibration: {self.height}x{self.width}")
        
    def _load_frames(self):
        """Load and process captured frames."""
        print("Loading captured frames...")
        from consts import border_size, displayed_aruco_code, marker_size as ms
        
        self.border_size = border_size
        self.displayed_aruco_code = displayed_aruco_code
        
        # ArUco setup
        aruco_dict_type = cv2.aruco.DICT_4X4_50
        aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        
        tt = torchvision.transforms.ToTensor()
        
        # Find latest capture directory
        caps_dir = 'captures_frames_multiview'
        ls = os.listdir(f'./{caps_dir}')
        captures = [f for f in ls if f.startswith('captures_frames_multiview_')]
        captures = sorted(captures, key=lambda x: int(x.split('_')[-1]))
        cap_dir = f'./{caps_dir}/{captures[-1]}'
        
        print(f"Loading frames from: {cap_dir}")
        files_sorted = sorted(
            glob.glob(f'{cap_dir}/*.png'),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        frames = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files_sorted]
        print(f"Loaded {len(frames)} frames")
        
        # Process frames and calculate homographies
        valid_frames = []
        H_list = []
        
        orig_img_corners = np.array([
            [border_size, border_size],
            [self.width - border_size, border_size],
            [self.width - border_size, self.height - border_size],
            [border_size, self.height - border_size]
        ], dtype=np.float32)
        
        orig_clases_np = self.orig_clases.cpu().numpy()
        
        for idx, frame in tqdm(enumerate(frames), desc="Processing frames", total=len(frames)):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if ids is None:
                continue
            
            # Check classifier prediction
            with torch.no_grad():
                pr = self.predict_raw(tt(frame).to(self.device).unsqueeze(0))
            
            if displayed_aruco_code in ids and pr.argmax(1).item() in orig_clases_np:
                displayed_aruco_code_index = np.where(ids.flatten() == displayed_aruco_code)[0][0]
                c = corners[displayed_aruco_code_index][0]
                
                try:
                    unbordred_corners = find_border_drop_point(gray, c)
                except:
                    continue
                
                H, _ = cv2.findHomography(orig_img_corners, unbordred_corners, cv2.RANSAC)
                
                if H is not None:
                    valid_frames.append(frame)
                    H_list.append(H)
        
        print(f"Found {len(valid_frames)} valid frames")
        
        # Shuffle and limit
        random_idx = np.random.permutation(min(len(valid_frames), 5000))
        valid_frames = [valid_frames[i] for i in random_idx]
        H_list = [H_list[i] for i in random_idx]
        
        self.valid_frames = valid_frames
        
        if len(valid_frames) == 0:
            raise ValueError("No valid frames found!")
        
        # Create datasets
        train_split = int(len(valid_frames) * 0.7)
        val_split = int(len(valid_frames) * 0.85)
        
        train_dataset = framesDataset(valid_frames[:train_split], H_list[:train_split])
        val_dataset = framesDataset(valid_frames[train_split:val_split], H_list[train_split:val_split])
        test_dataset = framesDataset(valid_frames[val_split:], H_list[val_split:])
        
        batch_size = self.args.batch_size
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
    def _setup_augmentations(self):
        """Setup augmentation transforms."""
        self.jitter = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ])
        
        self.jitter_total_photo = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        ])
        
    def _setup_comet(self):
        """Setup Comet ML experiment tracking (optional)."""
        if self.args.no_comet:
            print("Comet ML tracking disabled.")
            return
            
        try:
            from comet_ml import start
            
            self.comet_experiment = start(
                api_key="Bg5eubpUjdi2CCiiA5OSoltfw",
                project_name="physicaladvproj-ablation",
                workspace="dannynis"
            )
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
            exp_full_name = f"ablation_{self.exp_name}_{timestamp}"
            self.comet_experiment.set_name(exp_full_name)
            
            self.comet_experiment.log_parameters({
                "experiment": self.exp_name,
                "latent_size": self.config["latent_size"],
                "rejuvenate": self.config["rejuvenate"],
                "num_epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                **DEFAULT_CLASSIFIER_WEIGHTS
            })
            
            print(f"Comet ML experiment: {exp_full_name}")
        except Exception as e:
            print(f"Could not setup Comet ML: {e}")
            self.comet_experiment = None
            
    def decode_latents_grad(self, latents):
        """Decode latents with gradient tracking."""
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    
    def decode_latents(self, latents):
        """Decode latents without gradient tracking."""
        with torch.no_grad():
            with torch.amp.autocast(self.device):
                latents = 1 / 0.18215 * latents
                imgs = self.vae.decode(latents).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    
    def warp(self, decoded_latents, H_t):
        """Warp patches to frame perspective."""
        dst_img_shape = self.valid_frames[0].shape[:2]
        warped_imgs = []
        for decoded_latent in decoded_latents:
            img = decoded_latent.unsqueeze(0).float().repeat(H_t.shape[0], 1, 1, 1)
            w = kornia.geometry.transform.warp_perspective(img, H_t, dst_img_shape)
            warped_imgs.append(w)
        return torch.stack(warped_imgs, dim=0)
    
    def run(self):
        """Run the full experiment."""
        print("\n" + "=" * 60)
        print(f"🧪 Running Experiment: {self.exp_name}")
        print(f"   Latent Size: {self.config['latent_size']}x{self.config['latent_size']}")
        print(f"   Rejuvenation: {'Enabled' if self.config['rejuvenate'] else 'Disabled'}")
        print(f"   Epochs: {self.args.epochs}")
        print("=" * 60 + "\n")
        
        latent_size = self.config["latent_size"]
        to_rejuvenate = self.config["rejuvenate"]
        num_epochs = self.args.epochs
        num_patches = self.args.batch_size
        blend_ratio = DEFAULT_BLEND_RATIO
        
        # Initialize latent batch
        latent_batch = torch.randn(
            (num_patches, 4, latent_size, latent_size), 
            device=self.device
        ) * 0.8
        latent_batch.requires_grad = True
        
        # Resizer
        resizer = torchvision.transforms.Resize((self.height, self.width))
        
        # Optimizer
        latent_opt = torch.optim.Adam([latent_batch], lr=0.1)
        
        # Training state
        orig_clases_np = self.orig_clases.cpu().numpy()
        best_latent = None
        best_success_rate = 0
        training_stopped = False
        aug_weight = 0.9
        epochs_to_50_percent = None
        target_classes = torch.tensor([], device=self.device)
        
        # Augmentor function
        def get_augmentor():
            return lambda x: self.augmentor_model(x).to(self.device) * aug_weight + x * (1 - aug_weight)
        augmentor = get_augmentor()
        
        losses = []
        success_rates = []
        aug_success_rates = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_success_rates = []
            patch_success_history = {i: [] for i in range(num_patches)}
            patch_augmented_success_history = {i: [] for i in range(num_patches)}
            
            torch.cuda.empty_cache()
            
            # Training loop
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                       desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            
            for batch_idx, (frames_batch, H_t_batch) in pbar:
                latent_opt.zero_grad()
                
                frames_batch = frames_batch.to(self.device)
                H_t_batch = H_t_batch.to(self.device)
                
                # Generate adversarial patch
                adv_patch = resizer(self.decode_latents_grad(latent_batch).float())
                
                # Apply augmentations
                if torch.rand(1).item() > 0.3:
                    adv_patch_aug = self.jitter(adv_patch)
                else:
                    adv_patch_aug = adv_patch
                    
                if torch.rand(1).item() > 0.3:
                    adv_patch_aug = torch.stack([augmentor(x).to(self.device) for x in adv_patch_aug])
                
                # Warp and blend
                w_mask = self.warp(adv_patch_aug * 0 + 1, H_t_batch)
                w_patch = self.warp(adv_patch_aug, H_t_batch)
                blended_frames = ((w_mask != 0) * -blend_ratio + 1) * frames_batch + w_patch * blend_ratio
                blended_frames = blended_frames.view(-1, 3, blended_frames.shape[-2], blended_frames.shape[-1])
                
                if torch.rand(1).item() > 0.3:
                    blended_frames = self.jitter_total_photo(blended_frames)
                
                # Get predictions
                with torch.autocast(device_type=self.device.split(':')[0]):
                    logits = self.predict_raw(blended_frames)
                    if (logits != logits).any():
                        raise ValueError("NaN in logits")
                    probs = torch.softmax(logits, dim=1)
                
                # Loss calculation
                orig_class_probs = probs[:, self.orig_clases]
                orig_loss = 5.0 * torch.log(orig_class_probs.sum(dim=1) + 1e-10).mean()
                
                if target_classes.numel() > 0:
                    target_probs = probs[:, target_classes]
                    target_loss = -3.0 * torch.log(target_probs.max(dim=1)[0] + 1e-10).mean()
                else:
                    target_loss = 0
                
                total_loss = orig_loss + target_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_([latent_batch], max_norm=3.0)
                latent_opt.step()
                
                epoch_losses.append(total_loss.item())
                
                # Evaluation
                with torch.no_grad():
                    predictions = logits.argmax(dim=1)
                    successful_attacks = sum(pred.item() not in orig_clases_np for pred in predictions)
                    success_rate = successful_attacks / len(predictions)
                    epoch_success_rates.append(success_rate)
                
                pbar.set_postfix({"loss": f"{total_loss.item():.3f}", "success": f"{success_rate:.1%}"})
            
            # Epoch stats
            avg_epoch_loss = np.mean(epoch_losses)
            avg_epoch_success = np.percentile(epoch_success_rates, 90)
            
            losses.append(avg_epoch_loss)
            success_rates.append(avg_epoch_success)
            
            if epochs_to_50_percent is None and avg_epoch_success >= 0.5:
                epochs_to_50_percent = epoch
            
            # Log to Comet
            if self.comet_experiment:
                self.comet_experiment.log_metric(f"loss", avg_epoch_loss, step=epoch)
                self.comet_experiment.log_metric(f"success_rate", avg_epoch_success, step=epoch)
                self.comet_experiment.log_metric(f"aug_weight", aug_weight, step=epoch)
            
            # Per-patch evaluation
            if epoch % 5 == 0 or avg_epoch_success > 0.3:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    all_patches = resizer(self.decode_latents(latent_batch).float())
                    val_batch_limit = min(2 if latent_size >= 16 else 3, len(self.val_loader))
                    
                    for patch_idx in range(num_patches):
                        patch_clean_successes = 0
                        patch_aug_successes = 0
                        total_tests = 0
                        single_patch = all_patches[patch_idx:patch_idx+1]
                        
                        for val_batch_idx, (val_frames, val_H_t) in enumerate(self.val_loader):
                            if val_batch_idx >= val_batch_limit:
                                break
                            
                            val_frames = val_frames.to(self.device)
                            val_H_t = val_H_t.to(self.device)
                            
                            # Clean patch test
                            w_mask = self.warp(single_patch * 0 + 1, val_H_t)
                            w_patch = self.warp(single_patch, val_H_t)
                            clean_blended = ((w_mask != 0) * -blend_ratio + 1) * val_frames + w_patch * blend_ratio
                            clean_batch = clean_blended.view(-1, *clean_blended.shape[2:])
                            
                            clean_logits = self.predict_raw_dev(clean_batch)
                            clean_predictions = clean_logits.argmax(dim=1)
                            
                            # Augmented patch test
                            aug_patch = self.jitter(single_patch)
                            try:
                                aug_patch = torch.stack([augmentor(x).to(self.device) for x in aug_patch])
                            except:
                                pass
                            w_mask_aug = self.warp(aug_patch * 0 + 1, val_H_t)
                            w_patch_aug = self.warp(aug_patch, val_H_t)
                            aug_blended = ((w_mask_aug != 0) * -blend_ratio + 1) * val_frames + w_patch_aug * blend_ratio
                            aug_blended = aug_blended.squeeze(0)
                            aug_blended = self.jitter_total_photo(aug_blended)
                            aug_batch = aug_blended.view(-1, *aug_blended.shape[1:])
                            
                            aug_logits = self.predict_raw_dev(aug_batch)
                            aug_predictions = aug_logits.argmax(dim=1)
                            
                            for pred in clean_predictions:
                                if pred.item() not in orig_clases_np:
                                    patch_clean_successes += 1
                                total_tests += 1
                            
                            for pred in aug_predictions:
                                if pred.item() not in orig_clases_np:
                                    patch_aug_successes += 1
                        
                        clean_rate = patch_clean_successes / total_tests if total_tests > 0 else 0
                        aug_rate = patch_aug_successes / total_tests if total_tests > 0 else 0
                        
                        patch_success_history[patch_idx].append(clean_rate)
                        patch_augmented_success_history[patch_idx].append(aug_rate)
                    
                    if len(patch_success_history[0]) > 0:
                        print(f"\n🏆 TOP 5 PATCHES (Augmented Performance):")
                        
                        latest_aug_performance = [(i, patch_augmented_success_history[i][-1]) for i in range(num_patches)]
                        latest_aug_performance.sort(key=lambda x: x[1], reverse=True)
                        best_patch_idx, best_patch_rate = latest_aug_performance[0]
                        
                        for rank, (patch_idx, aug_rate_val) in enumerate(latest_aug_performance[:5], 1):
                            clean_rate = patch_success_history[patch_idx][-1]
                            robustness = (aug_rate_val / clean_rate) if clean_rate > 0 else 0
                            print(f"  {rank}. Patch #{patch_idx+1:2d}: Clean {clean_rate:.1%} | Aug {aug_rate_val:.1%} | Robust {robustness:.1%}")
                        
                        print(f"📉 BOTTOM 3 PATCHES (Augmented Performance):")
                        for rank, (patch_idx, aug_rate_val) in enumerate(latest_aug_performance[-3:], 1):
                            clean_rate = patch_success_history[patch_idx][-1]
                            print(f"  {rank}. Patch #{patch_idx+1:2d}: Clean {clean_rate:.1%} | Aug {aug_rate_val:.1%}")
                        
                        aug_success_rates.append(best_patch_rate)
                        
                        if self.comet_experiment:
                            self.comet_experiment.log_metric("best_aug_rate", best_patch_rate, step=epoch)
                        
                        # Incremental aug_weight update
                        if torch.mean(torch.tensor([x[1] for x in latest_aug_performance[:1]])) > 0.7:
                            if aug_weight < 1:
                                if aug_weight < 0.7:
                                    aug_weight += 0.1
                                elif aug_weight < 0.9:
                                    aug_weight += 0.05
                                else:
                                    aug_weight += 0.01
                                aug_weight = min(aug_weight, 1.0)
                                print(f"Augmentation weight increased to {aug_weight:.2f}")
                                augmentor = get_augmentor()
                        
                        # Early stopping
                        if aug_weight >= 1 and best_patch_rate >= 0.9:
                            print(f"\n🎉 BREAKTHROUGH! Patch #{best_patch_idx+1} achieved {best_patch_rate:.1%} success rate!")
                            print(f"🛑 STOPPING TRAINING - 90% threshold exceeded!")
                            training_stopped = True
                        
                        # Rejuvenation
                        if to_rejuvenate and latest_aug_performance[0][1] > 0.2:
                            print(f"🔧 Rejuvenating weakest patches...")
                            latent_batch = latent_batch.clone().detach()
                            latent_batch_best = latent_batch[[x[0] for x in latest_aug_performance[:5]]]
                            latent_batch[[x[0] for x in latest_aug_performance[-5:]]] = latent_batch_best + (torch.randn_like(latent_batch_best) * 0.1)
                            latent_batch.requires_grad = True
                            latent_opt = torch.optim.Adam([latent_batch], lr=0.1)
            
            if training_stopped:
                break
            
            if avg_epoch_success > best_success_rate:
                best_success_rate = avg_epoch_success
                best_latent = latent_batch.clone().detach()
            
            if epoch % 10 == 0:
                print(f"  [{self.exp_name}] Epoch {epoch:3d}/{num_epochs} | Loss: {avg_epoch_loss:.3f} | Success: {avg_epoch_success:.1%}")
        
        # Save results
        final_aug_rate = aug_success_rates[-1] if aug_success_rates else 0
        
        self.results.update({
            "losses": losses,
            "success_rates": success_rates,
            "aug_success_rates": aug_success_rates,
            "best_success_rate": best_success_rate,
            "final_aug_rate": final_aug_rate,
            "epochs_to_50_percent": epochs_to_50_percent,
            "total_epochs_run": len(losses),
        })
        
        self._save_results(best_latent, resizer)
        
        print(f"\n✅ Experiment {self.exp_name} Complete!")
        print(f"   Best Success Rate: {best_success_rate:.1%}")
        print(f"   Final Aug Rate: {final_aug_rate:.1%}")
        print(f"   Epochs to 50%: {epochs_to_50_percent if epochs_to_50_percent else 'Not reached'}")
        print(f"   Results saved to: {self.output_dir}")
        
        # Cleanup
        del latent_batch, latent_opt, resizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return self.results
    
    def _save_results(self, best_latent, resizer):
        """Save all experiment results."""
        print(f"\n💾 Saving results to: {self.output_dir}")
        
        # Save latent tensors
        if best_latent is not None:
            torch.save(best_latent, self.output_dir / "latent_batch_best.pt")
            
            # Decode and save individual patch images
            with torch.no_grad():
                decoded_patches = resizer(self.decode_latents(best_latent).float())
                for patch_idx in range(decoded_patches.shape[0]):
                    patch_img = decoded_patches[patch_idx].cpu().permute(1, 2, 0).numpy()
                    patch_img = (patch_img * 255).astype(np.uint8)
                    cv2.imwrite(
                        str(self.output_dir / f"patch_{patch_idx+1:02d}.png"),
                        cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)
                    )
        
        # Save metadata
        with open(self.output_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save as JSON for easy reading
        json_results = {k: v for k, v in self.results.items() if not isinstance(v, (np.ndarray, torch.Tensor))}
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Generate and save plots
        self._save_plots()
        
        # Log to Comet
        if self.comet_experiment:
            self.comet_experiment.log_asset(str(self.output_dir / "latent_batch_best.pt"))
            self.comet_experiment.log_asset(str(self.output_dir / "metadata.json"))
            for png in self.output_dir.glob("*.png"):
                self.comet_experiment.log_image(str(png), name=png.stem)
    
    def _save_plots(self):
        """Generate and save training plots."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        axes[0].plot(self.results["losses"], linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.exp_name} - Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Success rate plot
        axes[1].plot(self.results["success_rates"], linewidth=2, label='Training')
        if self.results["aug_success_rates"]:
            # Aug rates are collected at varying intervals, just plot them evenly spaced
            num_aug_rates = len(self.results["aug_success_rates"])
            num_epochs = len(self.results["success_rates"])
            if num_aug_rates > 0 and num_epochs > 0:
                # Create evenly spaced x values spanning the epochs
                epochs_with_aug = np.linspace(0, num_epochs - 1, num_aug_rates).astype(int).tolist()
                axes[1].plot(epochs_with_aug, self.results["aug_success_rates"], 'o-', linewidth=2, label='Best Aug')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title(f'{self.exp_name} - Attack Success Rate')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.comet_experiment:
            self.comet_experiment.log_figure(figure_name="training_curves")
    
    def cleanup(self):
        """Clean up resources."""
        if self.comet_experiment:
            self.comet_experiment.end()
        
        del self.vae
        del self.augmentor_model
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single experiment:
  python run_ablation_experiment.py --experiment latent_8x8
  
  # Run on specific GPU:
  python run_ablation_experiment.py --experiment latent_16x16 --gpu 1
  
  # Run all experiments sequentially:
  python run_ablation_experiment.py --all
  
  # Run with custom parameters:
  python run_ablation_experiment.py --experiment latent_4x4 --epochs 50 --batch-size 16
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help='Name of the experiment to run'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all experiments sequentially'
    )
    parser.add_argument(
        '--epochs', '-n',
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=f'Number of training epochs (default: {DEFAULT_NUM_EPOCHS})'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Number of patches in batch (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--gpu', '-g',
        type=int,
        default=None,
        help='GPU device index to use (for parallel runs)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./results/ablation',
        help='Base output directory for results'
    )
    parser.add_argument(
        '--no-comet',
        action='store_true',
        help='Disable Comet ML logging'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments and exit'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable experiments:")
        for name, config in EXPERIMENT_CONFIGS.items():
            print(f"  {name}: latent_size={config['latent_size']}x{config['latent_size']}, rejuvenate={config['rejuvenate']}")
        return
    
    if not args.experiment and not args.all:
        parser.error("Either --experiment or --all must be specified")
    
    # Determine which experiments to run
    if args.all:
        experiments_to_run = list(EXPERIMENT_CONFIGS.keys())
    else:
        experiments_to_run = [args.experiment]
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    
    print("=" * 60)
    print("🚀 ABLATION STUDY EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Experiments to run: {experiments_to_run}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU: {args.gpu if args.gpu is not None else 'auto'}")
    print("=" * 60 + "\n")
    
    all_results = {}
    
    for exp_name in experiments_to_run:
        config = EXPERIMENT_CONFIGS[exp_name]
        output_dir = Path(args.output_dir) / f"{exp_name}_{timestamp}"
        
        print(f"\n{'#' * 60}")
        print(f"# Running: {exp_name}")
        print(f"{'#' * 60}\n")
        
        try:
            experiment = AblationExperiment(exp_name, config, args, output_dir)
            experiment.setup()
            results = experiment.run()
            all_results[exp_name] = results
            experiment.cleanup()
            
        except Exception as e:
            print(f"❌ Error running {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Clear memory between experiments
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save combined results
    if len(all_results) > 1:
        combined_output = Path(args.output_dir) / f"combined_results_{timestamp}"
        combined_output.mkdir(parents=True, exist_ok=True)
        
        with open(combined_output / "all_results.pkl", 'wb') as f:
            pickle.dump(all_results, f)
        
        # Summary JSON
        summary = {
            exp_name: {
                "best_success_rate": res["best_success_rate"],
                "final_aug_rate": res["final_aug_rate"],
                "epochs_to_50_percent": res["epochs_to_50_percent"],
                "total_epochs_run": res["total_epochs_run"],
            }
            for exp_name, res in all_results.items()
        }
        with open(combined_output / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Combined results saved to: {combined_output}")
    
    print("\n" + "=" * 60)
    print("🎉 All experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
