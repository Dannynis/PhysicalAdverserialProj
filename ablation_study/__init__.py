"""
Ablation Study Module

Scripts for running and analyzing ablation experiments comparing
different latent sizes and training strategies for adversarial patches.
"""

from pathlib import Path

ABLATION_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = ABLATION_DIR.parent

EXPERIMENT_CONFIGS = {
    "latent_4x4": {"latent_size": 4, "rejuvenate": False},
    "latent_8x8": {"latent_size": 8, "rejuvenate": False},
    "latent_16x16": {"latent_size": 16, "rejuvenate": False},
    "latent_32x32": {"latent_size": 32, "rejuvenate": False},
    "latent_16x16_with_rejuv": {"latent_size": 16, "rejuvenate": True},
}

__all__ = ['EXPERIMENT_CONFIGS', 'ABLATION_DIR', 'PROJECT_ROOT']
