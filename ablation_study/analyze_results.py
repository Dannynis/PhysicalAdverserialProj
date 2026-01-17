#!/usr/bin/env python3
"""
Analyze and visualize ablation study results.

Usage:
    # Analyze all results in the default directory:
    python analyze_results.py
    
    # Analyze specific run:
    python analyze_results.py --results-dir ./results/ablation/run_2026-01-17_12_30_00
    
    # Generate comparison plots:
    python analyze_results.py --compare latent_4x4 latent_8x8 latent_16x16
"""

import argparse
import os
import pickle
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob


def load_experiment_results(exp_dir):
    """Load results from an experiment directory."""
    exp_dir = Path(exp_dir)
    
    # Try to load metadata
    metadata_pkl = exp_dir / "metadata.pkl"
    metadata_json = exp_dir / "metadata.json"
    
    if metadata_pkl.exists():
        with open(metadata_pkl, 'rb') as f:
            return pickle.load(f)
    elif metadata_json.exists():
        with open(metadata_json, 'r') as f:
            return json.load(f)
    else:
        return None


def find_experiment_dirs(base_dir):
    """Find all experiment directories in the base directory."""
    base_dir = Path(base_dir)
    exp_dirs = {}
    
    for d in base_dir.iterdir():
        if d.is_dir() and (d / "metadata.pkl").exists():
            # Extract experiment name from directory
            exp_name = d.name.split('_2')[0]  # Split before timestamp
            exp_dirs[exp_name] = d
    
    return exp_dirs


def create_convergence_plot(results_dict, output_path):
    """Create success rate convergence plot for all experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'latent_4x4': '#e74c3c',
        'latent_8x8': '#f39c12',
        'latent_16x16': '#2ecc71',
        'latent_32x32': '#3498db',
        'latent_16x16_with_rejuv': '#9b59b6',
    }
    
    for exp_name, results in results_dict.items():
        if 'success_rates' in results:
            color = colors.get(exp_name, 'gray')
            ax.plot(results['success_rates'], 
                   label=exp_name, 
                   linewidth=2.5, 
                   color=color)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Attack Success Rate Convergence', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_loss_plot(results_dict, output_path):
    """Create loss convergence plot for all experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for exp_name, results in results_dict.items():
        if 'losses' in results:
            ax.plot(results['losses'], label=exp_name, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Convergence', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_comparison_bar_chart(results_dict, output_path):
    """Create bar chart comparing final metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    exp_names = list(results_dict.keys())
    
    # Best success rate
    ax = axes[0]
    best_rates = [results_dict[name].get('best_success_rate', 0) for name in exp_names]
    bars = ax.bar(exp_names, best_rates, color='#3498db')
    ax.set_ylabel('Best Success Rate')
    ax.set_title('Best Success Rate')
    ax.set_ylim(0, 1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar, rate in zip(bars, best_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Final aug rate
    ax = axes[1]
    aug_rates = [results_dict[name].get('final_aug_rate', 0) for name in exp_names]
    bars = ax.bar(exp_names, aug_rates, color='#2ecc71')
    ax.set_ylabel('Final Aug Rate')
    ax.set_title('Final Augmented Success Rate')
    ax.set_ylim(0, 1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar, rate in zip(bars, aug_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Epochs to 50%
    ax = axes[2]
    epochs = [results_dict[name].get('epochs_to_50_percent') or 
              results_dict[name].get('total_epochs_run', 30) 
              for name in exp_names]
    bars = ax.bar(exp_names, epochs, color='#e74c3c')
    ax.set_ylabel('Epochs')
    ax.set_title('Epochs to 50% Success')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for bar, ep, name in zip(bars, epochs, exp_names):
        label = str(ep) if results_dict[name].get('epochs_to_50_percent') else 'N/R'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                label, ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_latent_size_comparison(results_dict, output_path):
    """Compare different latent sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to non-rejuvenation experiments
    latent_results = {k: v for k, v in results_dict.items() 
                      if 'rejuv' not in k}
    
    colors = {
        'latent_4x4': '#e74c3c',
        'latent_8x8': '#f39c12',
        'latent_16x16': '#2ecc71',
        'latent_32x32': '#3498db',
    }
    
    for exp_name, results in sorted(latent_results.items()):
        if 'success_rates' in results:
            color = colors.get(exp_name, 'gray')
            ax.plot(results['success_rates'], 
                   label=exp_name.replace('_no_rejuv', ''), 
                   linewidth=2.5, 
                   color=color)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Effect of Latent Size on Attack Success Rate', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_rejuvenation_comparison(results_dict, output_path):
    """Compare rejuvenation vs no rejuvenation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to 16x16 experiments
    rejuv_results = {k: v for k, v in results_dict.items() if '16x16' in k}
    
    for exp_name, results in rejuv_results.items():
        if 'success_rates' in results:
            label = "With Rejuvenation" if 'rejuv' in exp_name else "Without Rejuvenation"
            color = '#2ecc71' if 'rejuv' in exp_name else '#e74c3c'
            ax.plot(results['success_rates'], 
                   label=label, 
                   linewidth=2.5, 
                   color=color)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Effect of Patch Rejuvenation on Convergence (16x16 Latent)', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(results_dict, output_path):
    """Create and save summary table."""
    rows = []
    
    for exp_name, results in results_dict.items():
        rows.append({
            "Experiment": exp_name,
            "Latent Size": f"{results.get('config', {}).get('latent_size', 'N/A')}",
            "Rejuvenation": "Yes" if results.get('config', {}).get('rejuvenate', False) else "No",
            "Best Success Rate": f"{results.get('best_success_rate', 0):.1%}",
            "Final Aug Rate": f"{results.get('final_aug_rate', 0):.1%}",
            "Epochs to 50%": results.get('epochs_to_50_percent') or "N/R",
            "Total Epochs": results.get('total_epochs_run', 'N/A'),
        })
    
    df = pd.DataFrame(rows)
    
    # Print table
    print("\n" + "=" * 80)
    print("📊 ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument(
        '--results-dir', '-r',
        type=str,
        default='./results/ablation',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for analysis (default: same as results-dir)'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        default=None,
        help='Specific experiments to compare'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔍 ABLATION STUDY RESULTS ANALYSIS")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")
    
    # Find and load all experiment results
    exp_dirs = find_experiment_dirs(results_dir)
    
    if not exp_dirs:
        # Try loading from run subdirectories
        for run_dir in results_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith('run_'):
                exp_dirs.update(find_experiment_dirs(run_dir))
    
    if not exp_dirs:
        print("❌ No experiment results found!")
        return
    
    print(f"Found {len(exp_dirs)} experiments:")
    for name, path in exp_dirs.items():
        print(f"  - {name}: {path}")
    
    # Load results
    results_dict = {}
    for exp_name, exp_dir in exp_dirs.items():
        results = load_experiment_results(exp_dir)
        if results:
            results_dict[exp_name] = results
    
    if args.compare:
        # Filter to specified experiments
        results_dict = {k: v for k, v in results_dict.items() if k in args.compare}
    
    print(f"\nLoaded {len(results_dict)} experiment results\n")
    
    # Generate analysis outputs
    if len(results_dict) > 0:
        create_summary_table(results_dict, output_dir / "summary.csv")
        create_convergence_plot(results_dict, output_dir / "convergence.png")
        create_loss_plot(results_dict, output_dir / "loss_convergence.png")
        create_comparison_bar_chart(results_dict, output_dir / "comparison_bars.png")
        
        # Specialized comparisons
        latent_results = {k: v for k, v in results_dict.items() if 'rejuv' not in k}
        if len(latent_results) > 1:
            create_latent_size_comparison(results_dict, output_dir / "latent_size_comparison.png")
        
        rejuv_results = {k: v for k, v in results_dict.items() if '16x16' in k}
        if len(rejuv_results) > 1:
            create_rejuvenation_comparison(results_dict, output_dir / "rejuvenation_comparison.png")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
