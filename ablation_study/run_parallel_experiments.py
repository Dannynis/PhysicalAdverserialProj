#!/usr/bin/env python3
"""
Run multiple ablation experiments in parallel across multiple GPUs.

Usage:
    # Run all experiments in parallel (auto-distribute across GPUs):
    python run_parallel_experiments.py
    
    # Run specific experiments in parallel:
    python run_parallel_experiments.py --experiments latent_4x4 latent_8x8
    
    # Specify GPUs to use:
    python run_parallel_experiments.py --gpus 0 1
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime

SCRIPT_DIR = Path(__file__).parent.absolute()

EXPERIMENTS = [
    "latent_4x4",
    "latent_8x8", 
    "latent_16x16",
    "latent_32x32",
    "latent_16x16_with_rejuv",
]


def run_experiment(exp_name, gpu_id, epochs, batch_size, output_dir, no_comet):
    """Run a single experiment as a subprocess."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_ablation_experiment.py"),
        "--experiment", exp_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--output-dir", output_dir,
    ]
    
    if gpu_id is not None:
        cmd.extend(["--gpu", str(gpu_id)])
    
    if no_comet:
        cmd.append("--no-comet")
    
    print(f"🚀 Starting {exp_name} on GPU {gpu_id}...")
    print(f"   Command: {' '.join(cmd)}")
    
    # Create log file for this experiment
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    with open(log_file, 'w') as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=SCRIPT_DIR.parent,  # Run from project root
        )
    
    return {
        "experiment": exp_name,
        "gpu": gpu_id,
        "return_code": result.returncode,
        "log_file": str(log_file),
    }


def main():
    parser = argparse.ArgumentParser(description='Run ablation experiments in parallel')
    parser.add_argument(
        '--experiments', '-e',
        nargs='+',
        choices=EXPERIMENTS,
        default=EXPERIMENTS,
        help='Experiments to run (default: all)'
    )
    parser.add_argument(
        '--gpus', '-g',
        nargs='+',
        type=int,
        default=None,
        help='GPU IDs to use (will distribute experiments across them)'
    )
    parser.add_argument(
        '--epochs', '-n',
        type=int,
        default=30,
        help='Number of epochs per experiment'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=10,
        help='Batch size (number of patches)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./results/ablation',
        help='Base output directory'
    )
    parser.add_argument(
        '--no-comet',
        action='store_true',
        help='Disable Comet ML logging'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run experiments sequentially instead of in parallel'
    )
    
    args = parser.parse_args()
    
    # Detect available GPUs
    import torch
    num_gpus = torch.cuda.device_count()
    
    if args.gpus is None:
        if num_gpus > 0:
            args.gpus = list(range(num_gpus))
        else:
            args.gpus = [None]  # CPU only
    
    print("=" * 60)
    print("🔬 PARALLEL ABLATION EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Experiments: {args.experiments}")
    print(f"Available GPUs: {num_gpus}")
    print(f"Using GPUs: {args.gpus}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print("=" * 60 + "\n")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_dir = str(Path(args.output_dir) / f"run_{timestamp}")
    
    results = []
    
    if args.sequential:
        # Run experiments one at a time
        for i, exp_name in enumerate(args.experiments):
            gpu_id = args.gpus[i % len(args.gpus)]
            result = run_experiment(
                exp_name, gpu_id, args.epochs, args.batch_size, 
                output_dir, args.no_comet
            )
            results.append(result)
            print(f"✅ {exp_name} completed (return code: {result['return_code']})")
    else:
        # Run experiments in parallel
        max_workers = min(len(args.experiments), len(args.gpus))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for i, exp_name in enumerate(args.experiments):
                gpu_id = args.gpus[i % len(args.gpus)]
                future = executor.submit(
                    run_experiment,
                    exp_name, gpu_id, args.epochs, args.batch_size,
                    output_dir, args.no_comet
                )
                futures[future] = exp_name
            
            for future in as_completed(futures):
                exp_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "✅" if result["return_code"] == 0 else "❌"
                    print(f"{status} {exp_name} completed (return code: {result['return_code']})")
                    print(f"   Log: {result['log_file']}")
                except Exception as e:
                    print(f"❌ {exp_name} failed with exception: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r["return_code"] == 0]
    failed = [r for r in results if r["return_code"] != 0]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"   - {r['experiment']}")
    
    if failed:
        print(f"❌ Failed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"   - {r['experiment']} (check {r['log_file']})")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
