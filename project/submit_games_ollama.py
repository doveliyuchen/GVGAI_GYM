#!/usr/bin/env python
"""
SLURM job submission script for running GVGAI LLM experiments with Ollama.
Submits a job that runs main_6.20_fixed.py with 6 games, qwen3 model, and multiple runs.
Ollama binary and models are stored in /vast directory.
"""

import json
import os
import sys
import yaml
import submitit
from pathlib import Path
from datetime import datetime
import subprocess

# Games to run - 6 games as requested
GAMES = ["zelda", "aliens", "boulderdash", "sokoban", "realsokoban", "escape"]
NUM_RUNS = 10
MODELS = ["qwen3"]  # Ollama qwen3 model
MODES = ["contextual"]  # Both modes

def calculate_optimal_workers(mem_gb):
    """
    Calculate optimal number of parallel workers based on available memory.
    For Ollama on GPU, we use fewer workers to avoid memory issues.
    """
    memory_per_worker = 8  # GB - more conservative for local inference
    optimal_workers = max(1, int(mem_gb / memory_per_worker))
    # Cap at 6 for Ollama stability on GPU
    return min(optimal_workers, 6)

def load_config(config_file="slurm_conf_ollama.yaml"):
    """Load SLURM configuration from YAML file."""
    config_path = Path(__file__).parent / config_file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_ollama_environment():
    """
    Setup Ollama environment: download binary to /vast and set environment variables.
    The actual Ollama service will be managed by OllamaClient in the LLM framework.
    """
    print(f"{'='*80}")
    print("Setting up Ollama environment...")
    print(f"{'='*80}\n")
    
    # Use /vast directory for Ollama binary and models
    user = os.environ.get("USER", "yl6394")
    vast_dir = Path(f"/vast/{user}")
    ollama_bin_dir = vast_dir / "bin"
    ollama_models_dir = vast_dir / "ollama_models"
    
    # Create directories if they don't exist
    ollama_bin_dir.mkdir(parents=True, exist_ok=True)
    ollama_models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Ollama binary directory: {ollama_bin_dir}")
    print(f"Ollama models directory: {ollama_models_dir}")
    
    # Check if ollama binary exists in /vast
    ollama_path = ollama_bin_dir / "ollama"
    
    if not ollama_path.exists():
        print("Ollama not found in /vast, downloading...")
        
        download_cmd = [
            "curl", "-L", 
            "https://ollama.com/download/ollama-linux-amd64",
            "-o", str(ollama_path)
        ]
        
        try:
            subprocess.run(download_cmd, check=True)
            subprocess.run(["chmod", "+x", str(ollama_path)], check=True)
            print(f"✓ Ollama downloaded to {ollama_path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download Ollama: {e}")
            return False
    else:
        print(f"✓ Found Ollama at: {ollama_path}")
    
    # Add ollama to PATH and set environment variables
    # These will be used by OllamaClient
    current_path = os.environ.get("PATH", "")
    if str(ollama_bin_dir) not in current_path:
        os.environ["PATH"] = f"{ollama_bin_dir}:{current_path}"
    
    os.environ["OLLAMA_MODELS"] = str(ollama_models_dir)  # Store models in /vast
    
    print(f"\nEnvironment variables set:")
    print(f"  PATH includes: {ollama_bin_dir}")
    print(f"  OLLAMA_MODELS: {os.environ['OLLAMA_MODELS']}")
    print(f"\n✓ Ollama environment ready. OllamaClient will manage the service.\n")
    
    return True

def run_job(slurm_config):
    """
    Run main_6.20_fixed.py with Ollama qwen3 model.
    This function runs inside the SLURM job.
    """
    # Calculate optimal workers based on allocated memory
    mem_gb = slurm_config['mem'] / 1000  # Convert MB to GB
    max_workers = calculate_optimal_workers(mem_gb)
    
    print(f"{'='*80}")
    print(f"Job started at: {datetime.now().isoformat()}")
    print(f"Allocated memory: {mem_gb}GB")
    print(f"GPU: {slurm_config.get('n_gpus', 0)}x {slurm_config.get('gpu_type', 'N/A')}")
    print(f"Calculated optimal workers: {max_workers}")
    print(f"Running games: {GAMES}")
    print(f"Number of runs per game: {NUM_RUNS}")
    print(f"Models: {MODELS}")
    print(f"Modes: {MODES}")
    print(f"{'='*80}\n")
    
    # Setup Ollama environment (binary download and environment variables)
    if not setup_ollama_environment():
        print("Failed to setup Ollama environment. Exiting.")
        return
    
    # Get the script directory
    script_dir = Path(__file__).parent.resolve()
    main_script = script_dir / "main_6.20_fixed.py"
    
    if not main_script.exists():
        print(f"ERROR: main_6.20_fixed.py not found at {main_script}")
        return
    
    # Run for qwen3 model
    model = "qwen3"
    print(f"\n{'='*80}")
    print(f"Starting experiments with model: {model}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*80}\n")
    
    # Build the command with all parameters
    # The OllamaClient in the LLM framework will handle starting/stopping the service
    cmd = f"python {main_script}"
    cmd += f" --games {' '.join(GAMES)}"
    cmd += f" --models {' '.join(MODELS)}"
    cmd += f" --modes {' '.join(MODES)}"
    cmd += f" --num_runs {NUM_RUNS}"
    cmd += f" --max_workers {max_workers}"
    cmd += f" --base_output_dir llm_agent_runs_output"
    
    print(f"Executing command:")
    print(cmd)
    print()
    
    # Execute the command
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"\n⚠️  WARNING: Command exited with code {exit_code} for model {model}")
    else:
        print(f"\n✓ Successfully completed experiments for model {model}")
    
    print(f"{'='*80}\n")
    print(f"All jobs completed at: {datetime.now().isoformat()}")

def main():
    """
    Main function to submit job to SLURM using submitit.
    """
    # Load configuration
    try:
        config = load_config()
        slurm_config = config['slurm']
    except FileNotFoundError:
        print("ERROR: slurm_conf_ollama.yaml not found in the current directory")
        print(f"Current directory: {Path(__file__).parent}")
        return 1
    except KeyError:
        print("ERROR: Invalid configuration file format. Missing 'slurm' key.")
        return 1
    
    # Create logs directory
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    executor = submitit.AutoExecutor(folder=str(logs_dir))
    
    job_record_file = Path(__file__).parent / "game_job_records_ollama.json"
    
    # Calculate time in minutes
    time_minutes = slurm_config['time_hours'] * 60
    
    # Configure SLURM parameters
    executor_params = {
        "slurm_job_name": slurm_config['exp_id'],
        "nodes": slurm_config['n_nodes'],
        "tasks_per_node": 1,
        "cpus_per_task": slurm_config['n_cpus'],
        "slurm_mem": slurm_config['mem'],
        "slurm_account": slurm_config['account'],
        "slurm_time": time_minutes,
    }
    
    # Add GPU parameters for Ollama
    if slurm_config.get('n_gpus', 0) > 0 and slurm_config.get('gpu_type'):
        executor_params["slurm_gres"] = f"gpu:{slurm_config['gpu_type']}:{slurm_config['n_gpus']}"
        print(f"GPU requested: {slurm_config['n_gpus']}x {slurm_config['gpu_type']}")
    
    executor.update_parameters(**executor_params)
    
    # Print configuration summary
    print("="*80)
    print("SLURM JOB SUBMISSION - OLLAMA QWEN3")
    print("="*80)
    print("\nSLURM Configuration:")
    print(f"  Job Name: {slurm_config['exp_id']}")
    print(f"  Nodes: {slurm_config['n_nodes']}")
    print(f"  CPUs: {slurm_config['n_cpus']}")
    print(f"  GPUs: {slurm_config['n_gpus']} x {slurm_config.get('gpu_type', 'N/A')}")
    print(f"  Memory: {slurm_config['mem']} MB ({slurm_config['mem']/1000:.1f} GB)")
    print(f"  Time: {slurm_config['time_hours']} hours ({slurm_config['time_hours']/24:.1f} days)")
    print(f"  Account: {slurm_config['account']}")
    if slurm_config.get('conda_env'):
        print(f"  Conda Env: {slurm_config['conda_env']}")
    print()
    print("Experiment Configuration:")
    print(f"  Games ({len(GAMES)}): {', '.join(GAMES)}")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Modes: {', '.join(MODES)}")
    print(f"  Runs per game: {NUM_RUNS}")
    print(f"  Max parallel workers: {calculate_optimal_workers(slurm_config['mem']/1000)}")
    total_experiments = len(GAMES) * len(MODES) * NUM_RUNS
    print(f"  Total experiments: {len(GAMES)} games × {len(MODES)} modes × {NUM_RUNS} runs = {total_experiments}")
    user = os.environ.get("USER", "yl6394")
    print(f"\nStorage Locations:")
    print(f"  Ollama binary: /vast/{user}/bin/ollama")
    print(f"  Ollama models: /vast/{user}/ollama_models/")
    print("="*80)
    
    # Submit the job
    print("\nSubmitting job to SLURM...")
    try:
        job = executor.submit(run_job, slurm_config)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to submit job: {e}")
        return 1
    
    print(f"✓ Job submitted successfully!")
    print(f"  Job ID: {job.job_id}")
    
    # Load existing job records if available
    if job_record_file.exists():
        with open(job_record_file, "r") as f:
            job_records = json.load(f)
    else:
        job_records = {}
    
    # Save job information
    job_records[str(job.job_id)] = {
        "games": GAMES,
        "models": MODELS,
        "modes": MODES,
        "num_runs": NUM_RUNS,
        "submitted_at": datetime.now().isoformat(),
        "config": {
            "nodes": slurm_config['n_nodes'],
            "cpus": slurm_config['n_cpus'],
            "gpus": slurm_config.get('n_gpus', 0),
            "gpu_type": slurm_config.get('gpu_type', 'none'),
            "mem_gb": slurm_config['mem'] / 1000,
            "time_hours": slurm_config['time_hours'],
            "conda_env": slurm_config.get('conda_env', 'none'),
        }
    }
    
    # Save job records to a JSON file
    with open(job_record_file, "w") as f:
        json.dump(job_records, f, indent=4)
    
    print(f"  Job record saved to: {job_record_file}")
    print()
    print("Monitor job with:")
    print(f"  squeue -u $USER                          # Check job status")
    print(f"  tail -f {logs_dir}/{job.job_id}_0_log.out   # View output")
    print(f"  tail -f {logs_dir}/{job.job_id}_0_log.err   # View errors")
    print()
    print("Cancel job with:")
    print(f"  scancel {job.job_id}")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
