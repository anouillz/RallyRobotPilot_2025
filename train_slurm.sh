#!/bin/bash
#SBATCH --job-name=rally_train          # Nom du job
#SBATCH --output=logs/%x_%j.out        # Log standard
#SBATCH --error=logs/%x_%j.err         # Log erreurs
#SBATCH --time=00:30:00                # Temps max (hh:mm:ss)
#SBATCH --tasks=1
#SBATCH --gpus-per-node=1 
#SBATCH --nodelist=calypso0
#SBATCH --cpus-per-task=10

source venv/bin/activate

python3 train.py
