#!/usr/bin/bash
#SBATCH --job-name="xgb_r2"
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=15gb
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

VENVNAME=TESTVENV

$HOME/Documents/EnvManager/environment_setup.sh -s $VENVNAME

# Sourcing automatically loads modules
source $TMPDIR/$VENVNAME/bin/activate 
pip3 install xgboost

python3 ./predictor_selection.py 20

