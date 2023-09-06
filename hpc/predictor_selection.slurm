#!/usr/bin/bash
#SBATCH --job-name="lreg_b"
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem=15gb
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

VENVNAME=TESTVENV

$HOME/Documents/EnvManager/environment_setup.sh -s $VENVNAME

# Sourcing automatically loads modules
source $TMPDIR/$VENVNAME/bin/activate 

python3 ./predictor_selection.py 10

