#!/usr/bin/bash
#SBATCH --job-name="west10"
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=30gb
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

VENVNAME=TESTVENV
cp ./hyperparams.py $TMPDIR/hyperparams.py

$HOME/Documents/EnvManager/environment_setup.sh -s $VENVNAME

# Sourcing automatically loads modules
source $TMPDIR/$VENVNAME/bin/activate 
pip3 install xgboost
pip3 install shap

python3 $TMPDIR/hyperparams.py 15

