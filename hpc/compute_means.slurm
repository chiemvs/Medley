#!/usr/bin/bash
#SBATCH --job-name="u500"
#SBATCH --cpus-per-task=20
#SBATCH --time=120:00:00
#SBATCH --mem=50gb
#SBATCH --partition=ivm
#SBATCH --nodelist=node241
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

module load 2022
module load Python

VENVNAME=TESTVENV

$HOME/Documents/EnvManager/environment_setup.sh -s $VENVNAME

source $TMPDIR/$VENVNAME/bin/activate

which python3
python3 --version

python3 ./compute_means.py
