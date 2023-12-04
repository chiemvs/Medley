#!/usr/bin/bash
#SBATCH --job-name="spi1"
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --mem=50gb
#SBATCH --partition=ivm
#SBATCH --nodelist=node240
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

VENVNAME=TESTVENV
cp ./calc_spi_from_ecad.py $TMPDIR/calc_spi_from_ecad.py

$HOME/Documents/EnvManager/environment_setup.sh -s $VENVNAME

source $TMPDIR/$VENVNAME/bin/activate
pip3 install xgboost

which python3
python3 --version

python3 $TMPDIR/calc_spi_from_ecad.py
