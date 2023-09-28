import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa

from pathlib import Path
from typing import Union
from itertools import permutations

sys.path.append(os.path.expanduser('~/Documents/Medley/'))

from Medley.interpretation import all_expids, predselpath

if __name__ == '__main__':
    pairs = permutations(all_expids,2)
    duplicates = []
    for id1, id2 in pairs:
        cmd = f'diff {predselpath}/{id1}_experiment.json {predselpath}/{id2}_experiment.json'
        ret = os.system(cmd) 
        # TODO: check return codes for duplicates
        if ret == 0: # 256 for differences
            duplicates.append((id1,id2))

