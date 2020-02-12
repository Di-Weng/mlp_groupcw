import os
import sys
import torch
"""
Usage:
This .py helps fix the old checkpoint name issue.
1) First create a dummy directory called "utils/"
2) Then create a dummy class called `Timer()` in a dummy file called `timmer.py`
3) Run this script to fix the old checkpoint files that store a 'utils.timer' that is no longer used
"""
input_ckpt = os.path.join('/home/hongyuan/Desktop/mlp_groupcw/framework/MPC', 'mockingjay-500000.ckpt')

all_states = torch.load(input_ckpt, map_location='cpu')
config = all_states['Settings']['Config']

for cluster in all_states['Settings']['Config']:
    if 'timer' in all_states['Settings']['Config']:
        del all_states['Settings']['Config']['timer']
        break

torch.save(all_states, input_ckpt)
print('Done fixing ckpt: ', input_ckpt)
