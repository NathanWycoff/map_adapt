import pandas as pd
import sys

exec(open('python/sim_settings.py').read())

mode = sys.argv[1]
if mode=='small':
    if sim=='synthetic':
        ds = range(len(settings))
    else:
        ds = datasets_to_use

    dd = []
    for s_i in ds:
        for seed in range(iters):
            dd.append(str(s_i)+' '+str(seed))

    pd.DataFrame(dd).to_csv('sim_args.txt', index = False, header=False)
elif mode=='hcr':
    exec(open('python/hcr_settings.py').read())

    dd = []
    for s_i in range(n_tau):
        for seed in range(hcr_iters):
            dd.append(str(s_i)+' '+str(seed))

    pd.DataFrame(dd).to_csv('hcr_zinb_args.txt', index = False, header=False)

    dd = []
    for s_i in range(hcr_iters):
        dd.append(str(seed))
    pd.DataFrame(dd).to_csv('hcr_comp_args.txt', index = False, header=False)
else:
    raise Exception("Unknown mode arg to clean_dir!")

import os
import glob

files = glob.glob('sim_out/'+simout_dir+"/*")
for f in files:
    os.remove(f)

