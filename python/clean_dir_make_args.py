import pandas as pd

print("e11 in cdma.")
exec(open('python/sim_settings.py').read())

if sim=='synthetic':
    ds = range(len(settings))
else:
    ds = datasets_to_use

dd = []
for s_i in ds:
    for seed in range(iters):
        dd.append(str(s_i)+' '+str(seed))

pd.DataFrame(dd).to_csv('sim_args.txt', index = False, header=False)

import os
import glob

files = glob.glob('sim_out/'+simout_dir+"/*")
for f in files:
    os.remove(f)
