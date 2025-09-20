import numpy as np
from datetime import datetime
import pickle
import re
import sys
import glob
import pandas as pd

if len(sys.argv)!=3:
    print("wrong number of arguments")

grpNum=int(sys.argv[1])
rowNum=int(sys.argv[2])

#this script loads wavefunction and checks norm
in_wvfunc_dir=f"./outData/group{grpNum}/row{rowNum}/wavefunction/"
dataFilesAll=[]
time_step_EndAll=[]
for oneDataFile in glob.glob(in_wvfunc_dir+"/psi*.pkl"):
    dataFilesAll.append(oneDataFile)
    match_time_step=re.search(r"psi(\d+)",oneDataFile)
    if match_time_step:
        time_step_EndAll.append(int(match_time_step.group(1)))
endInds=np.argsort(time_step_EndAll)

sorted_time_step_all=[time_step_EndAll [i] for i in endInds]
sortedDataFiles=[dataFilesAll[i] for i in endInds]
nm_all=[]


t_norm_start=datetime.now()
for j in range(0,len(sorted_time_step_all)):
    step_ind=sorted_time_step_all[j]
    file=sortedDataFiles[j]
    with open(file,"rb") as fptr:
        in_wvfunc_data=pickle.load(fptr)
    in_wvfunc_data=np.array(in_wvfunc_data)
    nm_all.append(np.linalg.norm(in_wvfunc_data,ord=2))

t_norm_end=datetime.now()
out_norm_pd=pd.DataFrame({'time_step': sorted_time_step_all, 'norm': nm_all})
out_norm_pd.to_csv(in_wvfunc_dir+'/out_norm.csv', index=False)
print(f"total norm time: ", t_norm_end-t_norm_start)

