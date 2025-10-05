import pandas as pd
import numpy as np
import glob
from datetime import datetime
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
from sympy import *
from sympy.simplify.fu import TR11,TR5
import re

grpNum=0
rowNum=0
inCsvFileName=f"./inParams/inParams{grpNum}.csv"

df=pd.read_csv(inCsvFileName)
row=df.iloc[rowNum,:]
j1H = int(row["j1H"])
j2H = int(row["j2H"])
g0=row["g0"]
omegam=row["omegam"]
omegap=row["omegap"]
omegac=row["omegac"]

er=row["er"]
thetaCoef=row["thetaCoef"]


theta=thetaCoef*np.pi

e2r=er**2
Deltam=omegam - omegap
lmd = (e2r - 1 / e2r) / (e2r + 1 / e2r) * Deltam
half=1/2
quarter=1/4
D=lmd**2*np.sin(theta)**2+omegap**2
mu=lmd*np.cos(theta)+Deltam
print("=" * 60)
print(f"Parameters from {inCsvFileName} (Row {rowNum})")
print("=" * 60)

# Print all parameters from the CSV
for col_name, value in row.items():
    print(f"{col_name:15s}: {value}")

print("=" * 60)

# Also print derived parameters that you calculate
print("\nDerived Parameters:")
print("-" * 60)

# Calculate derived values

theta = thetaCoef * np.pi
e2r = er**2
Deltam = omegam - omegap
lmd = (e2r - 1 / e2r) / (e2r + 1 / e2r) * Deltam
D = lmd**2 * np.sin(theta)**2 + omegap**2
mu = lmd * np.cos(theta) + Deltam

print(f"{'theta':15s}: {theta:.6f} (= {thetaCoef}*π)")
print(f"{'e2r':15s}: {e2r:.6f} (= er²)")
print(f"{'Deltam':15s}: {Deltam:.6f} (= omegam - omegap)")
print(f"{'lambda (lmd)':15s}: {lmd:.6f}")
print(f"{'D':15s}: {D:.6f}")
print(f"{'mu':15s}: {mu:.6f}")

print("=" * 60)

x1,x2,t=symbols("x1,x2,t",cls=Symbol,real=True)

rho=omegac*x1**2-half

F0=I*omegam*mu/(4*lmd*sin(theta))*x2**2+I*g0**2/D**2*rho**2* \
   ((2*omegap-D/(2*omegap)-mu/2)*lmd*sin(theta)+mu*D/(4*lmd*sin(theta)))



F1=I*g0**2/D*(omegap-mu/2)*rho**2


F2=I*g0**2/D**2* \
   (2*lmd**2*D*sin(theta)**2+4*mu*lmd**2*omegap*sin(theta)**2+mu*omegap**3-3*mu*lmd**2*omegap*sin(theta)**2)/(4*lmd*omegap*sin(theta))*rho**2



F3=I*g0/D*(half*mu-omegap)*sqrt(2*omegam)*rho*x2


F4=I*g0/D*sqrt(2*omegam)*(mu*omegap+2*lmd**2*sin(theta)**2)/(2*lmd*sin(theta))*rho*x2



F5=-I*mu/(4*lmd*sin(theta)*D)*(D*omegam*x2**2+g0**2*rho**2)


F6=I*mu*g0/D*sqrt(omegam/2)*rho*x2


F7=-I*mu*g0/(lmd*sin(theta)*D)*sqrt(omegam/2)*omegap*rho*x2


F8=I*mu*g0**2/(4*D**2*lmd*sin(theta))*(lmd**2*sin(theta)**2-omegap**2)*rho**2

F9=I*mu*g0**2/(2*D**2)*omegap*rho**2

F10=-I*g0/D*sqrt(2*omegam)*lmd*sin(theta)*rho*x2

F11=I*2*g0**2/D**2*lmd**2*sin(theta)**2*rho**2

F12=-I*2*g0**2/D**2*lmd*omegap*sin(theta)*rho**2

G=F0+F1*t-1/(2*omegap)*F1*sin(2*omegap*t) \
  +F2*cos(2*omegap*t)+F3*sin(omegap*t)+F4*cos(omegap*t) \
  +F5*exp(2*lmd*sin(theta)*t)+F6*sin(omegap*t)*exp(2*lmd*sin(theta)*t) \
  +F7*cos(omegap*t)*exp(2*lmd*sin(theta)*t)+F8*cos(2*omegap*t)*exp(2*lmd*sin(theta)*t)+F9*sin(2*omegap*t)*exp(2*lmd*sin(theta)*t) \
  +F10*exp(lmd*sin(theta)*t)+F11*sin(omegap*t)*exp(lmd*sin(theta)*t)+F12*cos(omegap*t)*exp(lmd*sin(theta)*t)


beta=(-I*half*omegac*rho+I*quarter*omegac+I*half*Deltam+half*lmd*sin(theta))*t





def f1(x1Val):
    return exp(-half*omegac*x1Val**2)*hermite(j1H,x1Val*sqrt(omegac))

def f2(x2Val):
    return  exp(-half*omegam*x2Val**2)*hermite(j2H,x2Val*sqrt(omegam))


s2=-g0/D*omegap*sqrt(2/omegam)*rho+x2*exp(lmd*sin(theta)*t) \
   -g0/D*sqrt(2/omegam)*sin(theta)*rho*sin(omegap*t)*exp(lmd*sin(theta)*t)*lmd \
   +g0/D*sqrt(2/omegam)*omegap*rho*cos(omegap*t)*exp(lmd*sin(theta)*t)


def phi(x1Val,x2Val,tVal):
    # s2Val=s2.subs([(x1,x1Val),(x2,x2Val),(t,tVal)]).evalf()
    f1_part=f1(x1Val)
    f2_part=f2(x2Val)
    return f1_part*f2_part

# x1Val=0.1
# x2Val=0.2
# tVal=10
# s2Val=s2.subs([(x1,x1Val),(x2,x2Val),(t,tVal)]).evalf()

# Create symbolic psi expression
f1_symbolic = exp(-half*omegac*x1**2)*hermite(j1H, x1*sqrt(omegac))
f2_symbolic = exp(-half*omegam*s2**2)*hermite(j2H, s2*sqrt(omegam))
psi = f1_symbolic * f2_symbolic * exp(G + beta)

# Lambdify psi
psi_func = lambdify([x1, x2, t], psi, modules='numpy')

L1 = 1;
L2 = 5;
N1=270*2
N2 = 300*2
# print(2/(np.abs(lmd*np.sin(theta))*N2))
dx1 = 2.0 * L1 /N1
dx2 = 2.0 * L2 / N2
x1ValsAll=[-L1 + dx1 * n1 for n1 in range(0,N1)]
x2ValsAll=[-L2 + dx2 * n2 for n2 in range(0,N2)]
dx1 = 2.0 * L1 / N1
dx2 = 2.0 * L2 /N2
tTot = 1.0
Q = 1e6
dt = tTot /Q
print(f"dx1={dx1}, dx2={dx2}, dt={dt}")

def compute_row_batch(row_indices, t, x1ValsAll, x2ValsAll, N2):
    """Compute a batch of rows"""
    rows = []
    for n1 in row_indices:
        row = np.zeros(N2, dtype=complex)
        x1Tmp = x1ValsAll[n1]
        for n2 in range(N2):
            x2Tmp = x2ValsAll[n2]
            row[n2] = psi_func(x1Tmp, x2Tmp, t)  # Use global psi_func
        rows.append((n1, row))
    return rows

def generate_discrete_solution_normalized_parallel(t, n_workers=None):
    """
    :param t: time
    :param n_workers: number of worker processes (None for cpu_count)
    :return: solution, row major order
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Create batches of row indices
    batch_size = max(1, N1 // n_workers)
    row_batches = [range(i, min(i + batch_size, N1))
                   for i in range(0, N1, batch_size)]

    # Create partial function (without psi_func since it's global)
    compute_batch_partial = partial(
        compute_row_batch,
        t=t,
        x1ValsAll=x1ValsAll,
        x2ValsAll=x2ValsAll,
        N2=N2
    )

    # Parallelize computation
    with Pool(n_workers) as pool:
        batch_results = pool.map(compute_batch_partial, row_batches)

    # Assemble solution matrix
    solution = np.zeros((N1, N2), dtype=complex)
    for batch in batch_results:
        for n1, row in batch:
            solution[n1, :] = row

    solution /= np.linalg.norm(solution, ord="fro")
    return solution.flatten()


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
diff_all=[]

t_diff_start=datetime.now()
for j in range(0,len(sorted_time_step_all)):
    step_ind=sorted_time_step_all[j]
    file=sortedDataFiles[j]
    with open(file,"rb") as fptr:
        in_wvfunc_data=pickle.load(fptr)
    in_wvfunc_data=np.array(in_wvfunc_data)
    analytic=generate_discrete_solution_normalized_parallel(step_ind*dt,24)
    diff=np.linalg.norm(in_wvfunc_data-analytic,ord=2)

    print(f"step_ind={step_ind}, diff={diff}")
    diff_all.append(diff)
t_diff_end=datetime.now()
out_diff_df = pd.DataFrame({'time_step': sorted_time_step_all, 'diff': diff_all})
out_diff_df.to_csv(in_wvfunc_dir+'/out_diff.csv', index=False)

print(f"total diff time: ", t_diff_end-t_diff_start)
# H10R=-half*omegac-half*Deltam-half*g0*sqrt(2*omegam)*cos(omegap*t)*x2+half*omegac**2*x1**2 \
#      +half*omegam*mu*x2**2+g0*omegac*sqrt(2*omegam)*cos(omegap*t)*x1**2*x2+half*I*lmd*sin(theta)
#
# rhs1=H10R*psi
#
# rhs2=-1j*g0*np.sqrt(2/omegam)*sin(omegap*t)*rho*diff(psi,x2)+1j*lmd*np.sin(theta)*x2*diff(psi,x2)
#
# rhs=rhs1+rhs2
#
# lhs=1j*diff(psi,t)
#
# tmp=lhs-rhs
#
#
#
# tmpVal=tmp.subs([(x1,x1Val),(x2,x2Val),(t,tVal)]).evalf()
# pprint(tmpVal)