import pandas as pd
import numpy as np
from scipy.special import hermite
import glob
import re
import pickle

grpNum=0
rowNum=0
inCsvFileName=f"./inParams/inParams{grpNum}.csv"

df=pd.read_csv(inCsvFileName)
row=df.iloc[rowNum,:]
j1H=row["j1H"]

j2H=row["j2H"]
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
j1H = int(row["j1H"])
j2H = int(row["j2H"])
g0 = row["g0"]
omegam = row["omegam"]
omegap = row["omegap"]
omegac = row["omegac"]
er = row["er"]
thetaCoef = row["thetaCoef"]

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
def f1(x1Val,j1H):
    part1=np.exp(-1/2*omegac*x1Val**2)
    # print(f"np.sqrt(omegac)*x1Val={np.sqrt(omegac)*x1Val}")
    part2=hermite(j1H)(np.sqrt(omegac)*x1Val)
    # print(f"part2={part2}")
    return part1*part2


def f2(x2Val,j2H):
    part1=np.exp(-1/2*omegam*x2Val**2)
    part2=hermite(j2H)(np.sqrt(omegam)*x2Val)
    return part1*part2

def s2Func(x1,x2,t):
    rho=omegac*x1**2-half
    return -g0/D*omegap*np.sqrt(2/omegam)*rho+x2*np.exp(lmd*np.sin(theta)*t) \
        -g0/D*np.sqrt(2/omegam)*np.sin(theta)*rho*np.sin(omegap*t)*np.exp(lmd*np.sin(theta)*t)*lmd \
        +g0/D*np.sqrt(2/omegam)*omegap*rho*np.cos(omegap*t)*np.exp(lmd*np.sin(theta)*t)
def psi(x1,x2,t):
    rho=omegac*x1**2-half

    s2=s2Func(x1,x2,t)

    phi_part1=f1(x1,j1H)
    phi_part2=f2(s2,j2H)
    phi_val=phi_part1*phi_part2

    F0=1j*omegam*mu/(4*lmd*np.sin(theta))*x2**2+1j*g0**2/D**2*rho**2* \
       ((2*omegap-D/(2*omegap)-mu/2)*lmd*np.sin(theta)+mu*D/(4*lmd*np.sin(theta)))
    F1=1j*g0**2/D*(omegap-mu/2)*rho**2

    F2=1j*g0**2/D**2* \
       (2*lmd**2*D*np.sin(theta)**2+4*mu*lmd**2*omegap*np.sin(theta)**2+mu*omegap**3-3*mu*lmd**2*omegap*np.sin(theta)**2)/(4*lmd*omegap*np.sin(theta))*rho**2


    F3=1j*g0/D*(half*mu-omegap)*np.sqrt(2*omegam)*rho*x2

    F4=1j*g0/D*np.sqrt(2*omegam)*(mu*omegap+2*lmd**2*np.sin(theta)**2)/(2*lmd*np.sin(theta))*rho*x2

    F5=-1j*mu/(4*lmd*np.sin(theta)*D)*(D*omegam*x2**2+g0**2*rho**2)

    F6=1j*mu*g0/D*np.sqrt(omegam/2)*rho*x2

    F7=-1j*mu*g0/(lmd*np.sin(theta)*D)*np.sqrt(omegam/2)*omegap*rho*x2

    F8=1j*mu*g0**2/(4*D**2*lmd*np.sin(theta))*(lmd**2*np.sin(theta)**2-omegap**2)*rho**2

    F9=1j*mu*g0**2/(2*D**2)*omegap*rho**2

    F10=-1j*g0/D*np.sqrt(2*omegam)*lmd*np.sin(theta)*rho*x2

    F11=1j*2*g0**2/D**2*lmd**2*np.sin(theta)**2*rho**2

    F12=-1j*2*g0**2/D**2*lmd*omegap*np.sin(theta)*rho**2

    G=F0+F1*t-1/(2*omegap)*F1*np.sin(2*omegap*t) \
      +F2*np.cos(2*omegap*t)+F3*np.sin(omegap*t)+F4*np.cos(omegap*t) \
      +F5*np.exp(2*lmd*np.sin(theta)*t)+F6*np.sin(omegap*t)*np.exp(2*lmd*np.sin(theta)*t) \
      +F7*np.cos(omegap*t)*np.exp(2*lmd*np.sin(theta)*t)+F8*np.cos(2*omegap*t)*np.exp(2*lmd*np.sin(theta)*t)+F9*np.sin(2*omegap*t)*np.exp(2*lmd*np.sin(theta)*t) \
      +F10*np.exp(lmd*np.sin(theta)*t)+F11*np.sin(omegap*t)*np.exp(lmd*np.sin(theta)*t)+F12*np.cos(omegap*t)*np.exp(lmd*np.sin(theta)*t)

    beta=(-1j*half*omegac*rho+1j*quarter*omegac+1j*half*Deltam+half*lmd*np.sin(theta))*t

    exp_val=np.exp(G+beta)

    return phi_val*exp_val




L1 = 5
L2 = 8
N1=270
N2 = 300

dx1 = 2.0 * L1 /N1
dx2 = 2.0 * L2 / N2

tTot = 1.0
Q = 1e3
dt = tTot /Q

print(f"dx1={dx1}, dx2={dx2}, dt={dt}")
x1ValsAll=[-L1 + dx1 * n1 for n1 in range(0,N1)]
x2ValsAll=[-L2 + dx2 * n2 for n2 in range(0,N2)]

def generate_discrete_solution_normalized(t):
    """

    :param t: time
    :return: solution, row major order
    """
    solution=np.zeros((N1,N2),dtype=complex)
    for n1 in range(0,N1):
        for n2 in range(0,N2):
            x1Tmp=x1ValsAll[n1]
            x2Tmp=x2ValsAll[n2]
            valTmp=psi(x1Tmp,x2Tmp,t)
            solution[n1,n2]=valTmp

    solution/=np.linalg.norm(solution,ord="fro")
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


for j in range(0,len(sorted_time_step_all)):
    step_ind=sorted_time_step_all[j]
    file=sortedDataFiles[j]
    with open(file,"rb") as fptr:
        in_wvfunc_data=pickle.load(fptr)
    in_wvfunc_data=np.array(in_wvfunc_data)
    analytic=generate_discrete_solution_normalized(step_ind*dt)
    diff=np.linalg.norm(in_wvfunc_data-analytic,ord=2)

    print(f"step_ind={step_ind}, diff={diff}")
