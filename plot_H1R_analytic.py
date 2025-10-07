import pandas as pd
import numpy as np
import glob
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from sympy import *
from sympy.simplify.fu import TR11,TR5
import re
import os
from pathlib import Path
from scipy import sparse



import sys
if len(sys.argv)!=3:
    print("wrong number of arguments")

grpNum=int(sys.argv[1])
rowNum=int(sys.argv[2])

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
L2 = 30;
N1=270*2
N2 = 1000
# print(2/(np.abs(lmd*np.sin(theta))*N2))
dx1 = 2.0 * L1 /N1
dx2 = 2.0 * L2 / N2
x1ValsAll=np.array([-L1 + dx1 * n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2 + dx2 * n2 for n2 in range(0,N2)])
dx1 = 2.0 * L1 / N1
dx2 = 2.0 * L2 /N2
tTot = 1.0
Q = 1e2
dt = tTot /Q
print(f"dx1={dx1}, dx2={dx2}, dt={dt}")
#construct H6

leftMat=sparse.diags(-2*np.ones(N1),offsets=0,format="lil",dtype=complex) \
        +sparse.diags(np.ones(N1-1),offsets=1,format="lil",dtype=complex) \
        +sparse.diags(np.ones(N1-1),offsets=-1,format="lil",dtype=complex)

H6=-1/(2*dx1**2)*sparse.kron(leftMat,sparse.eye(N2,dtype=complex,format="lil"),format="lil")

#compute <Nc>
tmp0=sparse.diags(x1ValsAll**2,format="lil")
IN2=sparse.eye(N2,dtype=complex,format="lil")
NcMat1=sparse.kron(tmp0,IN2)

def avgNc(one_psi):
    """


    :return: number of photons for Psi
    """

    val=1/2*omegac*np.vdot(one_psi,NcMat1@one_psi)-1/2*np.vdot(one_psi,one_psi)+1/omegac*np.vdot(one_psi,H6@one_psi)

    return val

# compute Nm
S2=sparse.diags(np.power(x2ValsAll,2),format="csc")
Q2=sparse.diags(-2*np.ones(N2),offsets=0,format="csc",dtype=complex) \
   +sparse.diags(np.ones(N2-1),offsets=1,format="csc",dtype=complex) \
   +sparse.diags(np.ones(N2-1),offsets=-1,format="csc",dtype=complex)

IN1=sparse.eye(N1,dtype=complex,format="lil")
NmPart1=sparse.kron(IN1,S2)
NmPart2=sparse.kron(IN1,Q2)

def avgNm(one_psi):
    """

    :param j:  time step j, wavefunction is Psi
    :return: number of phonons for Psi
    """


    val=1/2*omegam*np.vdot(one_psi,NmPart1@one_psi)-1/2*np.vdot(one_psi,one_psi)-1/(2*omegam*dx2**2)*np.vdot(one_psi,NmPart2@one_psi)

    return val
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
    # nm_tmp=np.linalg.norm(solution, ord="fro")
    # print(f"t={t}, norm={nm_tmp}")
    solution_flattened=solution.flatten()
    Nc=avgNc(solution_flattened)
    Nm=avgNm(solution_flattened)
    print(f"Nm={Nm}, Nc={Nc}")
    return solution_flattened
# Create directory for saving plots
# Create directory for saving plots
output_dir = f"./plots_group{grpNum}_row{rowNum}"
Path(output_dir).mkdir(exist_ok=True,parents=True)
# Set up the figure parameters with larger fonts
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Generate plots at specific intervals
time_steps = np.linspace(0, tTot, int(Q)+1)
step_interval = 5  # Plot every 10th time step

for i in range(0, len(time_steps), step_interval):
    t_val = time_steps[i]
    print(f"Computing and plotting time step {i}/{len(time_steps)}: t = {t_val:.4f}")

    # Compute solution at this time
    sol = generate_discrete_solution_normalized_parallel(t_val)
    psi = sol.reshape(N1, N2)
    psi_abs = np.abs(psi)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create mesh grid for plotting
    X1_grid, X2_grid = np.meshgrid(x1ValsAll, x2ValsAll, indexing='ij')

    # Create the color plot
    im = ax.pcolormesh(X1_grid, X2_grid, psi_abs,
                       shading='auto', cmap='viridis')

    # Add colorbar with larger font
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|ψ(x₁, x₂)|', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    # Set labels and title with larger fonts
    ax.set_xlabel('x₁', fontsize=20)
    ax.set_ylabel('x₂', fontsize=20)
    ax.set_title(f'Wavefunction Absolute Value at t = {t_val:.4f}', fontsize=22)

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save the figure
    plt.savefig(output_dir+f'/psi_abs_t{i:04d}.png', dpi=100, bbox_inches='tight')
    plt.close()

print(f"Saved plots in 'plots' directory")