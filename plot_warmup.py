import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

groupNum=0  # Fixed typo from 'roupNum'
rowNum=0

inParamFileName="./inParams/inParams"+str(groupNum)+".csv"
# print("file name is "+inParamFileName)
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]

j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])

g0=float(oneRow.loc["g0"])
omegam=float(oneRow.loc["omegam"])
omegap=float(oneRow.loc["omegap"])
omegac=float(oneRow.loc["omegac"])
er=float(oneRow.loc["er"])#magnification

thetaCoef=float(oneRow.loc["thetaCoef"])
tTot=float(oneRow.loc["tTot"])
Q=float(oneRow.loc["Q"])
toWrite=int(oneRow.loc["toWrite"])

print("j1H="+str(j1H)+", j2H="+str(j2H)+", g0="+str(g0) \
      +", omegam="+str(omegam)+", omegap="+str(omegap) \
      +", omegac="+str(omegac)+", er="+str(er)
      +", thetaCoef="+str(thetaCoef)+f", tTot={tTot}, Q={Q}, toWrite={toWrite}")
theta = thetaCoef * np.pi
e2r = er**2
Deltam = omegam - omegap
lmd = (e2r - 1 / e2r) / (e2r + 1 / e2r) * Deltam
D = lmd**2 * np.sin(theta)**2 + omegap**2
mu = lmd * np.cos(theta) + Deltam

def phi(s2):
    return np.exp(-1/2*s2**2)

def psi(x2,t):
    part1=np.exp(1/2*lmd*np.sin(theta)*t)
    part2=phi(x2*np.exp(lmd*np.sin(theta)*t))
    return part1*part2

x2ValsAll=np.linspace(-10,10,1000)
tValsAll=np.linspace(0,5,20)

# Create output directory if it doesn't exist
output_dir = f"./plots_group{groupNum}_row{rowNum}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create and save a plot for each time value
for i, t in enumerate(tValsAll):
    # Calculate psi values for current time
    psi_vals = psi(x2ValsAll, t)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot psi function
    plt.plot(x2ValsAll, psi_vals, 'b-', linewidth=2)

    # Add labels and title
    plt.xlabel('x2', fontsize=12)
    plt.ylabel('ψ(x2, t)', fontsize=12)
    plt.title(f'ψ(x2, t) at t = {t:.3f}', fontsize=14)

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Add parameter info as text on the plot
    param_text = f'λ = {lmd:.3f}, θ/π = {thetaCoef:.3f}'
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save figure
    filename = f"{output_dir}/psi_t_{i:03d}_value_{t:.3f}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Saved plot {i+1}/{len(tValsAll)}: {filename}")

print(f"\nAll plots saved in {output_dir}/")