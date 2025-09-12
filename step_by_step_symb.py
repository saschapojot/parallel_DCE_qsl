from sympy import *
from sympy import expand_complex
from sympy.simplify.fu import TR11,TR5
import pandas as pd
import numpy as np


#symbolic computation for step by step notes

g0,lmd,theta=symbols("g0,lambda,theta",cls=Symbol,real=True)
x1,tau,s2,t=symbols("x1,tau,s2,t",cls=Symbol,real=True)
Deltam=symbols("Delta_m",cls=Symbol,real=True)
omegam,omegac,omegap=symbols("omega_m,omega_c,omega_p",cls=Symbol,positive=True)
x2,dx2=symbols("x2,dx2",cls=Symbol,real=True)
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
half=Rational(1,2)
quarter=Rational(1,4)
rho=omegac*x1**2-half

D=lmd**2*sin(theta)**2+omegap**2
mu=lmd*cos(theta)+Deltam






x2_in_tau=g0*sqrt(2/omegam)*rho*(lmd*sin(theta)*sin(omegap*tau)-omegap*cos(omegap*tau))/D\
    +(s2+g0*sqrt(2/omegam)*rho*omegap/D)*exp(-lmd*sin(theta)*tau)


x2_in_exp=g0*sqrt(2/omegam)*rho*(-I*half*lmd*sin(theta)-half*omegap)/D*exp(I*omegap*tau)\
    +g0*sqrt(2/omegam)*rho*(I*half*lmd*sin(theta)-half*omegap)/D*exp(-I*omegap*tau)\
    +(s2+g0*sqrt(2/omegam)*rho*omegap/D)*exp(-lmd*sin(theta)*tau)





c2=g0*sqrt(2/omegam)*sin(omegap*t)*rho-lmd*sin(theta)*x2
c0=-I*half*omegam*mu*x2**2-I*g0*sqrt(2*omegam)*cos(omegap*t)*x2*rho

z1_lhs=I*g0**2/(4*omegap*D**2)\
       *(mu*lmd**2*sin(theta)**2-mu*omegap**2+2*omegap*D)*rho**2*sin(2*omegap*t)

z2_lhs=I*g0**2/(2*omegap*D**2)\
    *lmd*sin(theta)*(D-mu*omegap)*rho**2*cos(2*omegap*t)

z3_lhs=I*omegam*mu/(4*lmd*sin(theta))* \
       (x2-g0/D*sqrt(2/omegam)*rho*(lmd*sin(theta)*sin(omegap*t)-omegap*cos(omegap*t)))**2

z4_lhs=I*g0**2/D*(omegap-mu/2)*rho**2*t

z5_lhs=I*g0/D*sqrt(2*omegam)*(mu-omegap)*rho* \
       (x2-g0/D*sqrt(2/omegam)*rho*\
        (lmd*sin(theta)*sin(omegap*t)-omegap*cos(omegap*t)))*sin(omegap*t)


z6_lhs=I*g0/D*sqrt(2*omegam)*lmd*sin(theta)*rho*(x2-g0/D*sqrt(2/omegam)*rho*(lmd*sin(theta)*sin(omegap*t)-omegap*cos(omegap*t)))*cos(omegap*t)

z7_lhs=-I*g0**2/(2*omegap*D**2)*lmd*sin(theta)*(D-mu*omegap)*rho**2

lhs=z1_lhs+z2_lhs+z3_lhs+z4_lhs+z5_lhs+z6_lhs+z7_lhs


rhs_const=I*omegam*mu/(4*lmd*sin(theta))*x2**2+I*g0**2/D**2*rho**2*((2*omegap-D/(2*omegap)-mu/2)*lmd*sin(theta)+mu*D/(4*lmd*sin(theta)))

rhs_t=I*g0**2/D*(omegap-mu/2)*rho**2*t

rhs_sin_2t=I*g0**2/(4*D*omegap)*(mu-2*omegap)*rho**2*sin(2*omegap*t)

rhs_cos_2t=I*g0**2/D**2*\
           (2*lmd**2*D*sin(theta)**2+4*mu*lmd**2*omegap*sin(theta)**2+mu*omegap**3-3*mu*lmd**2*omegap*sin(theta)**2)/(4*lmd*omegap*sin(theta))*rho**2*cos(2*omegap*t)

rhs_sin_t=I*g0/D*(half*mu-omegap)*sqrt(2*omegam)*rho*x2*sin(omegap*t)

rhs_cos_t=I*g0/D*sqrt(2*omegam)*(mu*omegap+2*lmd**2*sin(theta)**2)/(2*lmd*sin(theta))*rho*x2*cos(omegap*t)

F0=I*omegam*mu/(4*lmd*sin(theta))*x2**2+I*g0**2/D**2*rho**2*\
   ((2*omegap-D/(2*omegap)-mu/2)*lmd*sin(theta)+mu*D/(4*lmd*sin(theta)))



F1=I*g0**2/D*(omegap-mu/2)*rho**2


F2=I*g0**2/D**2*\
   (2*lmd**2*D*sin(theta)**2+4*mu*lmd**2*omegap*sin(theta)**2+mu*omegap**3-3*mu*lmd**2*omegap*sin(theta)**2)/(4*lmd*omegap*sin(theta))*rho**2



F3=I*g0/D*(half*mu-omegap)*sqrt(2*omegam)*rho*x2


F4=I*g0/D*sqrt(2*omegam)*(mu*omegap+2*lmd**2*sin(theta)**2)/(2*lmd*sin(theta))*rho*x2



F5=-I*mu/(4*lmd*sin(theta)*D)*(D*omegam*x2**2+g0**2*rho**2)


F6=I*mu*g0/D*sqrt(omegam/2)*rho*x2


F7=-I*mu*g0/(lmd*sin(theta)*D)*sqrt(omegam/2)*omegap*rho*x2


F8=I*mu*g0**2/(4*D**2*lmd*sin(theta))*(lmd**2*sin(theta)**2-omegap**2)*rho**2

F9=I*mu*g0**2/(2*D**2)*omegap*rho**2

z8_lhs=-I*omegam*mu/(4*lmd*sin(theta))\
       *(x2*exp(lmd*sin(theta)*t)-g0/D*sqrt(2/omegam)*rho*(lmd*sin(theta)*sin(omegap*t)-omegap*cos(omegap*t))*exp(lmd*sin(theta)*t))**2


F10=-I*g0/D*sqrt(2*omegam)*lmd*sin(theta)*rho*x2

F11=I*2*g0**2/D**2*lmd**2*sin(theta)**2*rho**2

F12=-I*2*g0**2/D**2*lmd*omegap*sin(theta)*rho**2

z9_lhs=-I*g0/D*sqrt(2*omegam)*lmd*sin(theta)*rho*(x2*exp(lmd*sin(theta)*t)-g0/D*sqrt(2/omegam)*rho*(lmd*sin(theta)*sin(omegap*t)-omegap*cos(omegap*t))*exp(lmd*sin(theta)*t))


z9_rhs=F10*exp(lmd*sin(theta)*t)+F11*sin(omegap*t)*exp(lmd*sin(theta)*t)+F12*cos(omegap*t)*exp(lmd*sin(theta)*t)


G=F0+F1*t-1/(2*omegap)*F1*sin(2*omegap*t)\
    +F2*cos(2*omegap*t)+F3*sin(omegap*t)+F4*cos(omegap*t)\
    +F5*exp(2*lmd*sin(theta)*t)+F6*sin(omegap*t)*exp(2*lmd*sin(theta)*t)\
    +F7*cos(omegap*t)*exp(2*lmd*sin(theta)*t)+F8*cos(2*omegap*t)*exp(2*lmd*sin(theta)*t)+F9*sin(2*omegap*t)*exp(2*lmd*sin(theta)*t)\
    +F10*exp(lmd*sin(theta)*t)+F11*sin(omegap*t)*exp(lmd*sin(theta)*t)+F12*cos(omegap*t)*exp(lmd*sin(theta)*t)

val0=F0
val1=F1*t
val2=-1/(2*omegap)*F1*sin(2*omegap*t)
val3=F2*cos(2*omegap*t)

val4=F3*sin(omegap*t)
val5=F4*cos(omegap*t)

val6=F5*exp(2*lmd*sin(theta)*t)
val7=F6*sin(omegap*t)*exp(2*lmd*sin(theta)*t)

val8=F7*cos(omegap*t)*exp(2*lmd*sin(theta)*t)
val9=F8*cos(2*omegap*t)*exp(2*lmd*sin(theta)*t)

val10=F9*sin(2*omegap*t)*exp(2*lmd*sin(theta)*t)
val11=F10*exp(lmd*sin(theta)*t)

val12=F11*sin(omegap*t)*exp(lmd*sin(theta)*t)
val13=F12*cos(omegap*t)*exp(lmd*sin(theta)*t)





s2_br=-g0/D*omegap*sqrt(2/omegam)*rho+x2*exp(lmd*sin(theta)*t)\
    -g0/D*sqrt(2/omegam)*rho*(lmd*sin(theta)*sin(omegap*t)-omegap*cos(omegap*t))*exp(lmd*sin(theta)*t)


s2=-g0/D*omegap*sqrt(2/omegam)*rho+x2*exp(lmd*sin(theta)*t)\
    -g0/D*sqrt(2/omegam)*sin(theta)*rho*sin(omegap*t)*exp(lmd*sin(theta)*t)*lmd\
    +g0/D*sqrt(2/omegam)*omegap*rho*cos(omegap*t)*exp(lmd*sin(theta)*t)


phase=exp((-I*half*omegac*rho+I*quarter*omegac+I*half*Deltam+half*lmd*sin(theta))*t)

psi=phase*exp(G)

H1R_poly=half*omegac*rho-quarter*omegac-half*Deltam\
    +half*omegam*mu*x2**2+g0*sqrt(2*omegam)*cos(omegap*t)*x2*rho+half*I*lmd*sin(theta)



H10R=-half*omegac-half*Deltam-half*g0*sqrt(2*omegam)*cos(omegap*t)*x2+half*omegac**2*x1**2\
    +half*omegam*mu*x2**2+g0*omegac*sqrt(2*omegam)*cos(omegap*t)*x1**2*x2+half*I*lmd*sin(theta)


beta=(-I*half*omegac*rho+I*quarter*omegac+I*half*Deltam+half*lmd*sin(theta))*t


x1_tmp=1
x2_tmp=2
tau_tmp=0.1
beta_val=beta.subs([(x1,x1_tmp),(x2,x2_tmp),(t,tau_tmp)])
pprint(beta_val.evalf())



