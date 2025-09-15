import pandas as pd
from pathlib import Path
import sys
from decimal import Decimal, getcontext
import shutil

#python readCSV.py groupNum rowNum
#this script reads csv and creates directory
if len(sys.argv)!=3:
    print("wrong number of arguments")

groupNum=int(sys.argv[1])
rowNum=int(sys.argv[2])

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

# print("j1H="+str(j1H)+", j2H="+str(j2H)+", g0="+str(g0)\
#       +", omegam="+str(omegam)+", omegap="+str(omegap)\
#       +", omegac="+str(omegac)+", er="+str(er)+", thetaCoef="+str(thetaCoef))


outDir="./outData/group"+str(groupNum)+"/row"+str(rowNum)+"/"

Path(outDir).mkdir(exist_ok=True,parents=True)

def format_using_decimal(value, precision=10):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

j1HStr=format_using_decimal(j1H)
j2HStr=format_using_decimal(j2H)

g0Str=format_using_decimal(g0)

omegam_Str=format_using_decimal(omegam)

omegap_Str=format_using_decimal(omegap)

omegac_Str=format_using_decimal(omegac)

erStr=format_using_decimal(er)

thetaCoef_Str=format_using_decimal(thetaCoef)
parallel_num=24

params2cppInFile=[
    j1HStr+"\n",
    j2HStr+"\n",
    g0Str+"\n",
    omegam_Str+"\n",
    omegap_Str+"\n",
    omegac_Str+"\n",
    erStr+"\n",
    thetaCoef_Str+"\n",
    str(groupNum)+"\n",
    str(rowNum)+"\n",
    str(parallel_num)+"\n",


    ]

cppInParamsFileName=outDir+"/cppIn.txt"
with open(cppInParamsFileName,"w+") as fptr:
    fptr.writelines(params2cppInFile)


shutil.copy(inParamFileName,outDir)