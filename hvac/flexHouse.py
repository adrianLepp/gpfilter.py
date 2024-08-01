# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import GPy

dir = 'flexHouseData'


'''
Wie sehen meine Daten von Flex house aus?


Ich habe 7 Zonen, von denen jeweils folgende Werte erfasst werden:

- T	Innentemperatur
- H	binäre Heizungssteuerung?
- S	Steuergröße?
- HT	Vergleichbar zur Innentemperatur
- HRELH	größer als Innentemperatur
- HI	vergleichbarer Verlauf zur Sonneneinstrahlung
- Hco2	konstant, wenn Temperatur konst geregelt wird

folgende weitere Daten stehen zur Verfügung:
- P	Heizenergie?
- G	Sonneneinstrahlung
- Ta	Außentemperatur
- WD	?
- Ws	?

Was für Modelle könnte ich aufstellen?

dT = f (T,P, G, Ta)

- Grey Box Model mit thermodynamischen Differentialgleichungen
- Gaussian Process als rein datengetrieben.

---------------------------------------------------------------------------

DATA

PRBS

T0 - T7     Temperatur in Zonen
H0 - H7:    binäre Heizsteuerung
S0 - S7 = 0
P           wie H nur = 5000
G           Solar irridiation < 0.30
Ta          Außtenmemperatur 0 < Ta < 5
Wd          0 < Wd < 350
Ws          1 < Wd < 7
HT0 - HT7   vgl temperature Kurven? 14 < < 28
HRelH1 - 7  vtl Temperature nur 24 < HrelH < 37
HI1 - HI7   starke Peaks  für jeden Tag <3500, sonst sehr klein
HcO2_1 - 7  sehr zeitdiskret mit einem Ausreißer.


THERM

T0 - 7      nahezu konst auf 20
S0 - 7      Peaks vorhanden
P           varriierende Heizung
HT0 - HT7   wie T
HI0 - 7     vergleiche Sonneneinstrahlung
HC02_1 - 7  nahehzu konst
'''


def getCSVData(file_path: str, sep = ',', decimal= '.') -> dict:
    df = pd.read_csv(file_path, sep=sep, decimal=decimal)
    columns = df.columns

    data = {dp: [] for dp in columns}

    for dpName in columns:
        data[dpName] = df[dpName].tolist()

    return data

def filterData(headers, data:dict) -> dict:
    filteredData = {}
    for header in headers:
        filteredData[header] = data[header]
    return filteredData

def plotData(header:str, time, data:dict):
    plt.figure()
    plt.plot(time, data)
    #plt.plot(data)
    plt.xlabel('time ')
    plt.title(header)
    #plt.close()
    
# %% Get data
    

filename = '5MinTHERM3'

flexHouseData = getCSVData(dir + '/' + filename + '.csv' )

# %% plot data


t = flexHouseData['time']

ts = []

for i in range(len(t)):
    ts.append(datetime.fromisoformat(t[i]).timestamp())
    

testData = filterData(['time', 'T0', "H0", 'P', 'G','Ta'],flexHouseData)

testData['dT0'] = []

for i in range(len(testData['T0'])-1):
    testData['dT0'].append(testData['T0'][i+1]-testData['T0'][i])
    
testData['T0'] = testData['T0'][0:-1]
testData['P'] = testData['P'][0:-1] 
testData['G'] = testData['G'][0:-1]
testData['Ta'] = testData['Ta'][0:-1]

#ts = (datetime.fromisoformat(date).timestamp() for date in t)

#ts = ts - ts[0]

#ts = (time - ts[0] for time in ts)

#ts = ts / 1000 / 60 #ts should be in minutes now

#TODO: convert datetime to ts

# for key in testData.keys():
#     plotData(key, ts, testData[key])

# %% init a GP


xD = np.array([testData['T0'], testData['P'],testData['G'], testData['Ta']]).transpose()
yD = np.array([testData['dT0']]).transpose()

gp = GPy.models.GPRegression(xD,yD)

gp.optimize()


xTest = xD[1:500,:]
yTest = yD[1:500,:]
yP = np.zeros(yTest.size)

# %%

for i in range(len(xTest)):
    yP[i], _ = gp.predict(xTest[i:i+1,:])
    
    
plt.figure()
plt.plot(yP)





