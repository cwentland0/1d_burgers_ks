import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt
import time

def animatedLinePlot(dataLocs, dataNames, sampRates):

    numSims = len(dataLocs)
    data = [None]*numSims; points = [None]*numSims; samps = [None]*numSims; 
    x = [None]*numSims; 
    minSamps = np.inf; dataMax = -np.inf; dataMin = np.inf;
    for simNum, loc in enumerate(dataLocs):
        data[simNum] = np.load(dataLocs[simNum]) 
        points[simNum], samps[simNum] = data[simNum].shape
        x[simNum] = np.linspace(0,1,num=points[simNum])
        if ( int(samps[simNum]/sampRates[simNum]) < minSamps ):
            minSamps = int(samps[simNum]/sampRates[simNum]) 
        if ( np.amax(data[simNum]) > dataMax):
            dataMax = np.amax(data[simNum])
        if ( np.amin(data[simNum]) < dataMin):
            dataMin = np.amin(data[simNum])

    fig = plt.figure()
    ax = fig.add_subplot(111) 

    colors = ['r','g','b','m','c']
    import pdb; pdb.set_trace
    for t in range(0,minSamps):

        ax.cla()
        for simNum, simData in enumerate(data):
        
            snap = t*sampRates[simNum]
            ax.plot(x[simNum],simData[:,snap],colors[simNum])
            
        ax.set_ylim([dataMin,dataMax]) 
        ax.set_xlim([0,1])
        ax.legend(dataNames)
        plt.pause(0.05) 

        