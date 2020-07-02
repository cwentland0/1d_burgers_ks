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

def plotErrMultiple():

    fomDir = './Data/u_burgers_mu1_4.3_mu2_0.021_FOM.npy'
    fomSol = np.load(fomDir)
    romDirs = ['./Data/u_burgers_mu1_4.3_mu2_0.021_PODG.npy',
               './Data/u_burgers_mu1_4.3_mu2_0.021_PODG-MZ.npy',
               './Data/u_burgers_mu1_4.3_mu2_0.021_GMan_decoder.npy',
               './Data/u_burgers_mu1_4.3_mu2_0.021_GMan_encoder.npy']

    plotStyles = ['b','b--','r','r--']
    plotLabels = ['PODG','PODG-MZ','NLM (decoder)','NLM (encoder)']

    nX, nSamp = fomSol.shape

    tf = 35
    t = np.linspace(0,tf,nSamp)

    figErr = plt.figure()
    axErr = figErr.add_subplot(111)

    for simNum, romLoc in enumerate(romDirs):
        romSol = np.load(romLoc)

        l2Err_time = np.sqrt(np.sum(np.square(fomSol - romSol),axis=0))


        axErr.plot(t,l2Err_time,plotStyles[simNum])
        axErr.set_xlabel('t')
        axErr.set_ylabel('L2 Error')
        plt.savefig('./Images/l2Err.png')

    axErr.legend(plotLabels)
    plt.show()