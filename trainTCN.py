import sys
sys.path.append('/home/chris/Research/Libraries/keras-tcn/tcn')
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow.keras.backend as K
from tcn import TCN
from sklearn.model_selection import train_test_split
import math
import h5py
import numpy as np 
from annFuncs import scaleOp, extractJacobian 
import matplotlib.pyplot as plt 

# reorganize data matrix into correct format for TCN 
# data assumed to be in [nchannels x timesteps]
# need to get it into [batch_size x timesteps x nchannels]
def constructReceptField(lbWindow, dataMat):

    nchannels,totTimesteps = dataMat.shape
    for step in range(lbWindow,totTimesteps+1):
        # window = np.reshape(dataMat[:,step-lbWindow:step].T,(1,lbWindow,nchannels))
        window = np.expand_dims(dataMat[:,step-lbWindow:step].T,axis=0)
        if (step == lbWindow):
            dataWindows = window 
        else:
            dataWindows = np.append(dataWindows,window,axis=0) 

    return dataWindows

def main(): 

    modelLabel = input("Input the name of this model: ")

    ####### RUNTIME PARAMETERS ######
    firstTime = False

    projFOMDir = './Data/TCNTraining_POD_k'
    fullFOMDir = './Data/trainingData'
    decoderLoc  = './Models/decoder_k10_param_kookjin.h5'
    podBasisLoc = './Data/PODBasis/podBasis_param_samp1.npy'
    normDataLoc = './Data/PODBasis/normData_param_samp1.npy'
    projType = 'PODG'

    valPerc = 0.1 

    romSize = 10
    projFOMDir = projFOMDir+str(romSize)

    subSamp = False 
    sampRate = 10 

    ######## MODEL PARAMETERS #######
    # these need to be set first to determine the lookback window
    maxEpochs = 1000 
    earlyStopEpochs = 1000 
    batchSize = 50 
    learn_rate = 0.002 
    decay_rate = 0. 
    initializationDist = 'glorot_uniform'

    numLayers = 3
    activations = ['relu','relu','linear']
    kernel_size = [5]*3
    dilationOrder = [3]*3

    loss_func = mean_squared_error

    lookbackWindow = kernel_size[0]*(2**dilationOrder[0])

    ###### LOAD DATA AND ORGANIZE ######
    # need to compute the closure term if this is the first time running the model
    if firstTime:
        normData = np.load(normDataLoc)
        if (projType == 'PODG'): 
            VMat = np.load(podBasisLoc) 
            VMat = VMat[:,:romSize]
        elif(projType == 'GMan'):
            decoder = load_model(decoderLoc)

    ####### LOAD PARAMETRIC DATA ########
    mu_1_vals = 4.25 + (1.25/9.)*np.linspace(0,9,10)
    mu_2_vals = 0.015 + (0.015/7)*np.linspace(0,7,8)
    MU_1, MU_2 = np.meshgrid(mu_1_vals,mu_2_vals)
    MU_1_vec = MU_1.flatten(order='F')
    MU_2_vec = MU_2.flatten(order='F')

    for paramIter,mu_1 in enumerate(MU_1_vec): 
        print("Loading parameter instance: "+str(paramIter+1))
        mu_2 = MU_2_vec[paramIter]
        outputLabel = '_burgers_mu1_'+str(mu_1)+'_mu2_'+str(mu_2)

        # load code (always necessary)
        codeLoad = np.load(os.path.join(projFOMDir,'code'+outputLabel+'_'+projType+'.npy'))
        if subSamp:
            codeLoad = codeLoad[:,::sampRate] 
        
        # compute closure if not already computed and stored
        if firstTime:

            # load RHS and RHS wrt projected solution
            RHSLoad     = np.load(os.path.join(fullFOMDir,'RHS'+outputLabel+'_FOM.npy'))
            RHSProjLoad = np.load(os.path.join(projFOMDir,'RHS'+outputLabel+'_'+projType+'.npy'))

            # load data necessary to compute closure, plus code 
            if subSamp:
                RHSLoad = RHSLoad[:,::sampRate]
                RHSProjLoad = RHSProjLoad[:,::sampRate]

            # compute exact closure term
            closureLoad = np.zeros(codeLoad.shape,dtype=np.float64)
            if (projType == 'PODG'): 
                closureLoad = np.dot(VMat.T,RHSLoad - RHSProjLoad)
                closureLoad = scaleOp(closureLoad,normData) 
            elif (projType == 'GMan'): 
                for samp in range(codeLoad.shape[1]):
                    jacob = extractJacobian(decoder,np.squeeze(codeLoad[:,samp]))
                    closureLoad[:,samp] = np.dot(np.linalg.pinv(jacob),RHSLoad[:,samp] - RHSProjLoad[:,samp]) 
                closureLoad = scaleOp(closureLoad,romParams['normData']) 

            # save to file so this doesn't need to be computed again
            np.save(os.path.join(projFOMDir,'closure'+outputLabel+'_'+projType+'.npy'),closureLoad) 
        
        # just load from file
        else:
            closureLoad = np.load(os.path.join(projFOMDir,'closure'+outputLabel+'_'+projType+'.npy'))
            if subSamp:
                closureLoad = closureLoad[:,::sampRate]

        # organize data 
        if (paramIter == 0):
            codeFull = constructReceptField(lookbackWindow,codeLoad)
            closureFull = closureLoad[:,lookbackWindow-1:]
        else:
            codeFull = np.append(codeFull,constructReceptField(lookbackWindow,codeLoad),axis=0)
            closureFull = np.append(closureFull,closureLoad[:,lookbackWindow-1:],axis=1)

    closureFull = closureFull.T

    # closureMax = np.amax(closureFull)
    # closureMin = np.amin(closureFull)
    # codeMax = np.amax(codeFull) 
    # codeMin = np.amin(codeFull)
    # figMode = plt.figure()
    # axMode = figMode.add_subplot(111)
    # axMode2 = axMode.twinx()
    # numSamps,numModes = closureFull.shape
    # modeNums = np.arange(0,numModes) + 1
    # for i in range(numSamps):
    #     axMode.clear()
    #     axMode2.clear()
    #     axMode.plot(modeNums,codeFull[i,-1,:],'b')
    #     axMode.set_ylim([codeMin,codeMax])
    #     axMode2.plot(modeNums,closureFull[i,:],'r')
    #     axMode2.set_ylim([closureMin,closureMax])
    #     plt.pause(0.01)

    # import pdb; pdb.set_trace()

    ####### TRAINING AND VALIDATION SETS #######
    codeTrain, codeVal, closureTrain, closureVal  = train_test_split(codeFull,closureFull,test_size=valPerc,random_state=24)
    codeTrainMin = np.amin(codeTrain)
    codeTrainMax = np.amax(codeTrain)
    codeTrain = (codeTrain - codeTrainMin)/(codeTrainMax - codeTrainMin)
    codeValMin = np.amin(codeVal)
    codeValMax = np.amax(codeVal)
    codeVal = (codeVal - codeValMin)/(codeValMax - codeValMin)
    np.save(os.path.join(projFOMDir,'normData_'+modelLabel+'.npy'),np.array([codeTrainMin,codeTrainMax]))

    ######## MODEL CONSTRUCTION AND TRAINING ##########
    def tcnClosure():
        input1 = Input(shape=codeTrain.shape[1:],name='inputCode')
        x = input1
        for layer in range(numLayers-1):
            x = TCN(nb_filters = romSize,
                    kernel_size = kernel_size[layer],
                    padding = 'causal',
                    return_sequences = True,
                    activation = activations[layer],
                    kernel_initializer=initializationDist,
                    dilations = [2**i for i in range(dilationOrder[layer])],
                    name='tcn_'+str(layer+1))(x) 

        x = TCN(nb_filters = romSize,
                    kernel_size = kernel_size[-1],
                    padding = 'causal',
                    return_sequences = False,
                    activation = activations[-1],
                    kernel_initializer=initializationDist,
                    dilations = [2**i for i in range(dilationOrder[-1])],
                    name='tcn_'+str(numLayers))(x)

        return Model(input1,x)

    TCNModel = tcnClosure()
    TCNModel.summary()

    opt_func = Adam(lr=learn_rate,decay=decay_rate)
    TCNModel.compile(optimizer=opt_func,loss=loss_func)

    earlyStop = EarlyStopping(patience=earlyStopEpochs,restore_best_weights=True)
    TCNModel.fit(x=codeTrain,y=closureTrain,batch_size=batchSize,epochs=maxEpochs,
        validation_data=(codeVal,closureVal),verbose=2,callbacks=[earlyStop])

    ####### SAVE MODEL #######
    if not os.path.exists('./Models'): os.makedirs('./Models') 
    modelName = './Models/TCN_'+modelLabel+'.h5'
    TCNModel.save(modelName)

if __name__ == "__main__":
    main()