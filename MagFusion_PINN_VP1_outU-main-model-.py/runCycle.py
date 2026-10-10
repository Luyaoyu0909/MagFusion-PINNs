import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from MagFusion_PINN_Function_v2 import *

# TODO ******************** Data Path ******************** #
# Use a list of paths instead of a single path
dataObs_filepath_list = [

    r"input/Mag_Air984_Sea328_x_y_z_Bx_By_Bz.dat",
    r"input/Mag_Air2460_Sea820_x_y_z_Bx_By_Bz.dat",
    r"input/Mag_Air3444_Sea1148_x_y_z_Bx_By_Bz.dat",
    r"input/Mag_Air4920_Sea1640_x_y_z_Bx_By_Bz.dat"

]

dataCol_filepath = r"input/Mag_Col_random65600_x_y_z-cube.dat"
Log_file = 'Log'

# TODO ******************** Parameters ******************** #
# %% # ! NN Hyper-Parameters
nLayer = 6
nNeuron = 40
actiFun = 'swish'  # 'tanh' 'gelu' 'siren' 'swish' 'relu' 'sigmoid'
opti = 'Adam'  # 'Adam'
DW = 1  # Dynamic Weights 1/0 for On/Off
k = 1
c = 0
# 12000: no noise; 12001: with noise
nEpo = 12000
n0 = 3000
dn = 600  # 2023WuCX
BS = 3000
rdSeed = 12345

# TODO ********** Architecture ********** #
# embedType: 'none' = standard MLP; 'rff' = Random Fourier Features;
embedType = 'none'
numFreqs = 32  # rff: number of frequency components per coordinate (input dim 3->2*m); pos: frequency series per axis (input dim 3->6*m)
freqScale = 4.0  # rff: sigma = 4~10 commonly used
sirenOmega0 = 30.0  # SIREN omega0 (recommended by Sitzmann et al., 2020)

if __name__ == '__main__':

    ######## Load  points ########
    print("Loading Col points data...")
    data_col = np.loadtxt(dataCol_filepath, delimiter=',', skiprows=1)
    xCol = data_col[:, 0:1]
    yCol = data_col[:, 1:2]
    zCol = data_col[:, 2:3]
    numPiont_Col = xCol.shape[0]
    col_Data = [np.float32(xCol), np.float32(yCol), np.float32(zCol)]

    # Iterate over every file in the list
    for dataObs_filepath in dataObs_filepath_list:

        dataset_name = os.path.splitext(os.path.basename(dataObs_filepath))[0]

        print(f"\n{'=' * 60}")
        print(f"*** Start Training on Dataset: {dataset_name} ***")
        print(f"{'=' * 60}")

        # TODO ********************** Data Loading  ************************* #
        ######## Load  points ########
        data_obs = np.loadtxt(dataObs_filepath, delimiter=',', skiprows=1)
        xObs = data_obs[:, 0:1]
        yObs = data_obs[:, 1:2]
        zObs = data_obs[:, 2:3]
        BxObs = data_obs[:, 3:4]
        ByObs = data_obs[:, 4:5]
        BzObs = data_obs[:, 5:6]

        numPiont_Obs = xObs.shape[0]
        Npoints = numPiont_Col + numPiont_Obs
        nBatPerEpo = math.ceil(Npoints / BS)

        obs_Data = [np.float32(xObs), np.float32(yObs), np.float32(zObs),
                    np.float32(BxObs), np.float32(ByObs), np.float32(BzObs)]

        tf.compat.v1.reset_default_graph()
        np.random.seed(rdSeed)
        tf.random.set_seed(rdSeed)

        ######## Network input dim depends on embedding
        if embedType == 'none':
            inDim = 3
        elif embedType == 'rff':
            inDim = 2 * numFreqs
        else:
            raise ValueError("embedType must be 'none', 'rff' ")

        layers = [inDim] + [nNeuron] * nLayer + [1]

        ######## Save dirs for models & logs #########
        embedTag = 'noEmb' if embedType == 'none' else f'{embedType}{numFreqs}sig{freqScale}'
        dir_Name = (f'NN{nLayer}_{nNeuron}_{actiFun}_{embedTag}_{opti}'
                    f'_RADk{k}c{c}n{n0}d{dn}_nEpo{nEpo}_BS{BS}_Seed{rdSeed}')

        Model_file = f'NN_Models/NumObs{numPiont_Obs}/{dir_Name}'
        fileLossVars_file = f"{Log_file}/NumObs{numPiont_Obs}"
        if not os.path.exists(fileLossVars_file):
            os.makedirs(fileLossVars_file)
        fileLossVars_path = f"{fileLossVars_file}/{dir_Name}_LossVars.dat"

        ######## Model & Training
        cfg_info = [f'actiFun={actiFun}', f'embedType={embedType}']
        if embedType == 'rff':
            cfg_info.append(f'numFreqs={numFreqs} (input 3->{2 * numFreqs})')
            cfg_info.append(f'freqScale={freqScale}')
        if actiFun == 'siren':
            cfg_info.append(f'sirenOmega0={sirenOmega0}')

        print(f'*** Training run: {actiFun} + {embedType} ***', flush=True)
        print('>>> Config: ' + ', '.join(cfg_info), flush=True)
        print(f"Number:{Npoints}")
        print(f'BS: {BS}')
        print(f"nBatPerEpo: {nBatPerEpo}")

        model = GT_Lap3D(layers, actiFun, opti, DW, obs_Data, col_Data, nBatPerEpo, BS,
                         embedType=embedType, numFreqs=numFreqs,
                         freqScale=freqScale, sirenOmega0=sirenOmega0)

        model.train(nEpo, nBatPerEpo, BS, Model_file, k, c, n0, dn)

        ######## Save Train Log
        EPOCHs = np.array(range(1, nEpo + 1))[:, None]
        log_data = {
            'Epoch': EPOCHs.flatten(),
            'Loss': model.loss_log,
            'Bx_obs_loss': model.Bx_obs_loss_log,
            'By_obs_loss': model.By_obs_loss_log,
            'Bz_obs_loss': model.Bz_obs_loss_log,
            'Lap_col_loss': model.Lap_col_loss_log,
            'w_obs': model.w_obs_log,
            'Train_Time': model.train_time_log,
            'Learning_Rate': model.lr_log
        }
        df = pd.DataFrame(log_data)
        df.to_csv(fileLossVars_path, index=False, float_format='%6.3e')
        print(f'Saving training info to file: {fileLossVars_path}', flush=True)

        ######## Release TF session/graph resources before next dataset
        model.sess.close()
        print(f"Finished dataset: {dataset_name}\n", flush=True)