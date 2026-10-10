import math
from MagFusion_PINN_Function_v2 import *
import pandas as pd
# TODO ******************** Data Path ******************** #
dataObs_filepath = r"input/Mag_Air1476_Sea492_x_y_z_Bx_By_Bz_snr30.dat"
dataCol_filepath = r"input/Mag_Col_random65600_x_y_z-cube.dat"
Log_file = 'Log'
# TODO ******************** Parameters ******************** #
# %% # ! NN Hyper-Parametersn
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
embedType = 'rff'
numFreqs = 32       # rff: number of frequency components per coordinate
freqScale = 4     # rff: sigma = 4~10 commonly used
sirenOmega0 = 30.0  # SIREN omega0 (recommended by Sitzmann et al., 2020)

# TODO ********************** Data Loading  ************************* #
######## Load <OBS> points ########
data_obs = np.loadtxt(dataObs_filepath,delimiter=',' ,skiprows=1)
xObs = data_obs[:, 0:1]
yObs = data_obs[:, 1:2]
zObs = data_obs[:, 2:3]

BxObs = data_obs[:, 3:4]
ByObs = data_obs[:, 4:5]
BzObs = data_obs[:, 5:6]
numPiont_Obs = xObs.shape[0]

######## Load <Col> points ########
data_col = np.loadtxt(dataCol_filepath, delimiter=',', skiprows=1)
xCol = data_col[:, 0:1]
yCol = data_col[:, 1:2]
zCol = data_col[:, 2:3]

numPiont_Col = xCol.shape[0]
Npoints = numPiont_Col + numPiont_Obs

nBatPerEpo = math.ceil( Npoints / BS )

obs_Data = [np.float32(xObs), np.float32(yObs), np.float32(zObs),
            np.float32(BxObs), np.float32(ByObs), np.float32(BzObs)]

col_Data = [np.float32(xCol), np.float32(yCol), np.float32(zCol)]

######## Reset graph & seeds for reproducibility of each config
tf.compat.v1.reset_default_graph()
np.random.seed(rdSeed)
tf.random.set_seed(rdSeed)

######## Network input dim depends on embedding
if embedType == 'none':
    inDim = 3
elif embedType == 'rff':
    inDim = 2 * numFreqs
else:
    raise ValueError("embedType must be 'none', 'rff'")
layers = [inDim] + [nNeuron] * nLayer + [1]

######## Save dirs for models & logs #########
embedTag = 'noEmb' if embedType == 'none' \
    else f'{embedType}{numFreqs}sig{freqScale}'
dir_Name = (f'NN{nLayer}_{nNeuron}_{actiFun}_{embedTag}_{opti}'
            f'_RADk{k}c{c}n{n0}d{dn}_nEpo{nEpo}_BS{BS}_Seed{rdSeed}')
Model_file = f'NN_Models/NumObs{numPiont_Obs}/{dir_Name}'
fileLossVars_file = f"{Log_file}/NumObs{numPiont_Obs}"
if not os.path.exists(fileLossVars_file):
    os.makedirs(fileLossVars_file)
fileLossVars_path = f"{fileLossVars_file}/{dir_Name}_LossVars.dat"
# TODO ******************** End ******************** #
if __name__ == '__main__':
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
    print(f'BS：{BS}')
    print(f"nBatPerEpo：{nBatPerEpo}")

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
    print('Saving training info to file: %s' % (fileLossVars_path), flush=True)

    ######## Release TF session/graph resources before next config
    model.sess.close()
