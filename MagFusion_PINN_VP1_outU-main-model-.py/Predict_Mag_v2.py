import pandas as pd
from MagFusion_PINN_Function_v2 import *
import MagFusion_PINN_main_v2 as main

# TODO ******************** Save File Path ******************** #
PredSaveData_file = 'PredData'
# TODO ********************** Test Data Loading ************************* #
data_pred_path = r"input/Mag_Test_A3Remaining.dat"
data_pred = np.loadtxt(data_pred_path, delimiter=',', skiprows=1)
# TODO ******************** Parameters ******************** #
layers = main.layers
actiFun = main.actiFun
opti = main.opti
DW = main.DW
k = main.k; c = main.c; n0 = main.n0; dn = main.dn
nEpo = main.nEpo
nBatPerEpo = main.nBatPerEpo
BS = main.BS
rdSeed = main.rdSeed
# TODO ********************** end ************************* #
######## Load <Pred> points
xPred = data_pred[:, 0:1]
yPred = data_pred[:, 1:2]
zPred = data_pred[:, 2:3]

######## Reset graph & seeds
tf.compat.v1.reset_default_graph()
np.random.seed(rdSeed)
tf.random.set_seed(rdSeed)

######## Paths
dir_Name = main.dir_Name
Model_file = main.Model_file
PredDataSave_dir = f"{PredSaveData_file}/NumObs{main.numPiont_Obs}"
if not os.path.exists(PredDataSave_dir):
    os.makedirs(PredDataSave_dir)
PredMagDataSaveTest_path = \
    f"{PredDataSave_dir}/PredMag_RemainingTestData{main.numPiont_Obs}T{data_pred.shape[0]}_{dir_Name}.dat"

# TODO ********************** End ************************* #
if __name__ == "__main__":
    ######## Rebuild model & restore checkpoint
    model = GT_Lap3D(layers, actiFun, opti, DW, main.obs_Data, main.col_Data,
                     nBatPerEpo, BS,
                     embedType=main.embedType, numFreqs=main.numFreqs,
                     freqScale=main.freqScale, sirenOmega0=main.sirenOmega0)
    LoadModel_path = f'{Model_file}/model_{nEpo}.ckpt'
    model.saver.restore(model.sess, save_path=LoadModel_path)

    ######## Predict <Pred> points & save
    time_start = time.time()
    Bx_pred_est, By_pred_est, Bz_pred_est = model.predict_MF(xPred, yPred, zPred)
    print(f"Fusion time: {(time.time() - time_start):.3f}s", flush=True)
    df = pd.DataFrame({
        'x': xPred.ravel(),
        'y': yPred.ravel(),
        'z': zPred.ravel(),
        'Bx': Bx_pred_est.ravel(),
        'By': By_pred_est.ravel(),
        'Bz': Bz_pred_est.ravel()
    })
    df.to_csv(PredMagDataSaveTest_path, sep=',', index=False)
    print(f"Prediction results saved to: {PredMagDataSaveTest_path}", flush=True)

    ######## Release TF session/graph resources before next config
    model.sess.close()