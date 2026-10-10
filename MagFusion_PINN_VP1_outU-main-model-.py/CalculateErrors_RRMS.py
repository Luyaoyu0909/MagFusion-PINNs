import numpy as np
import MagFusion_PINN_main_v2 as main
import Predict_Mag_v2 as pred

# 0:No Noise,1:With Noise
flag = 1

# Load the Reference Values
Test_data_ref = np.loadtxt('input/Mag_Test_A3Remaining.dat', delimiter=',', skiprows=1)
Bx_test_ref = Test_data_ref[:, 3]
By_test_ref = Test_data_ref[:, 4]
Bz_test_ref = Test_data_ref[:, 5]

match flag:
    case 0:
          # Load the Pinn Calculate Values
          Test_data_Pinn = np.loadtxt(pred.PredMagDataSaveTest_path, delimiter=',', skiprows=1)
          Bx_test_Pinn = Test_data_Pinn[:, 3]
          By_test_Pinn = Test_data_Pinn[:, 4]
          Bz_test_Pinn = Test_data_Pinn[:, 5]

          # Load the LSC Calculate Values
          Test_data_LSC = np.loadtxt('input/Mag_test_LSC_1968T6069_cauchy.dat', delimiter=',', skiprows=1)
          Bx_test_LSC = Test_data_LSC[:, 3]
          By_test_LSC = Test_data_LSC[:, 4]
          Bz_test_LSC = Test_data_LSC[:, 5]

          # Load the RBF Calculate Values
          Test_data_RBF = np.loadtxt('input/Mag_test_RBF_1968T6069.dat', delimiter=',', skiprows=1)
          Bx_test_RBF = Test_data_RBF[:, 3]
          By_test_RBF = Test_data_RBF[:, 4]
          Bz_test_RBF = Test_data_RBF[:, 5]

          # Load the ESM Calculate Values
          Test_data_ESM = np.loadtxt('input/Mag_test_DualLayerESM_1968T6069.dat', delimiter=',', skiprows=1)
          Bx_test_ESM = Test_data_ESM[:, 3]
          By_test_ESM = Test_data_ESM[:, 4]
          Bz_test_ESM = Test_data_ESM[:, 5]

          # Load the SDI Calculate Values
          Test_data_SDI = np.loadtxt('input/Mag_test_SDI_1968T6069.dat', delimiter=None, skiprows=0)
          Bx_test_SDI = Test_data_SDI[:, 3]
          By_test_SDI = Test_data_SDI[:, 4]
          Bz_test_SDI = Test_data_SDI[:, 5]

    case 1:
          # Load the Pinn Calculate Values
          Test_data_Pinn = np.loadtxt(pred.PredMagDataSaveTest_path, delimiter=',', skiprows=1)
          Bx_test_Pinn = Test_data_Pinn[:, 3]
          By_test_Pinn = Test_data_Pinn[:, 4]
          Bz_test_Pinn = Test_data_Pinn[:, 5]

          # Load the LSC Calculate Values
          Test_data_LSC = np.loadtxt('input/Mag_test_LSC_1968T6069_cauchy_snr30.dat', delimiter=',', skiprows=1)
          Bx_test_LSC = Test_data_LSC[:, 3]
          By_test_LSC = Test_data_LSC[:, 4]
          Bz_test_LSC = Test_data_LSC[:, 5]

          # Load the RBF Calculate Values
          Test_data_RBF = np.loadtxt('input/Mag_test_RBF_1968T6069_snr30.dat', delimiter=',', skiprows=1)
          Bx_test_RBF = Test_data_RBF[:, 3]
          By_test_RBF = Test_data_RBF[:, 4]
          Bz_test_RBF = Test_data_RBF[:, 5]

          # Load the ESM Calculate Values
          Test_data_ESM = np.loadtxt('input/Mag_test_DualLayerESM_1968T6069_snr30.dat', delimiter=',', skiprows=1)
          Bx_test_ESM = Test_data_ESM[:, 3]
          By_test_ESM = Test_data_ESM[:, 4]
          Bz_test_ESM = Test_data_ESM[:, 5]

          # Load the SDI Calculate Values
          Test_data_SDI = np.loadtxt('input/Mag_test_SDI_1968T6069_snr30.dat', delimiter=None, skiprows=0)
          Bx_test_SDI = Test_data_SDI[:, 3]
          By_test_SDI = Test_data_SDI[:, 4]
          Bz_test_SDI = Test_data_SDI[:, 5]


Bx_test_Pinn_Error = np.sqrt(np.sum((Bx_test_Pinn - Bx_test_ref)**2) / np.sum((Bx_test_ref)**2)) *100
Bx_test_SDI_Error = np.sqrt(np.sum((Bx_test_SDI - Bx_test_ref)**2) / np.sum((Bx_test_ref)**2)) *100
Bx_test_LSC_Error = np.sqrt(np.sum((Bx_test_LSC - Bx_test_ref)**2) / np.sum((Bx_test_ref)**2)) *100
Bx_test_RBF_Error = np.sqrt(np.sum((Bx_test_RBF - Bx_test_ref)**2) / np.sum((Bx_test_ref)**2)) *100
Bx_test_ESM_Error = np.sqrt(np.sum((Bx_test_ESM - Bx_test_ref)**2) / np.sum((Bx_test_ref)**2)) *100

By_test_Pinn_Error = np.sqrt(np.sum((By_test_Pinn - By_test_ref)**2) / np.sum((By_test_ref)**2)) *100
By_test_SDI_Error = np.sqrt(np.sum((By_test_SDI - By_test_ref)**2) / np.sum((By_test_ref)**2)) *100
By_test_LSC_Error = np.sqrt(np.sum((By_test_LSC - By_test_ref)**2) / np.sum((By_test_ref)**2)) *100
By_test_RBF_Error = np.sqrt(np.sum((By_test_RBF - By_test_ref)**2) / np.sum((By_test_ref)**2)) *100
By_test_ESM_Error = np.sqrt(np.sum((By_test_ESM - By_test_ref)**2) / np.sum((By_test_ref)**2)) *100

Bz_test_Pinn_Error = np.sqrt(np.sum((Bz_test_Pinn - Bz_test_ref)**2) / np.sum((Bz_test_ref)**2)) *100
Bz_test_SDI_Error = np.sqrt(np.sum((Bz_test_SDI - Bz_test_ref)**2) / np.sum((Bz_test_ref)**2)) *100
Bz_test_LSC_Error = np.sqrt(np.sum((Bz_test_LSC - Bz_test_ref)**2) / np.sum((Bz_test_ref)**2)) *100
Bz_test_RBF_Error = np.sqrt(np.sum((Bz_test_RBF - Bz_test_ref)**2) / np.sum((Bz_test_ref)**2)) *100
Bz_test_ESM_Error = np.sqrt(np.sum((Bz_test_ESM - Bz_test_ref)**2) / np.sum((Bz_test_ref)**2)) *100

print(f'{main.Model_file}\n')
print(f'Bx_test_Pinn_Error_RRMS: {Bx_test_Pinn_Error:.3f} %')
print(f'Bx_test_LSC_Error_RRMS: {Bx_test_LSC_Error:.3f} %')
print(f'Bx_test_RBF_Error_RRMS: {Bx_test_RBF_Error:.3f} %')
print(f'Bx_test_ESM_Error_RRMS: {Bx_test_ESM_Error:.3f} %')
print(f'Bx_test_SDI_Error_RRMS: {Bx_test_SDI_Error:.3f} % \n')

print(f'By_test_Pinn_Error_RRMS: {By_test_Pinn_Error:.3f} %')
print(f'By_test_LSC_Error_RRMS: {By_test_LSC_Error:.3f} %')
print(f'By_test_RBF_Error_RRMS: {By_test_RBF_Error:.3f} %')
print(f'By_test_ESM_Error_RRMS: {By_test_ESM_Error:.3f} %')
print(f'By_test_SDI_Error_RRMS: {By_test_SDI_Error:.3f} %\n')

print(f'Bz_test_Pinn_Error_RRMS: {Bz_test_Pinn_Error:.3f} %')
print(f'Bz_test_LSC_Error_RRMS: {Bz_test_LSC_Error:.3f} %')
print(f'Bz_test_RBF_Error_RRMS: {Bz_test_RBF_Error:.3f} %')
print(f'Bz_test_ESM_Error_RRMS: {Bz_test_ESM_Error:.3f} %')
print(f'Bz_test_SDI_Error_RRMS: {Bz_test_SDI_Error:.3f} %\n')