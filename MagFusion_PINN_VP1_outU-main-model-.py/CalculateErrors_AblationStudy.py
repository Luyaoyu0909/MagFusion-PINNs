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

          # Load the MLP Calculate Values
          Test_data_MLP = np.loadtxt(
                'input/PredMagMLP_RemainingTestData1968T6069_NN6_40_swish_Adam_RADk1c0n3000d600_nEpo12000_BS3000_Seed12345.dat', delimiter=',', skiprows=1)
          Bx_test_MLP = Test_data_MLP[:, 3]
          By_test_MLP = Test_data_MLP[:, 4]
          Bz_test_MLP = Test_data_MLP[:, 5]

          # Load the Swish_RFF Calculate Values
          Test_data_RFF = np.loadtxt('PredData/NumObs1968/PredMag_RemainingTestData1968T6069_NN6_40_swish_rff32sig4.0_Adam_RADk1c0n3000d600_nEpo12000_BS3000_Seed12345.dat', delimiter=',', skiprows=1)
          Bx_test_RFF = Test_data_RFF[:, 3]
          By_test_RFF = Test_data_RFF[:, 4]
          Bz_test_RFF = Test_data_RFF[:, 5]

          # Load the Siren Calculate Values
          Test_data_Siren = np.loadtxt('PredData/NumObs1968/PredMag_RemainingTestData1968T6069_NN6_40_siren_noEmb_Adam_RADk1c0n3000d600_nEpo12000_BS3000_Seed12345.dat', delimiter=',', skiprows=1)
          Bx_test_Siren = Test_data_Siren[:, 3]
          By_test_Siren = Test_data_Siren[:, 4]
          Bz_test_Siren = Test_data_Siren[:, 5]

    case 1:
          # Load the Pinn Calculate Values
          Test_data_Pinn = np.loadtxt(pred.PredMagDataSaveTest_path, delimiter=',', skiprows=1)
          Bx_test_Pinn = Test_data_Pinn[:, 3]
          By_test_Pinn = Test_data_Pinn[:, 4]
          Bz_test_Pinn = Test_data_Pinn[:, 5]

          # Load the MLP Calculate Values
          Test_data_MLP = np.loadtxt('input/PredMagMLP_RemainingTestData1968T6069_NN6_40_swish_Adam_RADk1c0n3000d600_nEpo12001_BS3000_Seed12345.dat', delimiter=',', skiprows=1)
          Bx_test_MLP = Test_data_MLP[:, 3]
          By_test_MLP = Test_data_MLP[:, 4]
          Bz_test_MLP = Test_data_MLP[:, 5]

          # Load the Swish_RFF Calculate Values
          Test_data_RFF = np.loadtxt('PredData/NumObs1968/PredMag_RemainingTestData1968T6069_NN6_40_swish_rff32sig4.0_Adam_RADk1c0n3000d600_nEpo12001_BS3000_Seed12345.dat', delimiter=',', skiprows=1)
          Bx_test_RFF = Test_data_RFF[:, 3]
          By_test_RFF = Test_data_RFF[:, 4]
          Bz_test_RFF = Test_data_RFF[:, 5]

          # Load the Siren Calculate Values
          Test_data_Siren = np.loadtxt('PredData/NumObs1968/PredMag_RemainingTestData1968T6069_NN6_40_siren_noEmb_Adam_RADk1c0n3000d600_nEpo12001_BS3000_Seed12345.dat', delimiter=',', skiprows=1)
          Bx_test_Siren = Test_data_Siren[:, 3]
          By_test_Siren = Test_data_Siren[:, 4]
          Bz_test_Siren = Test_data_Siren[:, 5]


# ---------------- Bx ----------------
Bx_test_Pinn_Error = Bx_test_Pinn - Bx_test_ref
Bx_test_Pinn_Error_RRMS = np.sqrt(np.sum(Bx_test_Pinn_Error**2) / np.sum(Bx_test_ref**2)) * 100
Bx_test_Pinn_Error_RMSE = np.sqrt(np.mean(Bx_test_Pinn_Error**2))
Bx_test_Pinn_Error_MAE = np.mean(np.abs(Bx_test_Pinn_Error))
Bx_test_Pinn_Error_STD = np.std(Bx_test_Pinn_Error)

Bx_test_MLP_Error = Bx_test_MLP - Bx_test_ref
Bx_test_MLP_Error_RRMS = np.sqrt(np.sum(Bx_test_MLP_Error**2) / np.sum(Bx_test_ref**2)) * 100
Bx_test_MLP_Error_RMSE = np.sqrt(np.mean(Bx_test_MLP_Error**2))
Bx_test_MLP_Error_MAE = np.mean(np.abs(Bx_test_MLP_Error))
Bx_test_MLP_Error_STD = np.std(Bx_test_MLP_Error)

Bx_test_RFF_Error = Bx_test_RFF - Bx_test_ref
Bx_test_RFF_Error_RRMS = np.sqrt(np.sum(Bx_test_RFF_Error**2) / np.sum(Bx_test_ref**2)) * 100
Bx_test_RFF_Error_RMSE = np.sqrt(np.mean(Bx_test_RFF_Error**2))
Bx_test_RFF_Error_MAE = np.mean(np.abs(Bx_test_RFF_Error))
Bx_test_RFF_Error_STD = np.std(Bx_test_RFF_Error)

Bx_test_Siren_Error = Bx_test_Siren - Bx_test_ref
Bx_test_Siren_Error_RRMS = np.sqrt(np.sum(Bx_test_Siren_Error**2) / np.sum(Bx_test_ref**2)) * 100
Bx_test_Siren_Error_RMSE = np.sqrt(np.mean(Bx_test_Siren_Error**2))
Bx_test_Siren_Error_MAE = np.mean(np.abs(Bx_test_Siren_Error))
Bx_test_Siren_Error_STD = np.std(Bx_test_Siren_Error)

# ---------------- By ----------------
By_test_Pinn_Error = By_test_Pinn - By_test_ref
By_test_Pinn_Error_RRMS = np.sqrt(np.sum(By_test_Pinn_Error**2) / np.sum(By_test_ref**2)) * 100
By_test_Pinn_Error_RMSE = np.sqrt(np.mean(By_test_Pinn_Error**2))
By_test_Pinn_Error_MAE = np.mean(np.abs(By_test_Pinn_Error))
By_test_Pinn_Error_STD = np.std(By_test_Pinn_Error)

By_test_MLP_Error = By_test_MLP - By_test_ref
By_test_MLP_Error_RRMS = np.sqrt(np.sum(By_test_MLP_Error**2) / np.sum(By_test_ref**2)) * 100
By_test_MLP_Error_RMSE = np.sqrt(np.mean(By_test_MLP_Error**2))
By_test_MLP_Error_MAE = np.mean(np.abs(By_test_MLP_Error))
By_test_MLP_Error_STD = np.std(By_test_MLP_Error)

By_test_RFF_Error = By_test_RFF - By_test_ref
By_test_RFF_Error_RRMS = np.sqrt(np.sum(By_test_RFF_Error**2) / np.sum(By_test_ref**2)) * 100
By_test_RFF_Error_RMSE = np.sqrt(np.mean(By_test_RFF_Error**2))
By_test_RFF_Error_MAE = np.mean(np.abs(By_test_RFF_Error))
By_test_RFF_Error_STD = np.std(By_test_RFF_Error)

By_test_Siren_Error = By_test_Siren - By_test_ref
By_test_Siren_Error_RRMS = np.sqrt(np.sum(By_test_Siren_Error**2) / np.sum(By_test_ref**2)) * 100
By_test_Siren_Error_RMSE = np.sqrt(np.mean(By_test_Siren_Error**2))
By_test_Siren_Error_MAE = np.mean(np.abs(By_test_Siren_Error))
By_test_Siren_Error_STD = np.std(By_test_Siren_Error)

# ---------------- Bz ----------------
Bz_test_Pinn_Error = Bz_test_Pinn - Bz_test_ref
Bz_test_Pinn_Error_RRMS = np.sqrt(np.sum(Bz_test_Pinn_Error**2) / np.sum(Bz_test_ref**2)) * 100
Bz_test_Pinn_Error_RMSE = np.sqrt(np.mean(Bz_test_Pinn_Error**2))
Bz_test_Pinn_Error_MAE = np.mean(np.abs(Bz_test_Pinn_Error))
Bz_test_Pinn_Error_STD = np.std(Bz_test_Pinn_Error)

Bz_test_MLP_Error = Bz_test_MLP - Bz_test_ref
Bz_test_MLP_Error_RRMS = np.sqrt(np.sum(Bz_test_MLP_Error**2) / np.sum(Bz_test_ref**2)) * 100
Bz_test_MLP_Error_RMSE = np.sqrt(np.mean(Bz_test_MLP_Error**2))
Bz_test_MLP_Error_MAE = np.mean(np.abs(Bz_test_MLP_Error))
Bz_test_MLP_Error_STD = np.std(Bz_test_MLP_Error)

Bz_test_RFF_Error = Bz_test_RFF - Bz_test_ref
Bz_test_RFF_Error_RRMS = np.sqrt(np.sum(Bz_test_RFF_Error**2) / np.sum(Bz_test_ref**2)) * 100
Bz_test_RFF_Error_RMSE = np.sqrt(np.mean(Bz_test_RFF_Error**2))
Bz_test_RFF_Error_MAE = np.mean(np.abs(Bz_test_RFF_Error))
Bz_test_RFF_Error_STD = np.std(Bz_test_RFF_Error)

Bz_test_Siren_Error = Bz_test_Siren - Bz_test_ref
Bz_test_Siren_Error_RRMS = np.sqrt(np.sum(Bz_test_Siren_Error**2) / np.sum(Bz_test_ref**2)) * 100
Bz_test_Siren_Error_RMSE = np.sqrt(np.mean(Bz_test_Siren_Error**2))
Bz_test_Siren_Error_MAE = np.mean(np.abs(Bz_test_Siren_Error))
Bz_test_Siren_Error_STD = np.std(Bz_test_Siren_Error)

# ---------------- Print Results ----------------
print(f'{main.Model_file}\n')

print('========== Pinn ==========')
print(f'Bx_test_Pinn_Error_RRMS: {Bx_test_Pinn_Error_RRMS:.3f} %')
print(f'Bx_test_Pinn_Error_RMSE: {Bx_test_Pinn_Error_RMSE:.3f}')
print(f'Bx_test_Pinn_Error_MAE: {Bx_test_Pinn_Error_MAE:.3f}')
print(f'Bx_test_Pinn_Error_STD: {Bx_test_Pinn_Error_STD:.3f}\n')

print(f'By_test_Pinn_Error_RRMS: {By_test_Pinn_Error_RRMS:.3f} %')
print(f'By_test_Pinn_Error_RMSE: {By_test_Pinn_Error_RMSE:.3f}')
print(f'By_test_Pinn_Error_MAE: {By_test_Pinn_Error_MAE:.3f}')
print(f'By_test_Pinn_Error_STD: {By_test_Pinn_Error_STD:.3f}\n')

print(f'Bz_test_Pinn_Error_RRMS: {Bz_test_Pinn_Error_RRMS:.3f} %')
print(f'Bz_test_Pinn_Error_RMSE: {Bz_test_Pinn_Error_RMSE:.3f}')
print(f'Bz_test_Pinn_Error_MAE: {Bz_test_Pinn_Error_MAE:.3f}')
print(f'Bz_test_Pinn_Error_STD: {Bz_test_Pinn_Error_STD:.3f}\n')

print('========== MLP ==========')
print(f'Bx_test_MLP_Error_RRMS: {Bx_test_MLP_Error_RRMS:.3f} %')
print(f'Bx_test_MLP_Error_RMSE: {Bx_test_MLP_Error_RMSE:.3f}')
print(f'Bx_test_MLP_Error_MAE: {Bx_test_MLP_Error_MAE:.3f}')
print(f'Bx_test_MLP_Error_STD: {Bx_test_MLP_Error_STD:.3f}\n')

print(f'By_test_MLP_Error_RRMS: {By_test_MLP_Error_RRMS:.3f} %')
print(f'By_test_MLP_Error_RMSE: {By_test_MLP_Error_RMSE:.3f}')
print(f'By_test_MLP_Error_MAE: {By_test_MLP_Error_MAE:.3f}')
print(f'By_test_MLP_Error_STD: {By_test_MLP_Error_STD:.3f}\n')

print(f'Bz_test_MLP_Error_RRMS: {Bz_test_MLP_Error_RRMS:.3f} %')
print(f'Bz_test_MLP_Error_RMSE: {Bz_test_MLP_Error_RMSE:.3f}')
print(f'Bz_test_MLP_Error_MAE: {Bz_test_MLP_Error_MAE:.3f}')
print(f'Bz_test_MLP_Error_STD: {Bz_test_MLP_Error_STD:.3f}\n')

print('========== Swish_RFF ==========')
print(f'Bx_test_RFF_Error_RRMS: {Bx_test_RFF_Error_RRMS:.3f} %')
print(f'Bx_test_RFF_Error_RMSE: {Bx_test_RFF_Error_RMSE:.3f}')
print(f'Bx_test_RFF_Error_MAE: {Bx_test_RFF_Error_MAE:.3f}')
print(f'Bx_test_RFF_Error_STD: {Bx_test_RFF_Error_STD:.3f}\n')

print(f'By_test_RFF_Error_RRMS: {By_test_RFF_Error_RRMS:.3f} %')
print(f'By_test_RFF_Error_RMSE: {By_test_RFF_Error_RMSE:.3f}')
print(f'By_test_RFF_Error_MAE: {By_test_RFF_Error_MAE:.3f}')
print(f'By_test_RFF_Error_STD: {By_test_RFF_Error_STD:.3f}\n')

print(f'Bz_test_RFF_Error_RRMS: {Bz_test_RFF_Error_RRMS:.3f} %')
print(f'Bz_test_RFF_Error_RMSE: {Bz_test_RFF_Error_RMSE:.3f}')
print(f'Bz_test_RFF_Error_MAE: {Bz_test_RFF_Error_MAE:.3f}')
print(f'Bz_test_RFF_Error_STD: {Bz_test_RFF_Error_STD:.3f}\n')

print('========== Siren ==========')
print(f'Bx_test_Siren_Error_RRMS: {Bx_test_Siren_Error_RRMS:.3f} %')
print(f'Bx_test_Siren_Error_RMSE: {Bx_test_Siren_Error_RMSE:.3f}')
print(f'Bx_test_Siren_Error_MAE: {Bx_test_Siren_Error_MAE:.3f}')
print(f'Bx_test_Siren_Error_STD: {Bx_test_Siren_Error_STD:.3f}\n')

print(f'By_test_Siren_Error_RRMS: {By_test_Siren_Error_RRMS:.3f} %')
print(f'By_test_Siren_Error_RMSE: {By_test_Siren_Error_RMSE:.3f}')
print(f'By_test_Siren_Error_MAE: {By_test_Siren_Error_MAE:.3f}')
print(f'By_test_Siren_Error_STD: {By_test_Siren_Error_STD:.3f}\n')

print(f'Bz_test_Siren_Error_RRMS: {Bz_test_Siren_Error_RRMS:.3f} %')
print(f'Bz_test_Siren_Error_RMSE: {Bz_test_Siren_Error_RMSE:.3f}')
print(f'Bz_test_Siren_Error_MAE: {Bz_test_Siren_Error_MAE:.3f}')
print(f'Bz_test_Siren_Error_STD: {Bz_test_Siren_Error_STD:.3f}\n')
