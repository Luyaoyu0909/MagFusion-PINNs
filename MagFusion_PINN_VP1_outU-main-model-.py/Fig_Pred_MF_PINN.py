'''
@Author  ：LYY
'''
import numpy as np
from matplotlib import pyplot as plt
import MagFusion_PINN_main_v2 as main
import Predict_Mag_v2 as pred

# TODO **********************Load Pred GM DATA************************* #
data_pred_ref = np.loadtxt(pred.data_pred_path,delimiter=',',skiprows=1)
x_pred_ref = data_pred_ref[:, 0]
y_pred_ref = data_pred_ref[:, 1]
z_pred_ref = data_pred_ref[:, 2]
Bx_pred_ref = data_pred_ref[:, 3]
By_pred_ref = data_pred_ref[:, 4]
Bz_pred_ref = data_pred_ref[:, 5]

#load the Test estimate data
data_pred_est = np.loadtxt(pred.PredMagDataSaveTest_path, delimiter=',', skiprows=1)

x_pred_est = data_pred_est[:, 0]
y_pred_est = data_pred_est[:, 1]
z_pred_est = data_pred_est[:, 2]
Bx_pred_est = data_pred_est[:, 3]
By_pred_est = data_pred_est[:, 4]
Bz_pred_est = data_pred_est[:, 5]

# TODO **********************Figure Test ************************* #
x_pred_unique = np.unique(x_pred_est)
y_pred_unique = np.unique(y_pred_est)

Nxpred = len(x_pred_unique)
Nypred = len(y_pred_unique)

Bx_pred_est_matrix = Bx_pred_est.reshape(Nxpred, Nypred)
Bx_pred_ref_matrix = Bx_pred_ref.reshape(Nxpred, Nypred)
Bx_Errors = np.abs(Bx_pred_est_matrix - Bx_pred_ref_matrix)

By_pred_est_matrix = By_pred_est.reshape(Nxpred, Nypred)
By_pred_ref_matrix = By_pred_ref.reshape(Nxpred, Nypred)
By_Errors = np.abs(By_pred_est_matrix - By_pred_ref_matrix)

Bz_pred_est_matrix = Bz_pred_est.reshape(Nxpred, Nypred)
Bz_pred_ref_matrix = Bz_pred_ref.reshape(Nxpred, Nypred)
Bz_Errors = np.abs(Bz_pred_est_matrix - Bz_pred_ref_matrix)

plt.figure(figsize=(16, 8))
plt.subplot(131)
plt.contourf(x_pred_unique, y_pred_unique, Bx_pred_est_matrix.T, 50, cmap='jet')
plt.title('Estimated Bx Component from Test Data By PINN (nT)')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.colorbar()
plt.subplot(132)
plt.contourf(x_pred_unique, y_pred_unique, Bx_pred_ref_matrix.T, 50, cmap='jet')
plt.title('Reference Bx Component from Test Data (nT)')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.colorbar()
plt.subplot(133)
plt.contourf(x_pred_unique, y_pred_unique, Bx_Errors.T, 50, cmap='jet')
plt.title('Absolute Errors For Bx (nT) By PINN')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.colorbar()
plt.show()

plt.figure(figsize=(16, 8))
plt.subplot(131)
plt.contourf(x_pred_unique, y_pred_unique, By_pred_est_matrix.T, 50, cmap='jet')
plt.title('Estimated By Component from Test Data By PINN (nT)')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.colorbar()
plt.subplot(132)
plt.contourf(x_pred_unique, y_pred_unique, By_pred_ref_matrix.T, 50, cmap='jet')
plt.title('Reference By Component from Test Data (nT)')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.colorbar()
plt.subplot(133)
plt.contourf(x_pred_unique, y_pred_unique, By_Errors.T, 50, cmap='jet')
plt.title('Absolute Errors For Bx (nT) By PINN')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.colorbar()
plt.show()

plt.figure(figsize=(16, 8))
plt.subplot(131)
plt.contourf(x_pred_unique, y_pred_unique, Bz_pred_est_matrix.T, 50, cmap='jet')
plt.title('Estimated Bz Component from Test Data By PINN (nT)')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.colorbar()
plt.subplot(132)
plt.contourf(x_pred_unique, y_pred_unique, Bz_pred_ref_matrix.T, 50, cmap='jet')
plt.title('Reference Bz Component from Test Data (nT)')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.colorbar()
plt.subplot(133)
plt.contourf(x_pred_unique, y_pred_unique, Bz_Errors.T, 50, cmap='jet')
plt.title('Absolute Errors For Bz (nT) By PINN')
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.colorbar()
plt.show()
