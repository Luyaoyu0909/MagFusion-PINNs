"""
Multi-source magnetic vector data fusion based on Radial Basis Functions (RBF).

Coordinate convention: X, Y, Z are in the NED coordinate system (Z positive downward).

References:
    [1] Hardy, R. L. (1971). Multiquadric equations of topography and other irregular surfaces.
        Journal of Geophysical Research, 76(8), 1905-1915.
        DOI: https://doi.org/10.1029/JB076i008p01905
    [2] SciPy Community. RBFInterpolator — SciPy v1.18.0 Manual.
        (This code is built on the RBFInterpolator of this library,
        https://docs.scipy.org.cn/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)

@Author: LYY
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
import time

class MagneticRBFInterpolator:
    def __init__(self, kernel='thin_plate_spline', smoothing=0.0):
        """
        Initialize the magnetic-field RBF interpolator.
        :param kernel: kernel type ('thin_plate_spline', 'multiquadric', 'linear', 'gaussian', etc.)
        :param smoothing: smoothing parameter
        """
        self.kernel = kernel
        self.smoothing = smoothing
        self.rbf_model = None

    def fit(self, coords_obs, data_obs):
        self.rbf_model = RBFInterpolator(
            coords_obs, 
            data_obs, 
            kernel=self.kernel, 
            smoothing=self.smoothing
        )

    def predict(self, coords_pred):
        """Perform the interpolation."""
        if self.rbf_model is None:
            raise ValueError("RBF model has not been fitted yet; call fit() first.")
        return self.rbf_model(coords_pred)

if __name__ == "__main__":
    # ==========================================
    # File paths and parameters
    # ==========================================
    flag = 1  # 0: noise-free data, 1: noisy data
    
    match flag:
        case 0:
            dataObs_filepath = "input/Mag_Air1476_Sea492_x_y_z_Bx_By_Bz.dat"
            out_suffix = ""
            smoothing_param = 0 # noise-free data
        case 1:
            dataObs_filepath = "input/Mag_Air1476_Sea492_x_y_z_Bx_By_Bz_snr30.dat"
            out_suffix = "_snr30"
            smoothing_param = 1e-2  # noisy data

    data_pred_path = "input/Mag_Test_A3Remaining.dat"
    kernel_type = 'thin_plate_spline'  # ('thin_plate_spline', 'multiquadric', 'linear', 'gaussian', etc.)

    # ==========================================
    # Load data and assemble coordinates
    # ==========================================
    try:
        obs = np.loadtxt(dataObs_filepath, delimiter=',', skiprows=1)
        xObs = obs[:, 0]
        yObs = obs[:, 1]
        zObs = obs[:, 2]
        Bx_obs = obs[:, 3]
        By_obs = obs[:, 4]
        Bz_obs = obs[:, 5]

        pred = np.loadtxt(data_pred_path, delimiter=',', skiprows=1)
        xPred = pred[:, 0]
        yPred = pred[:, 1]
        zPred = pred[:, 2]
        
        print(f"[INFO] Data loaded successfully: {len(xObs)} interpolation input points, {len(xPred)} target prediction points")
    except Exception as e:
        raise RuntimeError(f"Failed to read data, please check the file path: {e}")

    # RBFInterpolator requires coordinate matrices of shape (N, 3)
    coords_obs = np.column_stack((xObs, yObs, zObs))
    coords_pred = np.column_stack((xPred, yPred, zPred))

    print(f"\n[INFO] --- RBF mathematical interpolation baseline started ---")
    print(f"[INFO] Kernel: {kernel_type}")
    print(f"[INFO] Smoothing: {smoothing_param:.1e}")

    # ==========================================
    #  Instantiate the three-component RBF models and fit
    # ==========================================
    start_time_fit = time.time()
    
    rbf_Bx = MagneticRBFInterpolator(kernel=kernel_type, smoothing=smoothing_param)
    rbf_By = MagneticRBFInterpolator(kernel=kernel_type, smoothing=smoothing_param)
    rbf_Bz = MagneticRBFInterpolator(kernel=kernel_type, smoothing=smoothing_param)

    print("\n--- Building the Bx interpolation surface ---")
    rbf_Bx.fit(coords_obs, Bx_obs)
    
    print("--- Building the By interpolation surface ---")
    rbf_By.fit(coords_obs, By_obs)
    
    print("--- Building the Bz interpolation surface ---")
    rbf_Bz.fit(coords_obs, Bz_obs)
    
    print(f"\n[INFO] RBF models built, total time: {time.time() - start_time_fit:.3f} s")

    # ==========================================
    #  Full-domain field prediction and export
    # ==========================================
    print("\n[INFO] Performing full-domain interpolation prediction...")
    start_time_pred = time.time()
    
    Bx_pred = rbf_Bx.predict(coords_pred)
    By_pred = rbf_By.predict(coords_pred)
    Bz_pred = rbf_Bz.predict(coords_pred)
    
    print(f"[INFO] Full-domain interpolation prediction finished, total time: {time.time() - start_time_pred:.3f} s")

    # Assemble and save the results
    result = np.column_stack([xPred, yPred, zPred, Bx_pred, By_pred, Bz_pred])
    out_name = f'input/Mag_test_RBF_{len(xObs)}T{len(xPred)}{out_suffix}.dat'

    np.savetxt(
        out_name, result, delimiter=',',
        header='x,y,z,Bx,By,Bz', comments='', fmt='%.6f'
    )
    print(f"[INFO] Done, RBF interpolation results exported to: {out_name}")