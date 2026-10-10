"""
Multi-source magnetic vector data fusion based on Least Squares Collocation (LSC).

Theoretical framework (Moritz, 1980):
    Observation equation: l = A*x + s + n
    where:
        l   : observed magnetic anomaly
        A*x : deterministic trend term (regional background field)
        s   : random signal term (local magnetic anomaly)
        n   : observation noise

    Core LSC inference equations:
        s_pred = C_sn * (C_nn + D)^(-1) * (l - A*x)
        y_pred = A_pred*x + s_pred

    1. Trend removal: a spatial linear polynomial fits the regional background.
    2. Local collocation: moving windows (KDTree) select effective neighborhoods,
       avoiding the memory explosion of a global inversion.
    3. Covariance modeling: the auto-covariance matrix (C_nn) and the
       cross-covariance matrix (C_sn) are computed explicitly.

Coordinate convention: X, Y, Z are in the NED coordinate system (Z positive downward).

References:
    [1] Moritz, H. (1980). Advanced Physical Geodesy. Herbert Wichmann Verlag.
        (https://api.semanticscholar.org/CorpusID:118981648)

    [2] Forsberg, R., & Tscherning, C. C. (1981). The use of height data in gravity field
        approximation by collocation. Journal of Geophysical Research: Solid Earth, 86(B9), 7843-7854.
        DOI: https://doi.org/10.1029/JB086iB09p07843

@Author: LYY
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import cho_factor, cho_solve
import time

class MagneticLocalLSC:
    def __init__(self, n_neighbors=150, noise_std=0.0, cov_type='gaussian', scale_factor=1.2):
        """
        Initialize the Least Squares Collocation (LSC) solver.
        :param n_neighbors: number of observation points in the local collocation window
        :param noise_std: prior noise standard deviation of the observations (matrix D)
        :param cov_type: empirical covariance function ('gaussian', 'exponential', 'cauchy')
        :param scale_factor: scaling factor for the correlation length L
        """
        self.n_neighbors = n_neighbors
        self.noise_std = noise_std
        self.cov_type = cov_type.lower()
        self.scale_factor = scale_factor

        self.tree = None
        self.obs_coords = None
        self.residuals = None
        self.trend_coefs = None

    def _covariance_function(self, dist, L):
        """Empirical covariance function"""
        if self.cov_type == 'gaussian':
            return np.exp(-(dist ** 2) / (L ** 2))
        elif self.cov_type == 'exponential':
            return np.exp(-dist / L)
        elif self.cov_type == 'cauchy':
            return 1.0 / (1.0 + (dist / L) ** 2)
        else:
            raise ValueError(f"Unknown covariance model: {self.cov_type}")

    def fit(self, coords_obs, data_obs):
        """Trend removal and spatial index construction"""
        self.obs_coords = coords_obs

        # Build the design matrix A
        A = np.column_stack([np.ones(len(coords_obs)), coords_obs])

        # Solve the trend coefficients x by least squares
        self.trend_coefs, _, _, _ = np.linalg.lstsq(A, data_obs, rcond=None)

        # Compute the random-signal observations (l - A*x)
        self.residuals = data_obs - A @ self.trend_coefs

        # Build a KDTree spatial index to support local moving windows
        self.tree = cKDTree(self.obs_coords)

    def predict(self, coords_pred):
        """Execute the core LSC inference equation"""
        n_pred = len(coords_pred)
        pred_interp = np.zeros(n_pred)

        # Compute the deterministic trend at the prediction points (A_pred * x)
        A_pred = np.column_stack([np.ones(n_pred), coords_pred])
        trend_pred = A_pred @ self.trend_coefs

        # Perform local collocation point by point
        for i in range(n_pred):
            pt = coords_pred[i]

            # Select the local collocation window
            dists, idx = self.tree.query(pt, k=self.n_neighbors)
            local_coords = self.obs_coords[idx]
            l_res = self.residuals[idx]

            # Estimate the local stationary correlation length L
            L_corr = np.mean(dists) * self.scale_factor

            # Build the auto-covariance matrix (C_nn)
            diff = local_coords[:, np.newaxis, :] - local_coords[np.newaxis, :, :]
            dist_nn = np.sqrt(np.sum(diff ** 2, axis=-1))
            C_nn = self._covariance_function(dist_nn, L_corr)

            # Build the noise variance matrix (D = noise_std^2 * I)
            D = (self.noise_std ** 2) * np.eye(self.n_neighbors)

            # Assemble the collocation normal-equation matrix: (C_nn + D)
            M = C_nn + D

            # Build the cross-covariance vector (C_sn)
            dist_sn = np.sqrt(np.sum((pt - local_coords) ** 2, axis=1))
            C_sn = self._covariance_function(dist_sn, L_corr)

            # Solve s_pred = C_sn * (C_nn + D)^(-1) * l_res
            try:
                # Cholesky factorization for efficient inversion of the positive-definite matrix
                c, low = cho_factor(M)
                weights = cho_solve((c, low), l_res)
                s_pred = C_sn @ weights
            except np.linalg.LinAlgError:
                # Damped fallback for ill-conditioned matrices
                M_damped = M + 1e-6 * np.eye(self.n_neighbors)
                s_pred = C_sn @ np.linalg.solve(M_damped, l_res)

            # Final prediction = trend term + signal term
            pred_interp[i] = trend_pred[i] + s_pred
            if i % 3000 == 0 or i == n_pred - 1:
                print(f"       -> LSC inference progress: {i + 1}/{n_pred}")

        return pred_interp

if __name__ == "__main__":
    # ==========================================
    # File paths and parameters
    # ==========================================
    flag = 1  # 0: noise-free data, 1: noisy data

    match flag:
        case 0:
            dataObs_filepath = "input/Mag_Air1476_Sea492_x_y_z_Bx_By_Bz.dat"
            out_suffix = ""
            # For noise-free data, keep a tiny noise term so the matrix stays non-singular
            sigma_noise = 1e-4
        case 1:
            dataObs_filepath = "input/Mag_Air1476_Sea492_x_y_z_Bx_By_Bz_snr30.dat"
            out_suffix = "_snr30"
            sigma_noise = 0.1

    data_pred_path = "input/Mag_Test_A3Remaining.dat"

    # Core LSC parameters
    neighbors = 150
    cov_model = 'cauchy'  # ('gaussian', 'exponential', 'cauchy')
    # ==========================================
    #  Load data
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
        print(f"[INFO] Data loaded successfully: {len(xObs)} observation points, {len(xPred)} prediction points")
    except Exception as e:
        raise RuntimeError(f"Failed to read data, please check the path: {e}")

    coords_obs = np.column_stack((xObs, yObs, zObs))
    coords_pred = np.column_stack((xPred, yPred, zPred))

    print(f"\n[INFO] --- Least Squares Collocation (LSC) started ---")
    print(f"[INFO] Local collocation window size: {neighbors} points")
    print(f"[INFO] Empirical covariance model: {cov_model}")
    print(f"[INFO] Prior noise standard deviation: {sigma_noise:.1e}")

    # ==========================================
    #  Process the three components independently
    # ==========================================
    start_time_fit = time.time()

    lsc_model_Bx = MagneticLocalLSC(
        n_neighbors=neighbors,
        noise_std=sigma_noise,
        cov_type=cov_model
    )
    lsc_model_By = MagneticLocalLSC(
        n_neighbors=neighbors,
        noise_std=sigma_noise,
        cov_type=cov_model
    )
    lsc_model_Bz = MagneticLocalLSC(
        n_neighbors=neighbors,
        noise_std=sigma_noise,
        cov_type=cov_model
    )

    print("\n--- Building the Bx component LSC normal equations ---")
    lsc_model_Bx.fit(coords_obs, Bx_obs)

    print("\n--- Building the By component LSC normal equations ---")
    lsc_model_By.fit(coords_obs, By_obs)

    print("\n--- Building the Bz component LSC normal equations ---")
    lsc_model_Bz.fit(coords_obs, Bz_obs)

    print(f"\n[INFO]  LSC inference finished, time: {time.time() - start_time_fit:.3f} s")
    print(f"\n[INFO]  LSC fusion started")
    start_time_pred = time.time()
    Bx_pred = lsc_model_Bx.predict(coords_pred)
    By_pred = lsc_model_By.predict(coords_pred)
    Bz_pred = lsc_model_Bz.predict(coords_pred)
    print(f"\n[INFO]  LSC fusion finished, time: {time.time() - start_time_pred:.3f} s")
    # ==========================================
    # Export results
    # ==========================================
    result = np.column_stack([xPred, yPred, zPred, Bx_pred, By_pred, Bz_pred])
    out_name = f'input/Mag_test_LSC_{len(xObs)}T{len(xPred)}_{cov_model}{out_suffix}.dat'

    np.savetxt(
        out_name, result, delimiter=',',
        header='x,y,z,Bx,By,Bz', comments='', fmt='%.6f'
    )
    print(f"[INFO] Done, LSC results exported to: {out_name}")