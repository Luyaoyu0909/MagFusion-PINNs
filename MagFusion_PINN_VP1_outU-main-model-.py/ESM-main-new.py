"""
Multi-source magnetic vector data fusion based on a dual-layer equivalent source.

Algorithm principle:
    A cascade-fitting strategy balances the spectra of multi-source data:
    1. Deep-layer fitting: large depth and strong damping stably extract the
       low-frequency, long-wavelength background field.
    2. Shallow-layer fitting: small depth and weak damping fit the deep-layer
       residual, restoring high-frequency, short-wavelength details.

Coordinate convention: X, Y, Z are in the NED coordinate system (Z positive downward).

References:
    [1] Uppal, I., Uieda, L., Oliveira Jr., V. C., & Holme, R. (2025). Dual-layer gradient-boosted
        equivalent sources for magnetic data. Geophysical Journal International, 243(3), ggaf359.
        DOI: https://doi.org/10.1093/gji/ggaf359
    [2] Li, D., Guo, Z., Du, J., & Chen, C. (2021). Stable downward continuation of the total-field
        magnetic anomaly derived by using equivalent sources. Journal of Applied Geophysics, 193, 104429.
        DOI: https://doi.org/10.1016/j.jappgeo.2021.104429
    [3] Dampney, C. N. G. (1969). The equivalent source technique. Geophysics, 34(1), 39-53.
        DOI:https://doi.org/10.1190/1.1439996
    [4] Uieda, L., Soler, S. R., Pesce, A., et al. (2020). Harmonica: Forward modeling, inversion,
        and processing gravity and magnetic data. Fatiando a Terra Project
        (https://www.fatiando.org/harmonica).
        (This code is built on the EquivalentSources of this library, which uses scalar harmonic
        spline interpolation underneath, ensuring that each component strictly satisfies Laplace's
        equation in the source-free region and providing excellent numerical stability.)

@Author: LYY
"""
import numpy as np
import harmonica as hm
import time

class DualLayerEquivalentSource:
    def __init__(self, deep_depth, shallow_depth, deep_damping=1.0, shallow_damping=1e-4):
        """
        Initialize the dual-layer equivalent-source model.
        :param deep_depth: relative depth of the deep equivalent sources (extracts low frequencies)
        :param shallow_depth: relative depth of the shallow equivalent sources (fits high-frequency residuals)
        """
        self.deep_depth = deep_depth
        self.shallow_depth = shallow_depth

        # Deep model: strong damping, extremely smooth
        self.es_deep = hm.EquivalentSources(depth=deep_depth, damping=deep_damping)
        # Shallow model: weak damping, captures details
        self.es_shallow = hm.EquivalentSources(depth=shallow_depth, damping=shallow_damping)

    def fit(self, coords_obs, data_obs):
        """Cascade fitting: fit the deep background first, then the shallow residual"""
        print(f"[INFO] Starting deep equivalent-source fitting (Depth={self.deep_depth}m)...")
        self.es_deep.fit(coords_obs, data_obs)

        # Predict the deep model at the observation points
        pred_deep = self.es_deep.predict(coords_obs)

        # Compute the data residual (i.e., the high-frequency anomaly)
        residual = data_obs - pred_deep

        print(f"[INFO] Starting shallow equivalent-source residual fitting (Depth={self.shallow_depth}m)...")
        self.es_shallow.fit(coords_obs, residual)
        print("[INFO] Dual-layer cascade fitting finished.")

    def predict(self, coords_pred):
        """Superposition of the two-layer predictions"""
        pred_deep = self.es_deep.predict(coords_pred)
        pred_shallow = self.es_shallow.predict(coords_pred)
        return pred_deep + pred_shallow

if __name__ == "__main__":
    # ==========================================
    # File paths and parameters
    # ==========================================
    flag = 1  # 0: noise-free data, 1: noisy data
    match flag:
        case 0:
            dataObs_filepath = "input/Mag_Air1476_Sea492_x_y_z_Bx_By_Bz.dat"
            out_suffix = ""
        case 1:
            dataObs_filepath = "input/Mag_Air1476_Sea492_x_y_z_Bx_By_Bz_snr30.dat"
            out_suffix = "_snr30"

    data_pred_path = "input/Mag_Test_A3Remaining.dat"

    # ==========================================
    # Load data and convert coordinates
    # ==========================================
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

    # Z is positive downward; convert to elevation (positive upward) for the core algorithm
    zObs_up = -zObs
    zPred_up = -zPred

    coords_obs_up = (xObs, yObs, zObs_up)
    coords_pred_up = (xPred, yPred, zPred_up)

    # Z is positive downward; convert to elevation (positive upward) for the core algorithm
    zObs_up = -zObs
    zPred_up = -zPred

    coords_obs_up = (xObs, yObs, zObs_up)
    coords_pred_up = (xPred, yPred, zPred_up)

    # ==========================================
    # Parameter configuration
    # ==========================================
    delta_h = np.max(zObs) - np.min(zObs)

    # Depth strategy: the deep layer is typically 3-5x the maximum height range, the shallow layer 0.5-1x
    depth_deep = delta_h * 4.0
    depth_shallow = delta_h * 0.8

    print(f"\n[INFO] --- Dual-layer equivalent-source architecture started ---")
    print(f"[INFO] Observation height range: {delta_h:.2f} m")
    print(f"[INFO] Deep equivalent sources (low-frequency background): {depth_deep:.2f} m")
    print(f"[INFO] Shallow equivalent sources (high-frequency residual): {depth_shallow:.2f} m")

    # ==========================================
    # Instantiate the three-component dual-layer models and fit
    # ==========================================
    start_time_fit = time.time()
    dles_Bx = DualLayerEquivalentSource(depth_deep, depth_shallow, deep_damping=1.0, shallow_damping=1e-4)
    dles_By = DualLayerEquivalentSource(depth_deep, depth_shallow, deep_damping=1.0, shallow_damping=1e-4)
    dles_Bz = DualLayerEquivalentSource(depth_deep, depth_shallow, deep_damping=1.0, shallow_damping=1e-4)

    print("\n--- Fitting the Bx component ---")
    dles_Bx.fit(coords_obs_up, Bx_obs)
    print("\n--- Fitting the By component ---")
    dles_By.fit(coords_obs_up, By_obs)
    print("\n--- Fitting the Bz component ---")
    dles_Bz.fit(coords_obs_up, Bz_obs)
    print(f"[INFO] Dual-layer equivalent-source models fitted, time: {time.time() - start_time_fit:.3f} s")
    # ==========================================
    # Full-domain field prediction and export
    # ==========================================
    print("\n[INFO] Performing full-domain fused-field prediction...")
    start_time_pred = time.time()
    Bx_pred = dles_Bx.predict(coords_pred_up)
    By_pred = dles_By.predict(coords_pred_up)
    Bz_pred = dles_Bz.predict(coords_pred_up)
    print(f"[INFO] Full-domain fused-field prediction finished, time: {time.time() - start_time_pred:.3f} s")

    result = np.column_stack([xPred, yPred, zPred, Bx_pred, By_pred, Bz_pred])
    out_name = f'input/Mag_test_DualLayerESM_{len(xObs)}T{len(xPred)}{out_suffix}.dat'

    np.savetxt(
        out_name, result, delimiter=',',
        header='x,y,z,Bx,By,Bz', comments='', fmt='%.6f'
    )
    print(f"[INFO] Done, dual-layer fusion results exported to: {out_name}")
