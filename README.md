# MagFusion-PINNs

# Data and Code to reproduce the results in the manuscript:

**Multi-source Marine Magnetic Vector Data Fusion Method Based on Physics-Informed Neural Networks**

Yaoyu Lu¹, Longwei Chen²\*, Leyuan Wu³\*, Chieh-Hung Chen⁴, Philip W. Livermore⁵, Sjoerd de Ridder⁵, Thomas D. Sandnes⁵
---
This project fuses airborne and shipborne (marine) magnetic vector observations into a  consistent 3D magnetic vector field by learning the magnetic scalar potential **U** with a  PINNs: the network output is the potential, and the magnetic field components are obtained  as its spatial gradients, **B = ∇U**, so that the predicted field is constrained to be  curl-free and to satisfy Laplace's equation in the source-free region.

Classical interpolation/fusion baselines are provided for comparison:  **RBF**, **Least Squares Collocation (LSC)**, **Dual-layer Equivalent Sources (ESM)** and  **scatteredInterpolant (SDI)**.

## Method Overview

- **Network**: MLP with configurable activation (`tanh`, `relu`, `sigmoid`, `gelu`,  `swish`, `siren`) and optional coordinate embedding (`embedType`):
  - `none` — standard MLP baseline
  - `rff` — Random Fourier Features (Tancik et al., 2020)
- **Physics constraints**:
  - Observation loss: MSE between predicted and observed Bx, By, Bz
  - PDE loss: Laplacian of the potential = 0 at collocation points (source-free region)
  - The field is recovered as B = ∇U via automatic differentiation
- **Training strategies**:
  - Dynamic loss weighting (adaptive gradient-balanced weights, Wang et al. style)
  - Residual-based adaptive distribution (RAD) resampling of observation/collocation points
  - Exponential-decay learning rate, Adam optimizer

## Requirements

**Python** (3.9+ recommended)

```
tensorflow >= 2.x
numpy
scipy
pandas
matplotlib
harmonica
```

**MATLAB** (R2021a or later, for the figure scripts: uses `readmatrix` and `tiledlayout`)

## Usage

```bash
# 1. Train the PINN (edit data paths / hyper-parameters inside)
python MagFusion_PINN_main_v2.py          # or: python runCycle.py (batch mode)

# 2. Predict the magnetic field at target points
python Predict_Mag_v2.py

# 3. Run the classical baselines
python RBF-main-new.py
python LSC-main-new.py
python ESM-main-new.py                    # (MATLAB) run SDI.m

# 4. Evaluate
python CalculateErrors_RMSE.py            # also: MAE / STD / RRMS / AblationStudy

# 5. Figures
python Fig_Loss_plot_together_v2.py       # training curves
python Fig_Pred_MF_PINN.py                # prediction maps
(MATLAB) FigConstratMF.m, FigAirSeaMF.m, ...
```

## Data Format

All `.dat` files are comma-separated with a header line:

```
x,y,z,Bx,By,Bz
```

Coordinates follow the **NED convention (Z positive downward)**, units: m for coordinates,  
nT for the magnetic field components.

### Data Roles

| File                                                   | Role                   | Columns used         |
| ------------------------------------------------------ | ---------------------- | -------------------- |
| `Mag_Air1476_Sea492_x_y_z_Bx_By_Bz.dat` (and `_snr30`) | **Training set**       | x, y, z, Bx, By, Bz  |
| `Mag_Col_random65600_x_y_z-cube.dat`                   | **Collocation points** | x, y, z              |
| `Mag_Test_A3Remaining.dat`                             | **Test set**           | x, y, z (Bx, By, Bz) |
| `PredMag_RemainingTestData1968T6069_...dat` (PINNs)    | **Prediction results** | x, y, z, Bx, By, Bz  |

Key switches in the scripts:

| Switch         | Meaning                                                          |
| -------------- | ---------------------------------------------------------------- |
| `flag = 0 / 1` | 0: noise-free data, 1: noisy data (SNR = 30 dB)                  |
| `embedType`    | `'none'` (standard MLP) or `'rff'` (Random Fourier Features)     |
| `DW = 1 / 0`   | Dynamic loss weighting on/off                                    |
| `k, c, n0, dn` | RAD resampling parameters (power, offset, start epoch, interval) |

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks:  
  A deep learning framework for solving forward and inverse problems involving nonlinear  
  partial differential equations. *Journal of Computational Physics*, 378, 686–707.  
  <https://doi.org/10.1016/j.jcp.2018.10.045>
- Livermore, P. W., Wu, L., Chen, L., & de Ridder, S. (2024). Reconstructions of Jupiter's  
  magnetic field using physics-informed neural networks (PINN). *Monthly Notices of the  
  Royal Astronomical Society*, 533(4), 4058–4067. <https://doi.org/10.1093/mnras/stae1928>
- Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow  
  pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*,  
  43(5), A3055–A3081. <https://doi.org/10.1137/20M1318043>
- Tancik, M., Srinivasan, P. P., Mildenhall, B., et al. (2020). Fourier features let networks  
  learn high frequency functions in low dimensional domains. *NeurIPS 33*.  
  <https://arxiv.org/abs/2006.10739>

## Repository Layout

```
MagFusion-PINNs/
├── MagFusion_PINN_VP1_outU-main-model-.py/    # Python: PINN model, baselines, evaluation
│   ├── MagFusion_PINN_Function_v2.py          # PINN core: network, embedding, losses, training
│   ├── MagFusion_PINN_main_v2.py              # Main training script (single dataset)
│   ├── runCycle.py                            # Batch training over multiple datasets
│   ├── Predict_Mag_v2.py                      # Restore checkpoint and predict target points
│   ├── RBF-main-new.py                        # Baseline: Radial Basis Function interpolation
│   ├── LSC-main-new.py                        # Baseline: Least Squares Collocation
│   ├── ESM-main-new.py                        # Baseline: Dual-layer equivalent sources (Harmonica)
│   ├── CalculateErrors_*.py                   # RMSE / MAE / STD / RRMS / ablation study
│   ├── Fig_*.py                               # Loss curves, prediction maps, 3D point cloud, ...
│   ├── input/                                 # Observation / collocation / test data (.dat)
│   ├── NN_Models/                             # Saved TF checkpoints
│   ├── Log/                                   # Training logs and loss curves
│   └── PredData/                              # Prediction results
│
└── MagFusion-PINNs-model-Figure-.m/           # MATLAB: comparison figures
    ├── SDI.m                                  # Baseline: scatteredInterpolant fusion
    ├── FigAirSeaMF.m                          # Airborne / marine field maps
    ├── FigConstratMF.m                        # PINN vs. reference component maps + errors
    ├── FigConstratMF_SingleSource.m           # Single-source case
    └── FigConstratMFwithNoise.m               # Noisy (SNR = 30 dB) case
```
