import matplotlib.pyplot as plt
import numpy as np
import MagFusion_PINN_main_v2 as main
import Predict_Mag_v2 as pred
from matplotlib import rcParams
config = {
    "font.family": 'serif',
    "font.serif": 'Times New Roman',
    "axes.unicode_minus": False,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # "ztick.labelsize": 12,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    # "ztick.major.width": 2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    # "ztick.direction": "in",
    "axes.titlesize": 15,  # title font size
    "axes.titleweight": "bold",  # title font weight
    "axes.labelsize": 12,  # axis label font size
    "axes.labelweight": "bold"  # axis label font weight
}
rcParams.update(config)

# Load data
dataObs_filepath = main.dataObs_filepath
Obs_data = np.loadtxt(dataObs_filepath, delimiter=',',skiprows=1)
x_obs = Obs_data[:, 0]
y_obs = Obs_data[:, 1]
z_obs = Obs_data[:, 2]

dataCol_filepath = main.dataCol_filepath
Col_data = np.loadtxt(dataCol_filepath, delimiter=',',skiprows=1)
x_col = Col_data[:, 0]
y_col = Col_data[:, 1]
z_col = Col_data[:, 2]

data_pred = np.loadtxt(pred.data_pred_path, delimiter=',',skiprows=1)
xpred = data_pred[:, 0]
ypred = data_pred[:, 1]
zpred = data_pred[:, 2]

# Create the 3D plot
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points (original coordinates kept)
#Observations-Marine Magnetic 、Observations-Airborne Magnetic、Fusion Points
ax.scatter(x_obs[0:819], y_obs[0:819], z_obs[0:819], s=20, c='r', marker='.',alpha=1, label='Observations-Shipborne Magnetic', depthshade=True)
ax.scatter(x_obs[819:], y_obs[819:], z_obs[819:], s=20, c='g', marker='.',alpha=1, label='Observations-Airborne Magnetic', depthshade=True)
# ax.scatter(x_col[::1], y_col[::1], z_col[::1], s=2, c='b', marker='.', alpha=0.5,label='Collocation Points', depthshade=True)
ax.scatter(xpred[::1], ypred[::1], zpred[: :1], s=10, c='b', marker='.', alpha=0.5,label='Fusion Points', depthshade=True)

# Set axis labels and tick fonts
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)', labelpad=15)
plt.setp(plt.gca().get_xticklabels(), fontweight='bold')
plt.setp(plt.gca().get_yticklabels(), fontweight='bold')
plt.setp(plt.gca().get_zticklabels(), fontweight='bold')
ax.invert_zaxis()

# Tick padding (optional)
ax.tick_params(axis='z', which='major', pad=8)

# Legend and title
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9))
# ax.set_title('3D Distribution with Depth-Positive', pad=15)

# View angle
ax.view_init(elev=35, azim=-135)  # adjust the view to better show the depth direction

# Reference grid (optional)
ax.grid(True, alpha=0.5, linestyle=':')

plt.tight_layout()
plt.show()