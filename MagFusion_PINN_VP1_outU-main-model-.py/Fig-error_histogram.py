import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# File path configuration
# =============================================================================
# --- Reference data ---
REF_FILE = 'input/hy-Mag_Sea_TestSet_Remaining15851_x_y_z_Bx_By_Bz.dat'

# --- Prediction result files ---
PINN_FILE = 'PredData/NumObs5240/PredMag_RemainingTestData5240T15851_NN6_40_swish_noEmb_Adam_RADk1c0n3000d600_nEpo50000_BS3000_Seed12345.dat'
LSC_FILE = 'input/Mag_test_LSC_5240T15851_cauchy.dat'
RBF_FILE = 'input/Mag_test_RBF_5240T15851.dat'
ESM_FILE = 'input/Mag_test_DualLayerESM_5240T15851.dat'
SDI_FILE = 'input/Mag_test_SDI_5240T15851.dat'
# =============================================================================
#  Data loading and error computation
# =============================================================================
# ---------------- Load Reference ----------------
Test_data_ref = np.loadtxt(REF_FILE, delimiter=',', skiprows=1)
Bx_ref = Test_data_ref[:, 3]
By_ref = Test_data_ref[:, 4]
Bz_ref = Test_data_ref[:, 5]

# ---------------- Load Data ----------------
data_PINN = np.loadtxt(PINN_FILE, delimiter=',', skiprows=1)
data_LSC = np.loadtxt(LSC_FILE, delimiter=',', skiprows=1)
data_RBF = np.loadtxt(RBF_FILE, delimiter=',', skiprows=1)
data_ESM = np.loadtxt(ESM_FILE, delimiter=',', skiprows=1)
data_SDI = np.loadtxt(SDI_FILE, delimiter=',', skiprows=1)

# ---------------- Compute Errors ----------------
err_PINN = {
    'Bx': data_PINN[:, 3] - Bx_ref,
    'By': data_PINN[:, 4] - By_ref,
    'Bz': data_PINN[:, 5] - Bz_ref
}
err_LSC = {
    'Bx': data_LSC[:, 3] - Bx_ref,
    'By': data_LSC[:, 4] - By_ref,
    'Bz': data_LSC[:, 5] - Bz_ref
}
err_RBF = {
    'Bx': data_RBF[:, 3] - Bx_ref,
    'By': data_RBF[:, 4] - By_ref,
    'Bz': data_RBF[:, 5] - Bz_ref
}
err_ESM = {
    'Bx': data_ESM[:, 3] - Bx_ref,
    'By': data_ESM[:, 4] - By_ref,
    'Bz': data_ESM[:, 5] - Bz_ref
}
err_SDI = {
    'Bx': data_SDI[:, 3] - Bx_ref,
    'By': data_SDI[:, 4] - By_ref,
    'Bz': data_SDI[:, 5] - Bz_ref
}

# =============================================================================
# Plotting
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

labels = ['PINN', 'LSC', 'RBF', 'ESM', 'SDI']
colors = ['#D62728', '#1F77B4', '#2CA02C', '#9467BD', '#FF7F0E']
n_bins = 61

panels = [
    ('Bx', err_PINN['Bx'], err_LSC['Bx'], err_RBF['Bx'], err_ESM['Bx'], err_SDI['Bx'], '(a) Bx Error'),
    ('By', err_PINN['By'], err_LSC['By'], err_RBF['By'], err_ESM['By'], err_SDI['By'], '(b) By Error'),
    ('Bz', err_PINN['Bz'], err_LSC['Bz'], err_RBF['Bz'], err_ESM['Bz'], err_SDI['Bz'], '(c) Bz Error'),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

for i, (comp, e_pinn, e_lsc, e_rff, e_esm, e_sdi, title) in enumerate(panels):
    ax = axes[i]
    errs = [e_pinn, e_lsc, e_rff, e_esm, e_sdi]

    e_all = np.concatenate(errs)
    e_lim = max(abs(e_all.min()), abs(e_all.max())) * 1.05
    bins = np.linspace(-e_lim, e_lim, n_bins)

    for err, label, c in zip(errs, labels, colors):
        ax.hist(err, bins=bins, density=False, histtype='step',
                alpha=0.85, color=c, linewidth=2.5, label=label)

    ax.axvline(0, color='k', linestyle='--', linewidth=1.5)

    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.tick_params(axis='both', which='major', labelsize=17)
    for tick_label in (ax.get_xticklabels() + ax.get_yticklabels()):
        tick_label.set_fontweight('bold')

    ax.set_xlabel('Error (nT)', fontsize=20, fontweight='bold')
    if i == 0:
        ax.set_ylabel('Frequency (log scale)', fontsize=20, fontweight='bold')

    ax.text(0.03, 0.95, title, transform=ax.transAxes,
            fontsize=22, fontweight='bold',
            verticalalignment='top', horizontalalignment='left',
            color='black')

handles, leg_labels = axes[0].get_legend_handles_labels()

# ----------------- Global legend at the bottom -----------------
import matplotlib.lines as mlines

custom_lines = [mlines.Line2D([0], [0], color=c, lw=3) for c in colors]

legend = fig.legend(custom_lines, labels, loc='lower center', ncol=5,
                    frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.02))

for text in legend.get_texts():
    text.set_fontweight('bold')

plt.tight_layout(rect=[0, 0.15, 1, 1])

plt.savefig('RealData-Error_Histogram_Comparison_1x3_LogScale.png', dpi=600, bbox_inches='tight')
plt.show()