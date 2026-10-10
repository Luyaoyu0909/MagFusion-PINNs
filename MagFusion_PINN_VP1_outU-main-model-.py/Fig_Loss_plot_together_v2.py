import os
import matplotlib.pyplot as plt
import numpy as np
import MagFusion_PINN_main_v2 as main
from matplotlib import rcParams
config = {
    "font.family": 'Times New Roman',
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
    "axes.labelsize": 15,  # axis label font size
    "axes.labelweight": "bold"  # axis label font weight
}
rcParams.update(config)

# %% # ! Main
if __name__ == "__main__":
    Train_info = np.loadtxt(main.fileLossVars_path, delimiter=',', skiprows=1)
    epoch = Train_info[:, 0]
    loss = Train_info[:, 1]
    LObs_Bx = Train_info[:, 2]
    LObs_By = Train_info[:, 3]
    LObs_Bz = Train_info[:, 4]
    Lap = Train_info[:, 5]
    w_obs = Train_info[:, 6]
    time = Train_info[:, 7]
    Lr = Train_info[:, 8]

    # arch_label = f"{main.actiFun}+{main.embedType}"

    plt.figure(figsize=(10, 6))
    plt.semilogy(epoch, LObs_Bx, 'b-', label="Loss Bx")
    plt.semilogy(epoch, LObs_By, 'g-', label="Loss By")
    plt.semilogy(epoch, LObs_Bz, 'm-', label="Loss Bz")
    plt.semilogy(epoch, Lap, 'c-', label="Loss Laplace")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    # plt.title(arch_label)
    plt.setp(plt.gca().get_xticklabels(), fontweight='bold')
    plt.setp(plt.gca().get_yticklabels(), fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = f"{main.Log_file}/NumObs{main.numPiont_Obs}/Loss_{main.dir_Name}"
    plt.savefig(fig_path + '.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    plt.plot(epoch, Lr, 'r-')
    plt.xlabel("Epoch")
    plt.ylabel("Lr")
    # plt.title(arch_label)
    plt.setp(plt.gca().get_xticklabels(), fontweight='bold')
    plt.setp(plt.gca().get_yticklabels(), fontweight='bold')
    plt.grid(True)
    plt.tight_layout()

    fig_path = f"{main.Log_file}/NumObs{main.numPiont_Obs}/Lr_{main.dir_Name}"
    plt.savefig(fig_path + '.png', dpi=300, bbox_inches='tight')

    plt.show()

    train_time = time.sum()
    print(f"Total training time: {train_time:.3f} s")

