import matplotlib.pyplot as plt
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

rrms_Bx = [1.235, 0.724, 0.621, 0.507, 0.553, 0.462]
rrms_By = [1.291, 0.755, 0.537, 0.336, 0.468, 0.340]
rrms_Bz = [1.263, 0.742, 0.605, 0.469, 0.528, 0.499]

N = [656, 1312, 1968, 3280, 4592, 6560]

plt.figure()
plt.plot(N, rrms_Bx, 'r*-', label='Bx')
plt.plot(N, rrms_By, 'b*-', label='By')
plt.plot(N, rrms_Bz, 'g*-', label='Bz')
plt.xlabel('Number of training data points')
plt.ylabel('RRMS(%)')
plt.setp(plt.gca().get_xticklabels(), fontweight='bold')
plt.setp(plt.gca().get_yticklabels(), fontweight='bold')
plt.grid(True)
plt.legend()
plt.show()
