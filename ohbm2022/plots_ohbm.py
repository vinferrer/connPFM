import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

fontsize = 28
params = {
    "legend.fontsize": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
}
pylab.rcParams.update(params)
# Global variables
NRAND = 100
TR = 0.83
FIGSIZE = (45, 30)
HISTORY = "Deconvolution based on event-detection."
# Font size for plots
font = {"weight": "normal", "size": 28}
matplotlib.rc("font", **font)


ets_denoised_rss = np.loadtxt("/bcbl/home/public/PARK_VFERRER/toolbox_data/sub-002ParkMabCm_100/ets_AUC_denoised.txt")
ets_denoised_ets = np.loadtxt("/bcbl/home/public/PARK_VFERRER/toolbox_data/sub-002ParkMabCm_100/ets_AUC_denoised_ets.txt")
ets_original = np.loadtxt("/bcbl/home/public/PARK_VFERRER/toolbox_data/sub-002ParkMabCm_100/ets_AUC_original.txt")
rss = np.loadtxt("/bcbl/home/public/PARK_VFERRER/toolbox_data/sub-002ParkMabCm_100/ets_AUC_denoised_rss.txt")
mask=np.zeros(ets_original.shape)
mask[np.where(ets_denoised_rss>0)]=1
mask[np.where(ets_denoised_ets>0)]=2
mask[np.where(((ets_denoised_rss*ets_denoised_ets)>0))]=3
ets_denoised_rss = np.ma.masked_where(ets_denoised_rss < 0, ets_denoised_rss)
_ = plt.subplots(figsize=FIGSIZE)
ax0 = plt.subplot(111)
map1 = matplotlib.colors.ListedColormap(["#FF000000", "#67bd7f", "#f44336", "#f8ff66"])
map1.set_under(color="#FF000000", alpha="0")
# map2
divider = make_axes_locatable(ax0)
ax1 = divider.append_axes("bottom", size="25%", pad=1)
im = ax0.imshow(ets_original.T, cmap="Greys", vmax=0.1, aspect="auto")
im = ax0.imshow(mask.T, cmap=map1, aspect="auto", alpha=1)
ax1.plot(rss)
ax1.set_xlim(0, len(rss))
ax0.set_title("Edge-time series")
ax0.set_xlabel("Time (TR)")
ax0.set_ylabel("Edge-edge connections")
ax1.set_ylabel("RSS")
plt.show()
