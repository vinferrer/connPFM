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


ets_denoised_rss = np.loadtxt("/bcbl/home/public/PARK_VFERRER/toolbox_data/sub-002ParkMabCm_100/ets_denoised.txt")
ets_denoised_ets = np.loadtxt("/bcbl/home/public/PARK_VFERRER/toolbox_data/sub-002ParkMabCm_100/ets_denoised_ets.txt")
ets_original = np.loadtxt("/bcbl/home/public/PARK_VFERRER/toolbox_data/sub-002ParkMabCm_100/ets_original.txt")
rss = np.loadtxt("/bcbl/home/public/PARK_VFERRER/toolbox_data/sub-002ParkMabCm_100/ets_denoised_rss_v_rss.txt")
mask_orig=np.zeros(ets_original.shape)
mask_orig[np.where(ets_original>0)]=1
comb= np.zeros(ets_original.shape)
comb[np.where(((ets_denoised_rss*ets_denoised_ets)>0))]=1
mask_ets=np.zeros(ets_original.shape)
mask_ets[np.where(ets_denoised_ets>0)]=1
mask_ets=mask_ets-comb
mask_rss=np.zeros(ets_original.shape)
mask_rss[np.where(ets_denoised_rss>0)]=1
mask_rss=mask_rss-comb
plot_mask=mask_orig+comb+2*mask_ets+3*mask_rss
_ = plt.subplots(figsize=FIGSIZE)
ax0 = plt.subplot(111)
map1 = matplotlib.colors.ListedColormap(["#FF000000","#aba8a7", "#38e067", "#fa1100", "#f5ff21"])
# map2
divider = make_axes_locatable(ax0)
ax1 = divider.append_axes("bottom", size="25%", pad=1)
# im = ax0.imshow(ets_original.T, cmap="Greys", vmax=0.1, aspect="auto")
im = ax0.imshow(plot_mask.T, cmap=map1, aspect="auto")
ax1.plot(rss)
ax1.set_xlim(0, len(rss))
ax0.set_title("Edge-time series")
ax0.set_xlabel("Time (TR)")
ax0.set_ylabel("Edge-edge connections")
ax1.set_ylabel("RSS")
plt.show()
