import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 12})

out_name = "out_frozen_lake3.csv"  # Default: out.csv
hist_name = 'hist_frozen_lake.pkl'

DF = pd.read_csv(out_name)
DF.rename(columns={'Errors': 'Asymptotic Convergence Error'}, inplace=True)

with open(hist_name, "rb") as f:  # Python 3: open(..., 'rb')
    hist_td, hist_tdc, hist_vrtdc, hist_vrtd = pickle.load(f)
f.close()

# sns.set(style="ticks", palette="pastel")
fig, (ax1, ax2) = plt.subplots(ncols=2)

def easy_plot(hist, color, label, cut_off=None, percentile=95, fill=True):
    upper_loss = np.percentile(hist, percentile, axis=0)
    lower_loss = np.percentile(hist, 100 - percentile, axis=0)
    avg_loss = np.mean(hist, axis=0)
    x = np.arange(avg_loss.shape[0])

    if cut_off is None:
        ax1.plot(avg_loss, c=color, label=label)
    else:
        ax1.plot(list(avg_loss[:cut_off]), c=color, label=label)

    if fill:
        if cut_off is None:
            ax1.fill_between(x[:cut_off], lower_loss[:cut_off], upper_loss[:cut_off], color=color, alpha=0.3)
        else:
            ax1.fill_between(x[:cut_off], lower_loss[:cut_off], upper_loss[:cut_off], color=color, alpha=0.3)


easy_plot(hist_tdc, "orange", "TDC")
easy_plot(hist_td, "g", "TD")
easy_plot(hist_vrtd, "b", "VRTD: M=3000", cut_off=len(hist_td[0]))
easy_plot(hist_vrtdc, "r", "VRTDC: M=3000", cut_off=len(hist_td[0]))
ax1.set_ylim(0, 0.05)
ax1.legend(loc=1)
ax1.set_ylabel(r"Convergence Error $||\theta - \theta^\ast ||^2$")
ax1.set_xlabel("# of gradient computations")


sns.boxplot(x="Batch Size", y="Asymptotic Convergence Error", data=DF, hue="Algorithm", palette=["red", "blue"], ax=ax2, showmeans=True)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[0:], labels=labels[0:])


fig.tight_layout()
fig.set_size_inches(10, 4, forward=True)
fig.savefig("fig-side-by-side-corrected-2.png", dpi=300)
plt.show()
