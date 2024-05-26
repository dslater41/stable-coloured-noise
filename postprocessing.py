import numpy as np
import matplotlib.pyplot as plt


def crossing_intervals(
    signal: np.ndarray
) -> np.ndarray: 
    """
        Function that takes a signal as input and returns the zero-crossing lengths
    """
    clipped_signal=(2*(signal>=0)-1)
    crossings=np.concatenate((np.array([1]),(clipped_signal[:-1]*clipped_signal[1:])))#.astype(int)
    crossings=((1-crossings)/2).astype(int)
    crossing_locations = np.where(crossings==1)[0]
    return crossing_locations[1:]-crossing_locations[:-1]


def plot_pdf(
    signal: np.ndarray,
    plot_log_y: bool=False,
    bins: int=None,
)->None: 
    plt.figure()
    plt.hist(signal,edgecolor='k',alpha=0.7,density=True,bins=bins,range=(0,signal.max()))
    if plot_log_y:
        plt.yscale('log')
    plt.xlabel(r'Crossing interval lag $\tau$')
    plt.ylabel(r'Density function $f_X (\tau)$')
    plt.title('Crossings empirical PDF')
    plt.show()
