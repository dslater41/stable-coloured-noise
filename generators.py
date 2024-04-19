from typing import Callable
import numpy as np
from scipy.fft import fft, ifft 


def ffm(
    autcorrelation_function: Callable, 
    size: int, 
) -> np.ndarray: 
    """
        Generates a Gaussian coloured noise with `size` elements, and some prescribed 
        autocorrelation defined by `autocorrelation_function`
    """
    lags=range(size)
    autocorrelation=list(map(autcorrelation_function, lags))
    filtered_autocorrelation=np.array(autocorrelation[:int(size/2)+1]+autocorrelation[1:int(size/2)][::-1])

    gaussian_rvs=np.random.normal(size=size)

    filtered_autocorrelation_f=fft(filtered_autocorrelation)
    gaussian_rvs_f=fft(gaussian_rvs)

    return np.real(
        ifft(
            np.sqrt(np.abs(filtered_autocorrelation_f))*gaussian_rvs_f
        )
    )


def ffm_iterative(
    autocorrelation_function: Callable, 
    target_rvs: np.ndarray, 
    iterations: int=10, 
) -> np.ndarray:
    """
        Imprints an autocorrelation function onto a set of target_rvs (drawn from some distribution), using `iterations` iterations
    """
    size=len(target_rvs)
    gaussian_noise=ffm(autocorrelation_function,size)

    power_spectrum=np.abs(fft(gaussian_noise))
    iteration=target_rvs.copy()
    for _ in range(iterations): 
        iteration_fourier=fft(iteration)
        iteration_fourier=iteration_fourier/np.abs(iteration_fourier)*power_spectrum
        for value, location in zip(sorted(target_rvs), np.argsort(np.real(ifft(iteration_fourier)))):
            iteration[location] = value
    return iteration


def simulate_cauchy_coloured_noise(
    autocorrelation_function: Callable,
    size: int,
) -> np.ndarray:
    """
        Simulates a Cauchy coloured noise with some input autocorrelation of the Gaussian process `autocorrelation_function`
    """
    positive_gaussian_rvs = np.abs(np.random.normal(size=size))
    X=ffm(autocorrelation_function, size)
    Y=ffm_iterative(autocorrelation_function, positive_gaussian_rvs)
    return X/Y


if __name__=="__main__":
    print("Running...")

    def acf(tau: float) -> float: 
        half_life=64
        decay_constant = np.log(2) * half_life**-2
        return np.exp(-decay_constant*tau**2)
    
    size=2**20
    positive_gaussian_rvs = np.abs(np.random.normal(size=size))
    X=ffm(acf, size)
    Y=ffm_iterative(acf,positive_gaussian_rvs)
    Z=simulate_cauchy_coloured_noise(acf, size)


    import plotly.express as px 
    px.line(X[:2000]).show()
    px.line(Y[:2000]).show()
    px.line(Z[:2000]).show()
