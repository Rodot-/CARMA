import numpy as np

def magToFlux(mag):
    return 10**(-0.4*(mag-12))*1.74e5


def fluxToMag(flux):

    return np.log10(flux/1.74e5)/(-0.4)+12

