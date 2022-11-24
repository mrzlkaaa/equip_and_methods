import math

sd = 80
E0 = 660
s0 = 25


def simulate_spectrometer(sd, s0, E0, E):
    # H = 1/(2*3.14*sd*s0) * \
    #     math.exp(-(((E-E0)**2)/(2*sd**2))-(((E-E0)**2)/(2*s0**2)))
    H = 1/(2*3.14*sd*s0) * \
        math.exp(-(((E-E0)**2)/(2*sd**2))-(((E-E0)**2)/(2*s0**2)))
    print(H)
    return


if __name__ == '__main__':
    simulate_spectrometer(sd, s0, E0, 660)
