import math
import numpy as np
from matplotlib import pyplot as plt

sd = 80
E0 = 660
Ekv = 660
s0 = 25

def sigma_fg(s0, sd):
    s0d = np.sqrt(((s0**2)*(sd**2))/((s0**2)+(sd**2)))
    # print(s0d)
    return s0d

def peak_center_fg(s0, sd, E0, Ekv):
    E0d = (Ekv*(s0**2)+E0*(sd**2))/((s0**2)+(sd**2))
    # print(E0d)
    return E0d

def gauss(E, E0, sigma):
    a = 1/(math.sqrt(2*3.14)*sigma)
    return a*np.exp(-(E-E0)**2/(2*sigma**2))

def s_const(s0, sd, E0, Ekv):
    s = 1/(np.sqrt(2*3.14*((s0**2)+(sd**2))))*np.exp(-((E0-Ekv)**2)/(2*(s0**2)+(sd**2)))
    return s

def convoluted_gauss(s, sigma, peak_center, E):
    fg = s/(np.sqrt(2*3.14)*sigma)*np.exp(-(E-peak_center)**2/(2*sigma**2))
    return fg

def find_half_height_intersection(half_height, y):
    width = []
    for i in range(len(y)-1):
        if y[i-1] < half_height < y[i] or y[i-1] > half_height > y[i]:
            width.append(i)
            print(y[i])
    print(width[-1]-width[0])


if __name__ == '__main__':

    x = []
    y = []
    y2 = []
    y3 = []
    for i in range(360,960):
        x.append(i)
        y.append(convoluted_gauss(
            s_const(s0, sd, E0, Ekv), sigma_fg(s0, sd),
            peak_center_fg(s0, sd, E0, Ekv),
            i
        ))
        y2.append(gauss(i,Ekv,sd))
        y3.append(gauss(i,E0,s0))
    x = np.array(x)
    y = np.array(y)/np.array(y).sum()
    y_half = y.max()/2
    find_half_height_intersection(y_half, y)
    y2 = np.array(y2)/np.array(y2).sum()
    y3 = np.array(y3)/np.array(y3).sum()
    # plt.plot(x,y, label = "convoluted gauss")
    plt.plot(x,y2, label = "abs peak")
    plt.plot(x,y3, label = "beam peak")

    # plt.hlines(y=y_half, xmin=360, xmax=960)
    plt.legend()
    plt.xlabel("Energy, keV")
    plt.ylabel("Relative probability density")
    plt.savefig("0.png")
    # plt.savefig("1.png")

    # x = []
    # y = []
    # y2 = []
#     for i in range(460,860):
#         x.append(i)
#         y.append(gauss(i,E0,s0))
#         y2.append(gauss(i,E0,sd))

# # print(x,y)
# plt.plot(x,y)
# plt.plot(x,y2)
# plt.savefig("0.png")


