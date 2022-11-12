import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import curve_fit
from scipy.signal import peak_widths, find_peaks
from scipy.stats import norm
import math

d = [
        10, 1236, 15, 1205, 20,1253 , 25,1223, 30,1109, 35,1202, 40,1189, 45,1084,
        50,1113, 55,1158, 60,1423, 65,1783, 70,2259, 75,3000, 80,3648, 85,3769,
        90,3572, 95,3068, 100,2289, 105,1574, 110,1197, 115,1068, 120,893, 125,958,
        130,840, 135,891, 140,836, 145,832, 150,827, 155,764, 160,825, 165,837,
        170,806, 175,765, 180,740, 185,784, 190,702, 195,708, 200,671
    ]
print(len(d))
class AnylizePeaks:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.y_mean = y_data.mean()
        self.lr = linear_model.LinearRegression()
        # self.lr = linear_model.RidgeClassifier()
        # self.lr = linear_model.Ridge(alpha=.5)
    @classmethod
    def prepare_xy_data(cls, xy_arr):
        x_data = []
        y_data = []
        for n, i in enumerate(xy_arr, start=1):
            if n%2==0:
                y_data.append(i)
            else:
                x_data.append(i)
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return cls(x_data, y_data)

    def export_data(self):
        with open("input_formatted.txt", "w+") as f:
            for i in range(len(self.x_data)):
                f.write(f"{self.x_data[i]}\t{self.y_data[i]}\n")

    def msd(self):  #* average background noise value
        msd = math.sqrt(sum([(i-self.y_mean)**2 for i in self.y_data])/len(self.y_data))
        # msd = math.sqrt(sum([(i-self.y_mean)**2 for i in self.y_data])/len(self.y_data)-1)
        return msd

    def half_peak(self, max_value):
        return max_value/2

    def find_closest_points(self,value):
        left = []
        right = []
        for i in range(len(self.y_data)-1):
            if value > self.y_data[i] and value < self.y_data[i+1]:
                left = [[self.x_data[i], self.y_data[i]], [self.x_data[i+1], self.y_data[i+1]]] 
            elif value < self.y_data[i] and value > self.y_data[i+1]:
                right = [[self.x_data[i+1], self.y_data[i+1]], [self.x_data[i], self.y_data[i]]]
        print(left, right)
        return left, right

    def find_line_equation(self, array):
        x = [array[0][0], array[1][0]]
        y = [array[0][1], array[1][1]]
        A = np.vstack([x, np.ones(len(x))]).T  #* prerare matrix A
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        print(m, c)
        return m,c

    def get_peak_width(self):
        
        return

    def separate_peak_noise(self, width):
        signal_x = []
        signal_y = []
        noise_x = []
        noise_y = []
        for i in range(len(self.y_data)):
            if self.x_data[i] >= width[0] and self.x_data[i] <= width[1]:
                signal_y.append(self.y_data[i])
                signal_x.append(self.x_data[i])
            elif self.x_data[i] < width[0] or self.x_data[i] > width[0]:
                noise_x.append(self.x_data[i])
                noise_y.append(self.y_data[i])
        return np.array(signal_x), np.array(signal_y), np.array(noise_x), np.array(noise_y)
    
    def linear_regression(self, x, y):
        resh_x = x.reshape((-1,1))  #* reshape to [[val1], [val2], ... [valn]]
        reg = self.lr.fit(resh_x, y)
        linear_y = reg.predict(resh_x)
        return linear_y
    
    def gauss(self, x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def gauss_appr(self, x, y):
        x_norm = x/x.max()
        y_norm = y/y.max()
        parameters, covariance = curve_fit(self.gauss, x_norm, y_norm)
        gause_fit = self.gauss(x_norm, *parameters)*y.max()
        return gause_fit

    def get_chi(self, expected, observed):
        chi = 0
        for ind in range(len(observed)):
            chi += (observed[ind] - expected[ind])**2/expected[ind]
        # print(chi)
        return chi


# if __name__ == '__main__':
ap = AnylizePeaks.prepare_xy_data(d) 

plt.plot(ap.x_data, ap.y_data, "ro", label="experimental")
half_heigth = ap.half_peak(ap.y_data.max())
# plt.hlines(y=half_heigth, xmin=ap.x_data[0], xmax=ap.x_data[-1])

left_closest, right_closest = ap.find_closest_points(half_heigth)
left_m, left_c = ap.find_line_equation(left_closest)
right_m, right_c = ap.find_line_equation(right_closest)

left_line_range = np.array([60,90])
right_line_range = np.array([85,110])

# plt.plot(left_line_range, left_m*left_line_range+left_c, "green", label="left_line")
# plt.plot(right_line_range, right_m*right_line_range+right_c, "green", label="right_line")

signal_width = [55,115]

signal_x, signal_y, noise_x, noise_y = ap.separate_peak_noise(signal_width)

# plt.plot(signal_x, signal_y, "ro", label="signal")
# plt.plot(noise_x, noise_y, "bo", label="noise")

linear_y_noise = ap.linear_regression(noise_x, noise_y)
# plt.plot(noise_x, linear_y_noise, "black", label="linear noise")
chi_linear = ap.get_chi(noise_y, linear_y_noise)
print(chi_linear)



denoised_signal_y = signal_y-linear_y_noise.mean()
# plt.plot(signal_x, denoised_signal_y, "go", label="signal")


y_signal_noise_p, y_signal_noise_m = signal_y*1.05, signal_y*0.95 
y_gauss_signal = ap.gauss_appr(signal_x, denoised_signal_y)
y_gauss_signal_p = ap.gauss_appr(signal_x, y_signal_noise_p)
y_gauss_signal_m = ap.gauss_appr(signal_x, y_signal_noise_m)
y_gauss_true_signal = y_gauss_signal+linear_y_noise.mean()
# plt.plot(signal_x, y_gauss_true_signal, "green", label="gauss signal")
chi_gauss = ap.get_chi(signal_y, y_gauss_true_signal)
print(chi_gauss)
print(f"S/N raio is {signal_y.max()/noise_y.mean()}")
print(np.random.chisquare(38,1))
# ap.export_data()
#* doing automate peaks, width search
# peaks, _ = find_peaks(ap.y_data, height=(1500, 5000))
# print(f"peak indice {peaks}")
# width = peak_widths(ap.y_data, peaks, rel_height=1)
# print(f"widths {width}")
# plt.plot(ap.x_data[peaks], ap.y_data[peaks], "x", label="peaks")
# y_value_width = width[1:][0][0]
# x_value_indices0, x_value_indices1 = width[1:][1:]
# print(y_value_width, math.ceil(x_value_indices0[0]), math.ceil(x_value_indices1[0]))
# x_width0 = ap.x_data[math.ceil(x_value_indices0[0])]
# x_width1 = ap.x_data[math.ceil(x_value_indices1[0])]
# plt.hlines(y_value_width, x_width0, x_width1, color="C3")


#* fit for automating founded peaks
# y_gauss_signal = ap.gauss_appr(ap.x_data[math.ceil(x_value_indices0[0]):math.ceil(x_value_indices1[0])], 
#     ap.y_data[math.ceil(x_value_indices0[0]):math.ceil(x_value_indices1[0])])
# plt.plot(ap.x_data[math.ceil(x_value_indices0[0]):math.ceil(x_value_indices1[0])], y_gauss_signal, "C5", label="gauss signal")
# chi_gauss = ap.get_chi(ap.y_data[math.ceil(x_value_indices0[0]):math.ceil(x_value_indices1[0])], y_gauss_signal)
# print(chi_gauss)

plt.legend()
plt.savefig("0.png")
# plt.savefig("4.png")
