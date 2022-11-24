import sys
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt


# plt.plot(x, y, "ro", label = "experemental")
# plt.savefig("0.png")


class Poly_Appr:
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    @classmethod
    def load_input_data(cls):
        x_data, y_data = [], []
        with open("input.txt", "r") as f:
            data = f.readlines()
        for i in data:
            i = i.split("    ")
            x_data.append(float(i[0]))
            y_data.append(float(i[1]))
        return cls(x_data, y_data)

    #* creates new x and y arrays with [lenght = initial length - period (shift)]
    def sma(self, period):
        y_sma_array = []
        x_sma_array = []
        back_ind = len(self.x_data)-period+1
        # x_sma_array = self.x_data[-back_ind:]
        for i in range(len(self.y_data)-1):
            shift_y = self.y_data[i:period+i]
            shift_x = self.x_data[i:period+i]
            y_sma_array.append(reduce(lambda x, y: x+y, shift_y)/period)
            x_sma_array.append(reduce(lambda x, y: x+y, shift_x)/period)
            if period+i == len(self.y_data):
                break
        return x_sma_array, y_sma_array

    def poly_fit(self, x, y, degree):
        z = np.polyfit(x, y, degree)
        p = np.poly1d(z)
        with open("polynomial equiation.txt", "w+") as f:
            f.write(str(p))
        return p

    def get_chi(self, expected, observed):
        chi = 0
        for ind in range(len(observed)):
            chi += (abs(observed[ind]) -
                    abs(expected[ind]))**2/abs(expected[ind])
        print("chi squared ", chi)
        return chi

    def get_variance(self, expected, observed): #* it's like variance but not
        sub_sq = 0
        for ind in range(len(observed)):
            sub_sq += (abs(observed[ind]) -
                       abs(expected[ind]))**2
        coef = sub_sq/(len(expected)-1)
        return coef


if __name__ == '__main__':
    poly_appr = Poly_Appr.load_input_data()
    print(poly_appr.x_data, poly_appr.y_data)

    plt.plot(poly_appr.x_data, poly_appr.y_data, "ro", label="experemental")
    x_sma_ar_2, y_sma_ar_2 = poly_appr.sma(2)
    x_sma_ar_3, y_sma_ar_3 = poly_appr.sma(3)
    x_sma_ar_4, y_sma_ar_4 = poly_appr.sma(4)
    # plt.plot(x_sma_ar_2, y_sma_ar_2, color="blue", label="sma with 2 elements")
    plt.plot(x_sma_ar_3, y_sma_ar_3, color="green", label="sma with 3 elements")
    # plt.plot(x_sma_ar_3, y_sma_ar_3, color="black")
    # plt.plot(x_sma_ar_4, y_sma_ar_4, color="green")
    max_chi = np.random.chisquare(len(x_sma_ar_2)-1)
    print("max chi", max_chi)
    for i in range(1, 15):
        poly = poly_appr.poly_fit(x_sma_ar_2, y_sma_ar_2, i)
        y_poly_ar = poly(x_sma_ar_2)
        if 0.95 < poly_appr.get_variance(y_poly_ar, y_sma_ar_2) < 1.05:
            poly_appr.get_chi(y_poly_ar, y_sma_ar_2)
            print("sub_sq/degrees of freedom", poly_appr.get_variance(y_poly_ar, y_sma_ar_2))
            print(f"good fit with {i} degree")
            # break
    plt.plot(x_sma_ar_2, y_poly_ar, color="black", label="poly fit")
    plt.legend()
    plt.savefig("2.png")
