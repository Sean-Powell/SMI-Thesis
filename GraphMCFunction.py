from matplotlib import pyplot as plt
from time import time
from os import mkdir
import numpy as np
from math import sin, cos, tan, log, e

COEFFICIENT_BIT_LENGTH = 8
EXPONENT_BIT_LENGTH = 4
FUNCTION_BIT_LENGTH = 3
FLOAT_PRECISION_LENGTH = 10
X_EXTENSION = 0.2

DATASET_PATH = "C:/Users/seanp/PycharmProjects/SMI-Thesis/dataset.txt"
MONTECARLO_PATH = "C:/Users/seanp/PycharmProjects/SMI-Thesis/monte_carlo_func.txt"
x = []
y = []
s = []

time_str = str(time())
mkdir(time_str)

_float_bit_length = -1  # is automatically set by calculate_float_bit_length()
_term_bit_length = -1  #


def read_dataset():
    f = open(DATASET_PATH)
    i = 0
    for line in f:
        if i < 1:
            i += 1
        else:
            line = line.strip()
            line_split = line.split(" ")

            x_string = line_split[1]
            index = x_string.find("(")
            x_value = x_string[:index]
            x.append(float(x_value))

            y_string = line_split[5]
            index = x_string.find("(")
            y_value = y_string[:index + 1]
            s_value = y_string[index + 2: len(y_string) - 1]
            y.append(float(y_value))
            s.append(float(s_value))

            # line_split = line.split("\t")
            # x.append(float(line_split[1]))
            # y.append(float(line_split[2]))
            # s.append(float(line_split[3]))

def calculate_float_bit_length():
    global _float_bit_length
    number = '9' * FLOAT_PRECISION_LENGTH
    number = int(number)
    length = 0
    n = 2 ** length
    while n < number:
        length += 1
        n = 2 ** length
    _float_bit_length = length


def calculate_term_bit_length():
    global _term_bit_length
    length = 2  # for both signs
    length += (2 * _float_bit_length)
    length += COEFFICIENT_BIT_LENGTH
    length += EXPONENT_BIT_LENGTH
    length += FUNCTION_BIT_LENGTH
    _term_bit_length = length


def string_to_decimal(n):
    num = n
    dec_value = 0

    base1 = 1

    len1 = len(num)
    for i in range(len1 - 1, -1, -1):
        if num[i] == '1':
            dec_value += base1
        base1 = base1 * 2

    return dec_value


def binary_to_float(i, d):
    i = str(string_to_decimal(i))
    d = str(string_to_decimal(d))
    return i + "." + d


def decode_function_value(n, value, exponent):
    d = string_to_decimal(n)
    if d == 1:
        return sin(value ** exponent)
    if d == 2:
        return cos(value ** exponent)
    if d == 3:
        return tan(value ** exponent)
    if d == 4:
        return log(value ** exponent) # stops a log error in the case of x being 0
    if d == 5:
        return e ** (value + exponent)
    return value ** exponent


def evaluate_chromosome(chromosome, value):
    # todo add function decoding support
    index = 0
    ans = 0
    while index < (len(chromosome) - 1):
        coeff_sign = int(chromosome[index:index+1])
        index += 1
        coeff_i = chromosome[index: index + COEFFICIENT_BIT_LENGTH]
        index += COEFFICIENT_BIT_LENGTH
        coeff_d = chromosome[index: index + _float_bit_length]
        index += _float_bit_length

        func_binary = chromosome[index: index + FUNCTION_BIT_LENGTH]
        index += FUNCTION_BIT_LENGTH
        try:
            exp_sign = int(chromosome[index:index + 1])
        except ValueError:
            # i have no idea why this causes an error or why this occurs
            exp_sign = 0
        index += 1
        exp_i = chromosome[index: index + EXPONENT_BIT_LENGTH]
        index += EXPONENT_BIT_LENGTH
        exp_d = chromosome[index: index + _float_bit_length]
        index += _float_bit_length

        if coeff_sign:
            coeff_sign = "-"
        else:
            coeff_sign = "+"

        if exp_sign:
            exp_sign = "-"
        else:
            exp_sign = "+"

        coeff = float(coeff_sign + binary_to_float(coeff_i, coeff_d))
        exp = float(exp_sign + binary_to_float(exp_i, exp_d))
        func = decode_function_value(func_binary, value, exp)
        ans += func * coeff

    return ans


def plot_monte_carlo(chromosomes):
    plt.plot(x, y, 'o')
    plt.errorbar(x, y, yerr=s, fmt=' ')
    func_x = np.linspace(min(x), max(x) + 0.2, 200)
    for i in range(89):
        func_y = []
        for j in func_x:
            func_y.append(evaluate_chromosome(chromosomes[i], j))
        plt.plot(func_x, func_y)
    # for c in chromosomes:
    #     func_y = []
    #     for i in func_x:
    #         func_y.append(evaluate_chromosome(c, i))

        plt.plot(func_x, func_y)

    plt.xlim(0, max(x) + X_EXTENSION)
    plt.ylim(10, 200)
    plt.xlabel("Redshift")
    plt.ylabel("Distance modulus")
    plt.title("Monte-Carlo Simulation")
    plt.savefig(time_str + "/monte-carlo.png")
    plt.show()

read_dataset()
calculate_term_bit_length()
calculate_float_bit_length()

file_f = open(MONTECARLO_PATH)
chroms = []
for line in file_f:
    parts = line.split(",")
    chroms.append(parts[0])

plot_monte_carlo(chroms)
