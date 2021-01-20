from random import randint, uniform, shuffle
from math import modf, pow, floor, ceil, sin, cos, tan, log, e
from matplotlib import pyplot as plt
import numpy as np
from time import time
from os import mkdir

MIN_LENGTH = 2
MAX_LENGTH = 4
COEFFICIENT_MIN = 0
COEFFICIENT_MAX = 20
EXPONENT_MIN = 0
EXPONENT_MAX = 10
COEFFICIENT_BIT_LENGTH = 8
EXPONENT_BIT_LENGTH = 4
FUNCTION_BIT_LENGTH = 3
FLOAT_PRECISION_LENGTH = 10
MAX_NUMBER_OF_MUTATIONS = 40
DATASET_PATH = "C:/Users/seanp/PycharmProjects/SMI-Thesis/dataset.txt"
POPULATION_SIZE = 500
SELECTION_RATE = 40
MUTATION_RATE = 10
TERM_MUTATION_RATE = 20
GENERATION_COUNT = 6000
GRAPH_STEP = GENERATION_COUNT // 10

_float_bit_length = -1  # is automatically set by calculate_float_bit_length()
_term_bit_length = -1  # is automatically set by calculate_term_bit_length()

x = []
y = []
s = []

best_chi_squared = []

time_str = str(time())
mkdir(time_str)
file_f = open(time_str + "/output.txt", "w")


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


def convert_to_binary(to_covert, bit_length):
    return format(to_covert, '0' + str(bit_length) + 'b')


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


def create_number(min_number, max_number, bit_length):
    if max_number > (2 ** bit_length):
        raise Exception("Max number is too large for the specified bit length")

    number_sign = randint(0, 1)
    number = uniform(min_number, max_number)
    d, i = modf(number)
    i = int(i)
    d = int(floor(d * pow(10, FLOAT_PRECISION_LENGTH)))
    i_bin = convert_to_binary(i, bit_length)
    d_bin = convert_to_binary(d, _float_bit_length)
    return number_sign, i_bin, d_bin


def create_function():
    #  sin - 1
    #  cos - 2
    #  tan - 3
    #  ln  - 4
    #  e   - 5
    #  x   - any other
    index = randint(0, (2 ** FUNCTION_BIT_LENGTH) - 1)
    return convert_to_binary(index, FUNCTION_BIT_LENGTH)


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


def decode_function(n):
    d = string_to_decimal(n)
    if d == 1:
        return sin
    if d == 2:
        return cos
    if d == 3:
        return tan
    if d == 4:
        return log  # stops a log error in the case of x being 0
    if d == 5:
        return e
    return None


def decode_function_string(n, exponent):
    d = string_to_decimal(n)
    if d == 1:
        return 'sin(x**' + str(exponent) + ")"
    if d == 2:
        return 'cos(x**' + str(exponent) + ")"
    if d == 3:
        return 'tan(x**' + str(exponent) + ")"
    if d == 4:
        return 'log(x**' + str(exponent) + ")"
    if d == 5:
        return 'e**(x+' + str(exponent) + ")"
    return 'x**' + str(exponent)


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


def create_chromosome():
    chromosome = ""
    chromosome_length = randint(MIN_LENGTH, MAX_LENGTH)
    for _ in range(chromosome_length):
        chromosome = add_term(chromosome)

    return chromosome


def chromosome_to_string(chromosome):
    index = 0
    output = ""
    while index < len(chromosome) - 1:
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

        exp = float(exp_sign + binary_to_float(exp_i, exp_d))
        func_string = decode_function_string(func_binary, exp)
        output += (coeff_sign + binary_to_float(coeff_i, coeff_d) + func_string)
    return output


def build_chromosome(chromosome):
    index = 0
    parts = []
    functions = []

    while index < len(chromosome):
        coeff_sign = int(chromosome[index:index + 1])
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
        func = decode_function(func_binary)

        parts.append(coeff)
        parts.append(exp)
        functions.append(func)

    return functions, parts


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


def remove_term(chromosome):
    new_chromosome = ""
    number_of_terms = len(chromosome) // _term_bit_length
    term_to_remove = randint(0, number_of_terms - 1)
    index = 0
    for i in range(0, number_of_terms):
        if i < term_to_remove or i > term_to_remove:
            new_chromosome = new_chromosome + chromosome[index:index + _term_bit_length]
        index = index + _term_bit_length

    return new_chromosome


def add_term(chromosome):
    coeff_sign, coeff_i, coeff_d = create_number(COEFFICIENT_MIN, COEFFICIENT_MAX, COEFFICIENT_BIT_LENGTH)
    exp_sign, exp_i, exp_d = create_number(EXPONENT_MIN, EXPONENT_MAX, EXPONENT_BIT_LENGTH)
    function = create_function()
    chromosome += str(coeff_sign)
    chromosome += str(coeff_i)
    chromosome += str(coeff_d)
    chromosome += str(function)
    chromosome += str(exp_sign)
    chromosome += str(exp_i)
    chromosome += str(exp_d)
    return chromosome


def crossover(chromosome_a, chromosome_b):
    child = ""
    length_a = len(chromosome_a)
    length_b = len(chromosome_b)
    if length_a != length_b:
        while length_a < length_b:
            chromosome_a += "0"
            length_a = len(chromosome_a)

        while length_b < length_a:
            chromosome_b += "0"
            length_b = len(chromosome_b)

    for i in range(length_a - 1):
        i_a = chromosome_a[i]
        i_b = chromosome_b[i]
        if (i_a == '1' and i_b == '1') or (i_a == '0' and i_b == '0'):
            child += "0"
        else:
            child += "1"

    return child


def mutate(chromosome):  # todo add ability to remove and add terms during the mutation
    chance = randint(0, 100)
    num_of_terms = len(chromosome) // _term_bit_length
    if chance <= TERM_MUTATION_RATE:
        chance = randint(0, 1)
        if chance and num_of_terms < MAX_LENGTH:
            chromosome = add_term(chromosome)
        elif num_of_terms > MIN_LENGTH:
            chromosome = remove_term(chromosome)

    for _ in range(MAX_NUMBER_OF_MUTATIONS):
        index = randint(1, len(chromosome) - 1)
        gene = chromosome[index]
        if gene == "0":
            chromosome = chromosome[:index-1] + "1" + chromosome[index:]
        else:
            chromosome = chromosome[:index-1] + "0" + chromosome[index:]
    return chromosome


def fitness_function(chromosome):
    chi_2 = 0
    functions, parts = build_chromosome(chromosome)
    for index in range(len(x)):
        value = x[index]
        ans = 0
        for i in range(0, len(parts), 2):
            func_index = int(i / 2)
            try:
                ans += (parts[i] * (functions[func_index](value ** parts[i + 1])))
            except TypeError:
                # for when the function type is None when the it is just x
                ans += parts[i] * (value ** parts[i + 1])
        chi_2 += pow(((y[index] - ans) / s[index]), 2)
    return chi_2


def graph_chromosome(chromosome, index):
    plt.plot(x, y, 'o')
    plt.errorbar(x, y, yerr=s, fmt=' ')

    func_x = np.linspace(min(x), max(x) + 0.2, 200)
    func_y = []
    for i in func_x:
        func_y.append(evaluate_chromosome(chromosome, i))

    z = np.linspace(min(x), max(x), 100)
    mu = np.log((z ** 2.17145) * ((-z ** 2.82) + z + np.exp(z))) + 42.83 - 5. * np.log10(0.7)
    plt.plot(z, mu, c='g')
    plt.plot(func_x, func_y, c='r')
    plt.xlim(0, max(x) + 0.2)
    plt.xlabel("Redshift")
    plt.ylabel("Distance modulus")
    plt.title("Genetic Algorithm Rank:" + str(index))
    plt.savefig(time_str + "/" + str(index) + ".png")
    plt.show()


def graph_best_chi():
    number_of_gens = []
    for i in range(GENERATION_COUNT):
        number_of_gens.append(i)

    plt.plot(number_of_gens, best_chi_squared, label='best', color='blue')
    plt.xlabel("Generation Number")
    plt.ylabel("$\\chi^2$ value")

    plt.xticks(np.arange(0, GENERATION_COUNT, step=GRAPH_STEP))
    plt.ylim([0, 10000])
    plt.legend()
    plt.savefig(time_str + "/generations_chi.png")
    plt.show()


def merge_sort(population, chi_squared_values):
    if len(population) <= 1:
        return population, chi_squared_values

    split_point = ceil(len(population) / 2)
    pop_l = population[:split_point]
    pop_r = population[split_point:]
    chi_l = chi_squared_values[:split_point]
    chi_r = chi_squared_values[split_point:]

    pop_l, chi_l = merge_sort(pop_l, chi_l)
    pop_r, chi_r = merge_sort(pop_r, chi_r)

    return merge_lists(pop_l, chi_l, pop_r, chi_r)


def merge_lists(pop_l, chi_l, pop_r, chi_r):
    pop = []
    chi = []

    while chi_l and chi_r:
        if chi_l[0] < chi_r[0]:
            pop.append(pop_l[0])
            chi.append(chi_l[0])
            pop_l.remove(pop_l[0])
            chi_l.remove(chi_l[0])
        else:
            pop.append(pop_r[0])
            chi.append(chi_r[0])
            pop_r.remove(pop_r[0])
            chi_r.remove(chi_r[0])

    while chi_l:
        pop.append(pop_l[0])
        chi.append(chi_l[0])
        pop_l.remove(pop_l[0])
        chi_l.remove(chi_l[0])

    while chi_r:
        pop.append(pop_r[0])
        chi.append(chi_r[0])
        pop_r.remove(pop_r[0])
        chi_r.remove(chi_r[0])

    return pop, chi


def start():
    #  create the initial population
    population = []
    for _ in range(POPULATION_SIZE):
        population.append(create_chromosome())

    # start the generation loop
    for i in range(GENERATION_COUNT):
        print("Generation: " + str(i + 1))
        file_f.write("Generation: " + str(i + 1) + "\n")
        print("Population Size: " + str(len(population)))
        file_f.write("Population Size: " + str(len(population)) + "\n")
        start_time = time()  # get the start time
        population_chi_values = []
        next_generation = []
        best_ten = []
        #  evaluate the population
        for pop in population:
            try:
                population_chi_values.append(fitness_function(pop))
            except TypeError:
                population_chi_values.append(fitness_function(pop))
                print('wtf')

        population, population_chi_values = merge_sort(population, population_chi_values)
        best_chi_squared.append(population_chi_values[0])

        for j in range(10):
            best_ten.append(population[j])

        #  prune the population
        while len(population) > POPULATION_SIZE:
            population.remove(population[len(population) - 1])
            population_chi_values.remove(population_chi_values[len(population_chi_values) - 1])

        selection_amount = ceil((len(population) / 100) * SELECTION_RATE)
        parents = population[:selection_amount]

        for _ in range(selection_amount):
            index = randint(selection_amount, len(population) - 1)
            parents.append(population[index])
            population.remove(population[index])

        shuffle(parents)

        for j in range(0, len(parents), 2):
            child = crossover(parents[j], parents[j + 1])
            next_generation.append(child)
            next_generation.append(parents[j])
            next_generation.append(parents[j + 1])

        mutated = []
        for pop in next_generation:
            roll = randint(0, 100)
            if roll <= MUTATION_RATE:
                mutated.append(mutate(pop))

        for pop in mutated:
            next_generation.append(pop)

        end_time = time()

        print("Best chi^2: " + str(population_chi_values[0]))
        file_f.write("Best chi^2: " + str(population_chi_values[0]) + "\n")
        print("Worst chi^2: " + str(population_chi_values[len(population_chi_values) - 1]))
        file_f.write("Worst chi^2: " + str(population_chi_values[len(population_chi_values) - 1]) + "\n")
        print("Time taken: " + str(end_time - start_time))
        file_f.write("Time taken: " + str(end_time - start_time) + "\n")
        print("Best 10:")
        file_f.write("Best 10:" + "\n")
        for j in range(10):
            print(chromosome_to_string(best_ten[j]) + " | " + str(population_chi_values[j]))
            file_f.write(chromosome_to_string(best_ten[j]) + " | " + str(population_chi_values[j]) + "\n")
        print("------------------")
        file_f.write("------------------" + "\n")
        population = []
        for pop in next_generation:
            population.append(pop)

    population_chi_values = []
    for pop in population:
        population_chi_values.append(fitness_function(pop))

    population, population_chi_values = merge_sort(population, population_chi_values)

    for i in range(10):
        graph_chromosome(population[i], i + 1)

    graph_best_chi()
    file_f.close()


read_dataset()
calculate_float_bit_length()
calculate_term_bit_length()
start()

# chromosome = create_chromosome()
# print(chromosome_to_string(chromosome))
# print(evaluate_chromosome(chromosome, 1))
# fitness_function(chromosome)
