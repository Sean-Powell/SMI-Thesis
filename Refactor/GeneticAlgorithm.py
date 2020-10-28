from Refactor.Expression import Expression
from Refactor.Term import Term
from math import sin, cos, tan, log, pow, ceil, e
from random import randint
from time import time
import matplotlib.pyplot as plt
import numpy as np
import os

POP_SIZE = 500
MUTATION_RATE = 10
NUMBER_OF_GENERATIONS = 100
SELECTION_RATE = 10
CROSSOVER_RATE = 50

functions = {"sin": sin, "cos": cos, "tan": tan, "ln": log}
grammar = ['sin()', 'cos()', 'tan()', 'ln()', '']

y_values = []
x_values = []
s_values = []

best = []
number_of_gens = []

time_str = str(time())
os.mkdir(time_str)


def load_data_set():
    file = open("../SCPUnion2.1_mu_vs_z.txt")
    i = 0
    for line in file:
        if i < 5:
            i += 1
        else:
            line = line.strip()
            line_split = line.split("\t")
            x_values.append(float(line_split[1]))
            y_values.append(float(line_split[2]))
            s_values.append(float(line_split[3]))


def fitness_function(function):
    n = len(x_values) - 1
    chi2 = 0
    for i in range(1, n):
        chi2 += pow(((y_values[i] - function.evaluate(x_values[i])) / s_values[i]), 2)
    return chi2


# single point crossover function
def crossover(parent_1: Expression, parent_2: Expression):
    par_1_terms = parent_1.terms
    par_2_terms = parent_2.terms

    par_1_len = len(par_1_terms)
    par_2_len = len(par_2_terms)

    if par_1_len > par_2_len:
        smallest_len = par_2_len
        largest_len = par_1_len
    else:
        smallest_len = par_1_len
        largest_len = par_2_len

    crossover_point = randint(0, smallest_len)

    child_1_terms = []
    child_2_terms = []

    for i in range(largest_len):
        if i < crossover_point:
            child_1_terms.append(par_1_terms[i])
            child_2_terms.append(par_2_terms[i])
        else:
            child_1_terms.append(par_2_terms[i])
            child_2_terms.append(par_1_terms[i])

    child_1 = Expression(functions, grammar)
    child_1.set_terms(child_1_terms)
    child_2 = Expression(functions, grammar)
    child_2.set_terms(child_2_terms)

    return child_1, child_2


def mutation(function: Expression):  # todo test this function

    chance = randint(0, 1)
    if chance == 0 and len(function.terms) > 1:
        index = randint(0, len(function.terms) - 1)
        function.terms.remove(function.terms[index])
    elif len(function.terms) < function.max_size:
        sign = randint(0, 1)
        operation = function.grammar[randint(0, len(grammar) - 1)]
        exponent = randint(1, function.max_exponent)
        coefficient = randint(1, function.max_coefficient)
        function.terms.append(Term(sign, operation, exponent, coefficient))

    index = randint(0, (len(function.terms) - 1))
    term = function.terms[index]

    chance = randint(0, 1)
    if chance == 0:
        term.set_coefficient(term.coefficient - 1)
    elif chance == 1:
        term.set_coefficient(term.coefficient + 1)

    chance = randint(0, 1)
    if chance == 0:
        term.set_exponent(term.exponent - 1)
    elif chance == 1:
        term.set_exponent(term.exponent + 1)

    chance = randint(0, 1)
    if chance == 0:
        term.flip_sign()

    return function


def plot_function(function: Expression, rank):
    rank = rank + 1
    plt.plot(x_values, y_values, 'o')
    plt.errorbar(x_values, y_values, yerr=s_values, fmt=' ')

    func_x = np.linspace(0.000001, 1.4, 100)
    func_y = []
    for i in func_x:
        func_y.append(function.evaluate(i))

    plt.plot(func_x, func_y, c='r', label=function.get_string())
    plt.xlabel("Redshift")
    plt.ylabel("Distance modulus")
    plt.title("Genetic Algorithm Rank:" + str(rank))
    plt.savefig(time_str + "/" + str(rank) + ".png")
    plt.show()


def start():
    # create initial population
    population = []
    total_time = 0

    for p in range(POP_SIZE):
        population.append(Expression(functions, grammar))

    # creating and writing initial conditions to file:

    file = open(time_str + "/final_output.txt", "w")
    file.write("Population size: " + str(POP_SIZE) + "\n")
    file.write("Number of generations: " + str(NUMBER_OF_GENERATIONS) + "\n")
    file.write("Mutation Chance: " + str(MUTATION_RATE) + "\n")
    file.write("Crossover Chance: " + str(CROSSOVER_RATE) + "\n")
    file.write("Selection Chance: " + str(SELECTION_RATE) + "\n")
    file.write("---------------------\n")

    # main generation loop
    for i in range(NUMBER_OF_GENERATIONS):
        start_time = time()
        print("--- Generation", i, "---")
        print("Size of population", len(population))

        # evaluate the current population and sort the current population so the lowest (best) is first in the list
        for j in population:
            j.set_chi_2(fitness_function(j))

        population.sort(key=lambda l: l.chi_2, reverse=False)

        # select the fittest functions to be parents

        select_amount = ceil((len(population) / 100) * SELECTION_RATE)
        # select_amount = ceil((POP_SIZE / 100) * SELECTION_RATE)

        if select_amount % 2 != 0:
            select_amount += 1

        parents = []
        new_generation = []

        for j in range(select_amount):
            parents.append(population[j])

        # select random number of functions from the rest to add to parents

        selected = []
        for j in range(select_amount):
            while True:
                index = randint(select_amount + 1, (len(population) - 1))
                if index not in selected:
                    selected.append(index)
                    parents.append(population[index])
                    break

        # create children of parents

        for j in range(0, (len(parents)) - 1, 2):
            chance = randint(0, 100)

            child_1, child_2 = crossover(parents[j], parents[j + 1])
            chance = randint(0, 100)
            if chance <= MUTATION_RATE:
                new_generation.append(mutation(child_1))
            new_generation.append(child_1)
            chance = randint(0, 100)
            if chance <= MUTATION_RATE:
                new_generation.append(mutation(child_2))
            new_generation.append(child_2)

            chance = randint(0, 100)
            if chance <= MUTATION_RATE:
                new_generation.append(mutation(parents[j]))
            new_generation.append(parents[j])
            chance = randint(0, 100)
            if chance <= MUTATION_RATE:
                new_generation.append(mutation(parents[j + 1]))
            new_generation.append(parents[j + 1])

        population = new_generation

        if NUMBER_OF_GENERATIONS - 1 == i:

            print("-----------")
            print("Total time taken:", total_time, "seconds")
            print("Best 10")
            population.sort(key=lambda l: l.chi_2, reverse=False)

            plt.plot(number_of_gens, best, label='best', color='blue')
            # plt.plot(number_of_gens, worst, label='worst', color='red')
            # plt.plot(number_of_gens, average, label='average', color='black')

            plt.xlabel("Generation Number")
            plt.ylabel("$\\chi^2$ value")

            plt.xticks(np.arange(0, NUMBER_OF_GENERATIONS, step=1))

            plt.legend()
            plt.savefig(time_str + "/Generations_chi.png")
            plt.show()

            n = len(population) if 10 > len(population) else 10
            for j in range(n):
                print("----------")
                file.write(population[j].get_string() + "\n")
                population[j].print()
                print(population[j].get_chi_2())
                file.write(str(population[j].get_chi_2()) + "\n")
                plot_function(population[j], j)

            file.close()
        else:
            n = len(population) if 10 > len(population) else 10
            for j in range(n):
                file.write(population[j].get_string() + " | " + str(population[j].get_chi_2()) + "\n")
                print(population[j].get_string() + " | " + str(population[j].get_chi_2()))

            population = new_generation
            end_time = time()
            generation_time = end_time - start_time
            total_time += generation_time
            file.write("Generation took " + str(generation_time) + " seconds\n")

            print("Generation took", generation_time, "seconds")
            print("----------")
    print("Finished")

load_data_set()
start()