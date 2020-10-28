from Expression import Expression
from math import sin, cos, tan, log, pow, ceil
from random import randint, shuffle
from time import time
import matplotlib.pyplot as plt
import numpy as np
import os
import copy

POPULATION_SIZE = 500
MUTATION_CHANCE = 20
NUMBER_OF_GENERATIONS = 250
SELECTION_RATE = 25

functions = {"sin": sin, "cos": cos, "tan": tan, "ln": log}
grammar = ['sin()', 'cos()', 'tan()', 'ln()', '']

y = []
x = []
s = []

best = []
average = []
worst = []
number_of_gens = []

time_str = str(time())
os.mkdir(time_str)


def load_data_set():
    file = open("SCPUnion2.1_mu_vs_z.txt")
    i = 0
    for line in file:
        if i < 5:
            i += 1
        else:
            line = line.strip()
            line_split = line.split("\t")
            x.append(float(line_split[1]))
            y.append(float(line_split[2]))
            s.append(float(line_split[3]))


def fitness_function(func):
    n = len(x) - 1  # todo check if this -1 is needed when given the data set
    chi2 = 0
    for index in range(1, n):
        chi2 += pow(((y[index] - func.evaluate(x[index])) / s[index]), 2)
    return chi2


def get_terms(index, candidate_1: Expression, candidate_2: Expression):
    # terms = []
    # for _ in range(index):
    #     chance = randint(0, 1)
    #     if chance == 0:
    #         term = candidate_1.terms[randint(0, (len(candidate_1.terms) - 1))]
    #         chance = randint(0, 1)
    #         if chance == 1:
    #             term.exponent = candidate_2.terms[randint(0, len(candidate_2.terms) - 1)].exponent
    #         chance = randint(0, 1)
    #         if chance == 1:
    #             term.coefficient = candidate_2.terms[randint(0, len(candidate_2.terms) - 1)].coefficient
    #     else:
    #         term = candidate_2.terms[randint(0, (len(candidate_2.terms) - 1))]
    #         chance = randint(0, 1)
    #         if chance == 1:
    #             term.exponent = candidate_1.terms[randint(0, len(candidate_1.terms) - 1)].exponent
    #         chance = randint(0, 1)
    #         if chance == 1:
    #             term.coefficient = candidate_1.terms[randint(0, len(candidate_1.terms) - 1)].coefficient
    #     terms.append(copy.deepcopy(term))
    # return terms

    terms = []
    new_terms = []
    candidate_1_terms = candidate_1.terms
    candidate_2_terms = candidate_2.terms
    terms.extend(candidate_1_terms)
    terms.extend(candidate_2_terms)
    shuffle(terms)

    for i in range(index):
        term_index = randint(0, len(terms) - 1)
        new_terms.append(copy.deepcopy(terms[term_index]))
        terms.remove(terms[term_index])

    return new_terms


def crossover(parent_1: Expression, parent_2: Expression):
    n = len(parent_1.terms)
    j = len(parent_2.terms)
    k = ceil((n + j) / 2)

    if n == j:
        k = n * 2

    child_1 = Expression(functions, grammar, mutation_chance=MUTATION_CHANCE)
    child_1.set_terms(get_terms(n, parent_1, parent_2))
    child_2 = Expression(functions, grammar, mutation_chance=MUTATION_CHANCE)
    child_2.set_terms(get_terms(j, parent_1, parent_2))
    child_3 = Expression(functions, grammar, mutation_chance=MUTATION_CHANCE)
    child_3.set_terms(get_terms(k, parent_1, parent_2))

    if child_1.get_string() == child_2.get_string():
        child_1.set_mutation_rate(100)
        child_1.mutate()
        child_1.set_mutation_rate(MUTATION_CHANCE)
    if child_2.get_string() == child_3.get_string():
        child_2.set_mutation_rate(100)
        child_2.mutate()
        child_2.set_mutation_rate(MUTATION_CHANCE)
    if child_1.get_string() == child_3.get_string():
        child_3.set_mutation_rate(100)
        child_3.mutate()
        child_3.set_mutation_rate(MUTATION_CHANCE)

    if child_1.get_string() == parent_1.get_string():
        child_1.set_mutation_rate(100)
        child_1.mutate()
        child_1.set_mutation_rate(MUTATION_CHANCE)

    if child_1.get_string() == parent_2.get_string():
        child_1.set_mutation_rate(100)
        child_1.mutate()
        child_1.set_mutation_rate(MUTATION_CHANCE)

    if child_2.get_string() == parent_1.get_string():
        child_2.set_mutation_rate(100)
        child_2.mutate()
        child_2.set_mutation_rate(MUTATION_CHANCE)

    if child_2.get_string() == parent_2.get_string():
        child_2.set_mutation_rate(100)
        child_2.mutate()
        child_2.set_mutation_rate(MUTATION_CHANCE)

    if child_3.get_string() == parent_1.get_string():
        child_3.set_mutation_rate(100)
        child_3.mutate()
        child_3.set_mutation_rate(MUTATION_CHANCE)

    if child_3.get_string() == parent_2.get_string():
        child_3.set_mutation_rate(100)
        child_3.mutate()
        child_3.set_mutation_rate(MUTATION_CHANCE)

    return child_1, child_2, child_3


def one_point_cross(parent_1: Expression, parent_2: Expression):
    par_1_terms = parent_1.terms
    par_2_terms = parent_2.terms

    par_1_len = len(par_1_terms)
    par_2_len = len(par_2_terms)

    child_1_terms = []
    child_2_terms = []

    if par_1_len == par_2_len:
        # print("Equal cross")
        crossover_point = randint(1, par_1_len - 1)
        # print("crossover point", crossover_point)
        for i in range(crossover_point):
            child_1_terms.append(copy.deepcopy(par_1_terms[i]))
            child_2_terms.append(copy.deepcopy(par_2_terms[i]))
        for i in range(crossover_point, par_1_len):
            child_1_terms.append(copy.deepcopy(par_2_terms[i]))
            child_2_terms.append(copy.deepcopy(par_1_terms[i]))
    elif par_1_len > par_2_len:
        # print("one cross")
        crossover_point = randint(1, par_2_len)
        for i in range(crossover_point):
            child_1_terms.append(par_1_terms[i])
            child_2_terms.append(par_2_terms[i])
        for i in range(crossover_point, par_2_len):
            child_1_terms.append(par_2_terms[i])
            child_2_terms.append(par_1_terms[i])

        for i in range(par_2_len, par_1_len):
            child_1_terms.append(par_1_terms[i])
            child_2_terms.append(par_1_terms[i])
    else:
        # print("two cross")
        crossover_point = randint(1, par_1_len)
        for i in range(crossover_point):
            child_1_terms.append(par_1_terms[i])
            child_2_terms.append(par_2_terms[i])
        for i in range(crossover_point, par_1_len):
            child_1_terms.append(par_2_terms[i])
            child_2_terms.append(par_1_terms[i])

        for i in range(par_1_len, par_2_len):
            child_1_terms.append(par_2_terms[i])
            child_2_terms.append(par_2_terms[i])

    child_1 = copy.deepcopy(Expression(functions, grammar, mutation_chance=MUTATION_CHANCE))
    child_1.set_terms(child_1_terms)
    child_2 = copy.deepcopy(Expression(functions, grammar, mutation_chance=MUTATION_CHANCE))
    child_2.set_terms(child_2_terms)

    # check if either of the children or one of the children or parent being the same
    if child_1.get_string() == parent_1.get_string():
        child_1.set_mutation_rate(100)
        child_1.mutate()
        child_1.set_mutation_rate(MUTATION_CHANCE)

    if child_1.get_string() == parent_2.get_string():
        child_1.set_mutation_rate(100)
        child_1.mutate()
        child_1.set_mutation_rate(MUTATION_CHANCE)

    if child_2.get_string() == parent_1.get_string():
        child_2.set_mutation_rate(100)
        child_2.mutate()
        child_2.set_mutation_rate(MUTATION_CHANCE)
    if child_2.get_string() == parent_2.get_string():
        child_2.set_mutation_rate(100)
        child_2.mutate()
        child_2.set_mutation_rate(MUTATION_CHANCE)

    if child_1.get_string() == child_2.get_string():
        child_1.set_mutation_rate(100)
        child_2.mutate()
        child_1.set_mutation_rate(MUTATION_CHANCE)
    return child_1, child_2


def create_scatter_plot(function: Expression, rank):
    rank = rank + 1
    plt.plot(x, y, 'o')
    plt.errorbar(x, y, yerr=s, fmt=' ')

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
    population = []
    total_time = 0

    for p in range(POPULATION_SIZE):
        population.append(copy.deepcopy(Expression(functions, grammar, mutation_chance=MUTATION_CHANCE)))
        # population[p].set_chi_2(randint(0, POPULATION_SIZE))

    file = open(time_str + "/final_output.txt", "w")
    file.write("Population size: " + str(POPULATION_SIZE) + "\n")
    file.write("Number of generations: " + str(NUMBER_OF_GENERATIONS) + "\n")
    file.write("Mutation Chance: " + str(MUTATION_CHANCE) + "\n")
    file.write("Selection Chance: " + str(SELECTION_RATE) + "\n")
    file.write("---------------------\n")

    for i in range(NUMBER_OF_GENERATIONS):
        start_time = time()
        print("--- Generation", i + 1, "---")
        print("Size of population", len(population))
        file.write("--- Generation" + str(i + 1) + "---\n")
        file.write("Size of population" + str(len(population)) + "\n")
        chi_2_total = 0

        # Evaluating the populations chi^2
        for j in population:
            # print(j.get_string())
            chi_2 = fitness_function(j)
            chi_2_total += chi_2
            j.set_chi_2(fitness_function(j))

        # sorting the population based on its chi^2 with the lowest first
        population.sort(key=lambda l: l.chi_2, reverse=False)

        # prune the worst of the population if it is over the population size to stop exponential growth
        while len(population) - 1 > POPULATION_SIZE:
            population.remove(population[len(population) - 1])

        # calculate the select amount
        select_amount = ceil((len(population) / 100) * SELECTION_RATE)
        if select_amount % 2 != 0:
            select_amount += 1

        parents = []
        parents_index = []
        new_generation = []
        new_generation_terms = []

        # get the select_amount of best parents
        for j in range(select_amount):
            parents.append(copy.deepcopy(population[j]))
            # new_generation.append(copy.deepcopy(population[j]))
            # new_generation_terms.append(copy.copy(population[j].get_string()))

        # get a select_amount of random parents that are not already parents
        for j in range(select_amount):
            selected = False
            while not selected:
                index = randint(select_amount, (len(population) - 1))
                if index not in parents_index:
                    parents.append(copy.deepcopy(population[index]))
                    # print("Select amount of best parents appening: " + population[j].get_string() + " to next generation")
                    # new_generation.append(copy.deepcopy(population[j]))
                    parents_index.append(index)
                    selected = True

        # shuffle the list so parents are matched up randomly
        shuffle(parents)

        # loop through the parents while creating the offspring for the next generation
        for j in range(0, len(parents), 2):
            # print("-------")
            # print("Reproduction of " + parents[j].get_string() + " and  " + parents[j + 1].get_string())
            parent_a = copy.deepcopy(parents[j])
            parent_b = copy.deepcopy(parents[j + 1])
            child_a, child_b, child_c = crossover(parent_a, parent_b)
            # child_a, child_b = one_point_cross(parent_a, parent_b)
            # print("Child A: " + child_a.get_string())
            # print("Child B: " + child_b.get_string())
            # print("-------")
            # print("Appending child a to new generation: " + child_a.get_string())
            new_generation.append(copy.deepcopy(child_a))
            new_generation_terms.append(copy.copy(child_a.get_string()))
            # print("Appending child b to new generation: " + child_b.get_string())
            new_generation.append(copy.deepcopy(child_b))
            new_generation_terms.append(copy.copy(child_b.get_string()))
            new_generation.append(copy.deepcopy(child_c))
            new_generation_terms.append(copy.copy(child_c.get_string()))

            new_generation.append(copy.deepcopy(parent_a))
            new_generation.append(copy.deepcopy(parent_b))


        # a, b = one_point_cross(parents[len(parents) - 1], parents[0])
        # new_generation.append(copy.deepcopy(a))
        # new_generation.append(copy.deepcopy(b))

        # loop through the new generation rolling if mutation should occur
        for j in range(len(new_generation) - 1):
            obj = copy.deepcopy(new_generation[j])
            obj.mutate()
            if new_generation[j].get_string() != obj.get_string():
                #     print("Mutation of " + new_generation[j].get_string() + " to " + obj.get_string())
                new_generation.append(copy.deepcopy(obj))

        # for j in new_generation:
        #     chance = randint(0, 100)
        #     if chance <= MUTATION_CHANCE:
        #         j.mutation()

        print("Best chi^2:", population[0].get_chi_2())
        best.append(population[0].get_chi_2())
        print("worst chi^2:", population[len(population) - 1].get_chi_2())
        worst.append(population[len(population) - 1].get_chi_2())
        print("Average chi^2:", (chi_2_total / (len(population) - 1)))
        average.append((chi_2_total / (len(population) - 1)))
        number_of_gens.append(i)

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

            plt.xticks(np.arange(0, NUMBER_OF_GENERATIONS + 1, step=10))

            plt.legend()
            plt.savefig(time_str + "/Generations_chi.png")
            plt.show()

            n = len(population) if 10 > len(population) else 10
            # n = len(population)
            for j in range(n):
                print("----------")
                file.write(population[j].get_string() + "\n")
                population[j].print()
                print(population[j].get_chi_2())
                file.write(str(population[j].get_chi_2()) + "\n")
                create_scatter_plot(population[j], j)

            file.close()

        else:
            n = len(population) if 10 > len(population) else 10
            # n = len(population)
            for j in range(n):
                file.write(population[j].get_string() + " | " + str(population[j].get_chi_2()) + "\n")
                print(population[j].get_string() + " | " + str(population[j].get_chi_2()))

            population = []
            for j in new_generation:
                population.append(copy.deepcopy(j))
            # population = copy.deepcopy(new_generation)
            end_time = time()
            generation_time = end_time - start_time
            total_time += generation_time
            file.write("Generation took " + str(generation_time) + " seconds\n")
            if len(new_generation_terms) == len(set(new_generation_terms)):
                print("Generation contains no duplicates")
            print("Generation took", generation_time, "seconds")
            print("----------")


load_data_set()
start()
