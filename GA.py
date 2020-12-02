from Expression import Expression
from math import sin, cos, tan, log, pow, ceil, e
from random import shuffle, randint
from time import time
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import threading
import queue

POPULATION_SIZE = 500
MUTATION_CHANCE = 40
NUMBER_OF_GENERATIONS = 5
SELECTION_RATE = 25
NUMBER_OF_CHI_THREADS = 64
NUMBER_OF_REPRODUCTION_THREADS = 4
NUMBER_OF_MUTATION_THREADS = 16

functions = {"sin": sin, "cos": cos, "tan": tan, "ln": log, "e": e}
grammar = ['sin()', 'cos()', 'tan()', 'ln()', 'e[]', '']

y = []
x = []
s = []

best = []
worst = []
number_of_gens = []
processed = []
new_generation = []

time_str = str(time())
os.mkdir(time_str)

chiQueueLock = threading.Lock()
chiWorkQueue = queue.Queue()
reproductionQueueLock = threading.Lock()
reproductionWorkQueue = queue.Queue()
mutationQueueLock = threading.Lock()
mutationWorkQueue = queue.Queue()

exitFlag = False
chi_threads = []
repro_threads = []
mutation_threads = []


class GeneChiThread (threading.Thread):
    def __init__(self, threadID, q):
        threading.Thread.__init__(self)
        self.q = q
        self.threadID = threadID

    def run(self):
        print("Starting chi thread " + str(self.threadID) + "...")
        process_genes_chi(self.q)
        print("Finished chi thread " + str(self.threadID))


def process_genes_chi(q):
    while not exitFlag:
        chiQueueLock.acquire()
        if not chiWorkQueue.empty():
            data = q.get()
            chiQueueLock.release()
            data.set_chi_2(fitness_function(data))
            processed.append(copy.deepcopy(data))
        else:
            chiQueueLock.release()


class GeneReproductionThread (threading.Thread):
    def __init__(self, threadID, q):
        threading.Thread.__init__(self)
        self.q = q
        self.threadID = threadID

    def run(self):
        print("Starting reproduction thread " + str(self.threadID) + "...")
        process_genes_reproduction(self.q)
        print("Finished reproduction thread " + str(self.threadID))


def process_genes_reproduction(q):
    while not exitFlag:
        reproductionQueueLock.acquire()
        if not reproductionWorkQueue.empty():
            parent_a = q.get()
            parent_b = q.get()
            reproductionQueueLock.release()
            child_a, child_b, child_c = crossover(parent_a, parent_b)
            new_generation.append(copy.deepcopy(child_a))
            new_generation.append(copy.deepcopy(child_b))
            new_generation.append(copy.deepcopy(child_c))
            new_generation.append(copy.deepcopy(parent_a))
            new_generation.append(copy.deepcopy(parent_b))
        else:
            reproductionQueueLock.release()


class MutationThread(threading.Thread):
    def __init__(self, threadID, q):
        threading.Thread.__init__(self)
        self.q = q
        self.threadID = threadID

    def run(self):
        print("Starting mutate thread " + str(self.threadID) + "...")
        process_genes_mutation(self.q)
        print("Finished mutate thread " + str(self.threadID))


def process_genes_mutation(q):
    # obj = copy.deepcopy(new_generation[j])
    # obj.mutate()
    # if new_generation[j].get_string() != obj.get_string():
    #     new_generation.append(copy.deepcopy(obj))
    while not exitFlag:
        mutationQueueLock.acquire()
        if not mutationWorkQueue.empty():
            obj = q.get()
            mutationQueueLock.release()
            mutated = obj.mutate()
            if mutated:
                new_generation.append(copy.deepcopy(obj))
        else:
            mutationQueueLock.release()


def load_data_set():
    file = open("SCPUnion_mu_vs_z.txt")
    i = 0
    for line in file:
        if i < 4:
            i += 1
        else:
            line = line.strip()
            line_split = line.split("\t")
            x.append(float(line_split[1]))
            y.append(float(line_split[2]))
            s.append(float(line_split[3]))


def fitness_function(func : Expression):
    n = len(x)
    chi2 = 0
    for index in range(1, n):
        try:
            chi2 += pow(((y[index] - func.evaluate(x[index])) / s[index]), 2)
        except Exception:
            print(func.get_string())
    return chi2


def get_terms(index, candidate_1: Expression, candidate_2: Expression):
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


def create_scatter_plot(function: Expression, rank):
    rank = rank + 1
    plt.plot(x, y, 'o')
    plt.errorbar(x, y, yerr=s, fmt=' ')

    func_x = np.linspace(min(x), max(x), 100)
    func_y = []
    for i in func_x:
        func_y.append(function.evaluate(i))

    z = np.linspace(min(x), max(x), 100)
    mu = np.log((z**2.17145) * ((-z ** 2.82) + z + np.exp(z))) + 42.83 - 5. * np.log10(0.7)
    plt.plot(z, mu, c='g')
    plt.xlim(0, max(x))

    plt.plot(func_x, func_y, c='r', label=function.get_string())
    plt.xlabel("Redshift")
    plt.ylabel("Distance modulus")
    plt.title("Genetic Algorithm Rank:" + str(rank))
    plt.savefig(time_str + "/" + str(rank) + ".png")
    plt.show()


def start():
    global processed, exitFlag, new_generation
    population = []
    total_time = 0

    # create threads
    for i in range(NUMBER_OF_CHI_THREADS):
        thread = GeneChiThread(i, chiWorkQueue)
        thread.start()
        chi_threads.append(thread)

    for i in range(NUMBER_OF_REPRODUCTION_THREADS):
        thread = GeneReproductionThread(i, reproductionWorkQueue)
        thread.start()
        repro_threads.append(thread)

    for i in range(NUMBER_OF_MUTATION_THREADS):
        thread = MutationThread(i, mutationWorkQueue)
        thread.start()
        mutation_threads.append(thread)

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

        exitFlag = False
        processed = []

        # Evaluating the populations chi^2
        chi_start = time()
        chiQueueLock.acquire()

        for j in population:
            chiWorkQueue.put(j)

        chiQueueLock.release()

        while not chiWorkQueue.empty():
            pass

        chi_end = time()
        population = processed

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

        # get the select_amount of best parents
        for j in range(select_amount):
            parents.append(copy.deepcopy(population[j]))

        # get a select_amount of random parents that are not already parents
        for j in range(select_amount):
            selected = False
            while not selected:
                index = randint(select_amount, (len(population) - 1))
                if index not in parents_index:
                    parents.append(copy.deepcopy(population[index]))
                    parents_index.append(index)
                    selected = True

        # shuffle the list so parents are matched up randomly
        shuffle(parents)

        repro_start = time()
        # loop through the parents while creating the offspring for the next generation
        reproductionQueueLock.acquire()
        for parent in parents:
            reproductionWorkQueue.put(parent)

        reproductionQueueLock.release()

        while not reproductionWorkQueue.empty():
            pass

        repro_end = time()

        mutation_start = time()
        mutationQueueLock.acquire()
        for j in range(len(new_generation) - 1):
            mutationWorkQueue.put(copy.deepcopy(new_generation[j]))

        mutationQueueLock.release()

        while not mutationWorkQueue.empty():
            pass

        mutation_end = time()

        print("Best chi^2:", population[0].get_chi_2())
        best.append(population[0].get_chi_2())
        print("worst chi^2:", population[len(population) - 1].get_chi_2())
        worst.append(population[len(population) - 1].get_chi_2())
        number_of_gens.append(i)

        if NUMBER_OF_GENERATIONS - 1 == i:

            print("-----------")
            print("Total time taken:", total_time, "seconds")
            print("Best 10")
            population.sort(key=lambda l: l.chi_2, reverse=False)

            plt.plot(number_of_gens, best, label='best', color='blue')

            plt.xlabel("Generation Number")
            plt.ylabel("$\\chi^2$ value")

            plt.xticks(np.arange(0, NUMBER_OF_GENERATIONS + 1, step=50))
            plt.ylim([0, 10000])
            plt.legend()
            plt.savefig(time_str + "/Generations_chi.png")
            plt.show()

            n = len(population) if 10 > len(population) else 10
            file.write("Total time taken:" + str(total_time) + "seconds\n")
            file.write("Best 10:\n")
            for j in range(n):
                print("----------")
                file.write(population[j].get_string() + "\n")
                population[j].print()
                print(population[j].get_chi_2())
                file.write(str(population[j].get_chi_2()) + "\n")
                create_scatter_plot(population[j], j)

            file.close()
            exitFlag = True

            for t in chi_threads:
                t.join()

            for t in repro_threads:
                t.join()

        else:
            n = len(population) if 10 > len(population) else 10
            for j in range(n):
                file.write(population[j].get_string() + " | " + str(population[j].get_chi_2()) + "\n")
                print(population[j].get_string() + " | " + str(population[j].get_chi_2()))

            population = []
            for j in new_generation:
                population.append(copy.deepcopy(j))
            end_time = time()
            generation_time = end_time - start_time
            total_time += generation_time
            file.write("Generation took " + str(generation_time) + " seconds\n")
            print("Chi^2 time:", (chi_end - chi_start))
            print("Reproduction time:", (repro_end - repro_start))
            print("Mutation time:", (mutation_end - mutation_start))
            print("Generation took", generation_time, "seconds")
            print("----------")


load_data_set()
start()
