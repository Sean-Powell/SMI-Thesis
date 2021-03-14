import threading
import queue
from copy import deepcopy
from random import shuffle, randint
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from PolyExpression import PolyExpression
from time import time
import os

POPULATION_SIZE = 500
MUTATION_RATE = 40
NUMBER_OF_GENERATIONS = 6000
SELECTION_RATE = 25

MAX_COEFFICIENT = 100
MAX_EXPONENT = 5
MIN_LENGTH = 1
MAX_LENGTH = 6
MAX_MUTATION_SIZE = 100

NUMBER_OF_CHI_THREADS = 64
NUMBER_OF_REPRODUCTION_THREADS = 4
NUMBER_OF_MUTATION_THREADS = 16

DATASET_PATH = "SCPUnion_mu_vs_z.txt"

x = []
y = []
s = []

# save file stuff
time_str = str(time())
os.mkdir(time_str)
file_f = open(time_str + "/output.txt", "w")
number_of_gens = []
best_chi_squared = []
# thread stuff
exitFlag = False
chiSquaredQueue = queue.Queue()
chiSquaredLock = threading.Lock()
chiSquaredThreads = []

reproductionQueue = queue.Queue()
reproductionLock = threading.Lock()
reproductionThreads = []

processed = []
new_generation = []

# thread classes

class ChiSquaredThread(threading.Thread):
    def __init__(self, threadID, q):
        threading.Thread.__init__(self)
        self.q = q
        self.threadID = threadID

    def run(self):
        print("Starting chi thread " + str(self.threadID) + "...")
        process_genes_chi(self.q)
        print("Finished chi thread " + str(self.threadID))


class ReproductionThread(threading.Thread):
    def __init__(self, threadID, q):
        threading.Thread.__init__(self)
        self.q = q
        self.threadID = threadID

    def run(self):
        print("Starting reproduction thread " + str(self.threadID) + "...")
        process_reproduction(self.q)
        print("Finished reproduction thread " + str(self.threadID))

# thread functions


def process_genes_chi(q):
    while not exitFlag:
        chiSquaredLock.acquire()
        if not chiSquaredQueue.empty():
            data = q.get()
            chiSquaredLock.release()
            data.set_chi_squared(fitness_function(data))
            processed.append(deepcopy(data))
        else:
            chiSquaredLock.release()


def process_reproduction(q):
    while not exitFlag:
        reproductionLock.acquire()
        if not reproductionQueue.empty():
            parent_a = q.get()
            parent_b = q.get()
            reproductionLock.release()
            child_a, child_b = crossover(parent_a, parent_b)
            new_generation.append(deepcopy(child_a))
            new_generation.append(deepcopy(child_b))
            new_generation.append(deepcopy(parent_a))
            new_generation.append(deepcopy(parent_b))
        else:
            reproductionLock.release()
# ga functions


def read_dataset():
    f = open(DATASET_PATH)
    i = 0
    for line in f:
        if i < 4:
            # skip lines before data starts
            i += 1
        else:
            line = line.strip()
            line_split = line.split("\t")
            x.append(float(line_split[1]))
            y.append(float(line_split[2]))
            s.append(float(line_split[3]))


def plot_scatter_graph(func: PolyExpression, index):
    plt.plot(x, y, 'o')
    plt.errorbar(x, y, yerr=s, fmt=' ')

    func_x = np.linspace(0, max(x), 100)
    func_y = []
    for i in func_x:
        func_y.append(func.evaluate(i))

    z = np.linspace(min(x), max(x), 100)
    mu = np.log((z**2.17145) * ((-z ** 2.82) + z + np.exp(z))) + 42.83 - 5. * np.log10(0.7)
    plt.plot(z, mu, c='g')
    plt.plot(func_x, func_y, c='r')
    plt.xlim(0, max(x))
    plt.xlabel("Redshift")
    plt.ylabel("Distance modulus")
    plt.title("Genetic Algorithm Rank:" + str(index))
    plt.savefig(time_str + "/" + str(index) + ".png")

    plt.show()


def plot_generation_graph():
    plt.plot(number_of_gens, best_chi_squared, label='best', color='blue')

    plt.xlabel("Generation Number")
    plt.ylabel("$\\chi^2$ value")

    plt.xticks(np.arange(1, NUMBER_OF_GENERATIONS + 1, step=500))
    # plt.ylim([0, 10000])
    plt.legend()
    plt.savefig(time_str + "/generations_chi.png")
    plt.show()


def fitness_function(func: PolyExpression):
    chi_2 = 0
    for index in range(0, len(x)):
        chi_2 += pow(((y[index] - func.evaluate(x[index])) / s[index]), 2)
    return chi_2


def create_expression():
    return PolyExpression(max_length=MAX_LENGTH, min_length=MIN_LENGTH, max_coefficient=MAX_COEFFICIENT,
                          max_exponent=MAX_EXPONENT, mutation_rate=MUTATION_RATE, max_mutation_size=MAX_MUTATION_SIZE)


def create_population():
    population = []
    for i in range(POPULATION_SIZE):
        population.append(deepcopy(create_expression()))
    return population


def evaluate_population(population):
    chiSquaredLock.acquire()
    for i in range(POPULATION_SIZE):
        chiSquaredQueue.put(deepcopy(population[i]))

    chiSquaredLock.release()

    while not chiSquaredQueue.empty():
        pass

    return deepcopy(processed)


def prune_population(population):
    while len(population) - 1 >= POPULATION_SIZE:
        population.remove(population[len(population) - 1])
    return population


def crossover(parent_a: PolyExpression, parent_b: PolyExpression):
    len_a = len(parent_a.terms)
    len_b = len(parent_b.terms)
    terms_a = []
    terms_b = []
    if len_a == len_b:
        crossover_point = randint(0, ceil(len_a / 2))
        for i in range(len_a):
            if i <= crossover_point:
                terms_a.append(parent_a.terms[i])
                terms_b.append(parent_b.terms[i])
            else:
                terms_a.append(parent_b.terms[i])
                terms_b.append(parent_a.terms[i])
    else:
        if len_a < len_b:
            for i in range(len_b):
                if i < len_a:
                    terms_a.append(parent_a.terms[i])
                    terms_b.append(parent_b.terms[i])
                else:
                    terms_a.append(parent_b.terms[i])
        else:
            for i in range(len_a):
                if i < len_b:
                    terms_a.append(parent_a.terms[i])
                    terms_b.append(parent_b.terms[i])
                else:
                    terms_b.append(parent_a.terms[i])

    child_1 = deepcopy(create_expression())
    child_1.set_terms(terms_a)
    child_2 = deepcopy(create_expression())
    child_2.set_terms(terms_b)

    return child_1, child_2


def simulate_generation(population, generation_number):
    global new_generation
    global processed
    start_time = time()
    processed = []
    new_generation = []
    processed_population = deepcopy(evaluate_population(population))
    processed_population.sort(key=lambda l: l.get_chi_squared(), reverse=False)
    processed_population = deepcopy(prune_population(processed_population))

    selection_amount = ceil((len(processed_population) / 100) * SELECTION_RATE)
    parents = processed_population[:selection_amount]

    for i in range(selection_amount):
        index = randint(selection_amount, len(processed_population) - 1)
        parents.append(processed_population[index])
        processed_population.remove(processed_population[index])

    shuffle(parents)  # shuffle to increase diversity in reproduction
    reproductionLock.acquire()
    for p in parents:
        reproductionQueue.put(deepcopy(p))
    reproductionLock.release()

    while not reproductionQueue.empty():
        pass

    next_generation = []
    for i in new_generation:
        next_generation.append(deepcopy(i))

    # todo can be combined into the above loop
    for p in next_generation:
        mutation = deepcopy(p)
        mutated = mutation.mutate()
        if mutated:
            next_generation.append(deepcopy(mutation))

    end_time = time()
    time_taken = end_time - start_time
    population_summary(processed_population, generation_number, time_taken)
    return next_generation


def population_summary(population, generation_number, time_taken):
    if len(population) < 10:
        r = len(population)
    else:
        r = 10

    best_chi_squared.append(population[0].get_chi_squared())
    print("----------------------------")
    file_f.write("----------------------------" + "\n")
    print("GENERATION: " + str(generation_number))
    file_f.write("GENERATION: " + str(generation_number) + "\n")
    print("Best Chi Squared: " + str(population[0].get_chi_squared()))
    file_f.write("Best Chi Squared: " + str(population[0].get_chi_squared()) + "\n")
    print("Worst Chi Squared: " + str(population[len(population) - 1].get_chi_squared()))
    file_f.write("Worst Chi Squared: " + str(population[len(population) - 1].get_chi_squared()) + "\n")
    print("Time Taken: " + str(time_taken))
    file_f.write("Time Taken: " + str(time_taken) + "\n")
    print("Best 10 of generation:")
    file_f.write("Best 10 of generation:" + "\n")
    for i in range(1, (r + 1)):
        print(str(i) + ": " + population[i].get_equation('x') + " | " + str(population[i].get_chi_squared()))
        file_f.write(str(i) + ": " + population[i].get_equation('x') + " | " + str(population[i].get_chi_squared()) +
                     "\n")


def start():
    global file_f
    for i in range(NUMBER_OF_CHI_THREADS):
        thread = ChiSquaredThread(i, chiSquaredQueue)
        thread.start()
        chiSquaredThreads.append(thread)

    for i in range(NUMBER_OF_REPRODUCTION_THREADS):
        thread = ReproductionThread(i, reproductionQueue)
        thread.start()
        reproductionThreads.append(thread)

    read_dataset()
    population = deepcopy(create_population())

    for i in range(1, NUMBER_OF_GENERATIONS + 1):
        population = deepcopy(simulate_generation(population, i))
        number_of_gens.append(i)

    for i in range(1, 11):
        plot_scatter_graph(population[i], i)

    plot_generation_graph()
    file_f.close()


start()
