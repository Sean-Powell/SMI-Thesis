from Expression import Expression
from math import sin, cos, tan, log, pow, ceil
from random import randint


POPULATION_SIZE = 500
MUTATION_CHANCE = 5
NUMBER_OF_GENERATIONS = 100
SELECTION_RATE = 10

functions = {"sin": sin, "cos": cos, "tan": tan, "ln": log}
grammar = ['sin()', 'cos()', 'tan()', 'ln()', '']

y = []
x = []
s = []


def fitness_function(func):
    n = len(x) - 1  # todo check if this -1 is needed when given the data set
    chi2 = 0
    for i in range(1, n):
        chi2 += pow(((y[i] - func.evaluate(x[i])) / s[i]), 2)
    return chi2


def get_terms(index, candidate_1: Expression, candidate_2: Expression):
    terms = []
    for _ in range(index):
        chance = randint(0, 1)
        if chance == 0:
            term = candidate_1.terms[randint(0, (len(candidate_1.terms) - 1))]
            chance = randint(0, 1)
            if chance == 1:
                term.exponent = candidate_2.terms[randint(0, len(candidate_2.terms) - 1)].exponent
            chance = randint(0, 1)
            if chance == 1:
                term.coefficient = candidate_2.terms[randint(0, len(candidate_2.terms) - 1)].coefficient
        else:
            term = candidate_2.terms[randint(0, (len(candidate_2.terms) - 1))]
            chance = randint(0, 1)
            if chance == 1:
                term.exponent = candidate_1.terms[randint(0, len(candidate_1.terms) - 1)].exponent
            chance = randint(0, 1)
            if chance == 1:
                term.coefficient = candidate_1.terms[randint(0, len(candidate_1.terms) - 1)].coefficient
        terms.append(term)
    return terms


def crossover(candidate_1: Expression, candidate_2: Expression):
    n = len(candidate_1.terms)
    j = len(candidate_2.terms)
    k = ceil((n + j) / 2)

    child_1 = Expression(functions, grammar, mutation_chance=MUTATION_CHANCE)
    child_1.set_terms(get_terms(n, candidate_1, candidate_2))
    child_2 = Expression(functions, grammar, mutation_chance=MUTATION_CHANCE)
    child_2.set_terms(get_terms(j, candidate_1, candidate_2))
    child_3 = Expression(functions, grammar, mutation_chance=MUTATION_CHANCE)
    child_3.set_terms(get_terms(k, candidate_1, candidate_2))

    return child_1, child_2, child_3


def start():
    population = []
    for p in range(POPULATION_SIZE):
        population.append(Expression(functions, grammar, mutation_chance=MUTATION_CHANCE))
        population[p].set_chi_2(randint(0, POPULATION_SIZE))

    for i in range(NUMBER_OF_GENERATIONS):
        for j in population:
            j.set_chi_2(fitness_function(j))

        #  todo evaluate the chi_2 on  each of the population
        population.sort(key=lambda l: l.get_chi_2(), reverse=False)

        print("sorted population")

        select_amount = POPULATION_SIZE // SELECTION_RATE
        if select_amount % 2 != 0:
            select_amount += 1

        parents = []
        new_generation = []
        for j in range(select_amount):
            parents.append(population[j])

        for j in range((select_amount - 1)):
            a, b, c = crossover(parents[j], parents[j + 1])
            new_generation.append(a)
            new_generation.append(b)
            new_generation.append(c)

        for j in new_generation:
            chance = randint(0, 100)
            if chance < MUTATION_CHANCE:
                j.mutation()


start()
