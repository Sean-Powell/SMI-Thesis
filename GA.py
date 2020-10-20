from Expression import Expression
from math import sin, cos, tan, log


POPULATION_SIZE = 500

functions = {"sin": sin, "cos": cos, "tan": tan, "ln": log}
grammar = ['sin()', 'cos()', 'tan()', 'ln()', '']

population = []
for i in range(POPULATION_SIZE):
    population.append(Expression(functions, grammar))

print("Created population")

for i in population:
    i.print()
    i.mutation()
    i.print()
    print("-----")

# todo implement reproduction systems
