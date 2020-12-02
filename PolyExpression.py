from random import uniform, randint
from PolyTerm import PolyTerm
from copy import deepcopy


class PolyExpression:
    def __init__(self, min_length=1, max_length=6, max_exponent=6.0, max_coefficient=9.0, mutation_rate=10,
                 max_mutation_size=5.0):

        self.min_length = min_length
        self.max_length = max_length
        self.max_exponent = max_exponent
        self.max_coefficient = max_coefficient
        self.mutation_rate = mutation_rate
        self.max_mutation_size = max_mutation_size
        self.terms = []
        self.length = randint(min_length, max_length)
        self.chi_square = 0

        for i in range(self.length):
            self.terms.append(deepcopy(self.create_term()))

    def mutate(self):
        roll = randint(0, 100)
        if roll <= self.mutation_rate:
            size_roll = randint(0, 1)
            size_change = False
            if size_roll and self.length < self.max_length:
                self.length += 1
                self.terms.append(deepcopy(self.create_term()))
                size_change = True

            size_roll = randint(0, 1)
            if size_roll and self.length > self.min_length and not size_change:
                index = randint(0, self.length - 1)
                self.terms.remove(self.terms[index])
                self.length -= 1

            for i in range(self.length):
                exp_roll = randint(0, 1)
                coeff_roll = randint(0, 1)
                sign_roll = randint(0, 1)

                if exp_roll:
                    self.terms[i].exponent_change(uniform(-self.max_mutation_size, self.max_mutation_size))

                if coeff_roll:
                    self.terms[i].coefficient_change(uniform(-self.max_mutation_size, self.max_mutation_size))

                if sign_roll:
                    self.terms[i].flip_sign()
            return True
        else:
            return False

    def create_term(self):
        coefficient = uniform(0, self.max_coefficient)
        exponent = uniform(0, self.max_exponent)
        sign = randint(0, 1)
        return PolyTerm(coefficient, exponent, sign)

    def get_equation(self, x):
        equation = ""
        for i in range(self.length):
            if equation != "":
                equation += "+"
            equation += self.terms[i].build_term(x)

        return equation

    def evaluate(self, x):
        value = 0
        for i in range(self.length):
            value += self.terms[i].get_value(x)
        return value

    def set_chi_squared(self, value):
        self.chi_square = value

    def get_chi_squared(self):
        return self.chi_square

    def set_terms(self, terms):
        self.terms = terms
        self.length = len(terms)
