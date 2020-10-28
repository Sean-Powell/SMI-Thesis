from Term import Term
from random import randint


class Expression:
    terms = []
    max_exponent = 0
    max_coefficient = 0
    max_size = 0
    chi_2 = -1

    def __init__(self, functions, grammar, min_size=1, max_size=1, max_exponent=3, max_coefficient=9):
        self.functions = functions
        self.grammar = grammar
        self.terms = []
        self.max_exponent = max_exponent
        self.max_coefficient = max_coefficient
        self.max_size = max_size
        self.chi_2 = 0

        size = randint(min_size, max_size)
        for _ in range(size):
            sign = randint(0, 1)
            operation = grammar[randint(0, len(grammar) - 1)]
            exponent = randint(1, max_exponent)
            coefficient = randint(1, max_coefficient)
            self.terms.append(Term(sign, operation, exponent, coefficient))

    def evaluate(self, x):
        ans = 0
        for term in self.terms:
            ans += float(eval(term.build_term(x), self.functions))
        return ans

    def print(self):
        output = ""
        for term in self.terms:
            output += str(term.build_term('x'))
        print(output)

    def get_string(self):
        output = ""
        for term in self.terms:
            output += str(term.build_term('x'))
        return output

    # def mutation(self):
    #     # 10% chance for a random term to be removed (iff length of terms is more than 1) or added to the expression
    #     chance = randint(0, 100)
    #     if chance <= self.mutation_chance:
    #         chance = randint(0, 1)
    #         if chance == 0 and len(self.terms) > 1:
    #             index = randint(0, (len(self.terms) - 1))
    #             self.terms.remove(self.terms[index])
    #         elif chance == 1:
    #             sign = randint(0, 1)
    #             operation = self.grammar[randint(0, len(self.grammar) - 1)]
    #             exponent = randint(1, self.max_exponent)
    #             coefficient = randint(1, self.max_coefficient)
    #             self.terms.append(Term(sign, operation, exponent, coefficient))
    #     else:
    #         # for term in self.terms:
    #         index = randint(0, (len(self.terms) - 1))
    #         term = self.terms[index]
    #
    #         chance = randint(0, 1)
    #         if chance == 0:
    #             term.set_coefficient(term.coefficient - 1)
    #         elif chance == 1:
    #             term.set_coefficient(term.coefficient + 1)
    #
    #         chance = randint(0, 1)
    #         if chance == 0:
    #             term.set_exponent(term.exponent - 1)
    #         elif chance == 1:
    #             term.set_exponent(term.exponent + 1)
    #
    #         chance = randint(0, 1)
    #         if chance == 0:
    #             term.flip_sign()

    def set_terms(self, terms):
        self.terms = terms

    def get_chi_2(self):
        return self.chi_2

    def set_chi_2(self, chi_2):
        self.chi_2 = chi_2

