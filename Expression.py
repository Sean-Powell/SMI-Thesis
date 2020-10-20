from Term import Term
from random import randint


class Expression:
    functions = {}
    grammar = []
    terms = []
    max_exponent = 0
    max_coefficient = 0

    def __init__(self, functions, grammar, min_size=1, max_size=5, max_exponent=3, max_coefficient=9):
        self.functions = functions
        self.grammar = grammar
        self.terms = []
        self.max_exponent = max_exponent
        self.max_coefficient = max_coefficient
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
            print("Expression", term.build_term(x))
            ans += float(eval(term.build_term(x), self.functions))

        print(ans)

    def print(self):
        output = ""
        for term in self.terms:
            output += str(term.build_term('x'))
        print(output)

    def mutation(self):
        # 50% chance for each term to mutate
        for term in self.terms:
            chance = randint(0, 1)
            if chance == 0:
                term.set_coefficient(randint(1, self.max_coefficient))
            chance = randint(0, 1)
            if chance == 0:
                term.set_exponent(randint(1, self.max_exponent))
            chance = randint(0, 1)
            if chance == 0:
                term.flip_sign()
