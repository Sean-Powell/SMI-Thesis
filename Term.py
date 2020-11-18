class Term:
    sign = ""
    operation = ""
    exponent = 1
    coefficient = 1

    def __init__(self, sign, operation, exponent, coefficient):
        if sign == 1:
            self.sign = "-"
        else:
            self.sign = "+"

        self.operation = operation
        self.exponent = exponent
        self.coefficient = coefficient

    def build_term(self, x):
        index = self.operation.find("(")
        if index != -1:
            expresion = self.sign + str(self.coefficient) + "*(" + self.operation[:(index + 1)] + str(x) + "**" +\
                        str(self.exponent) + self.operation[(index + 1):] + ")"
        else:
            index = self.operation.find("[")
            if index != -1:
                expresion = self.sign + str(self.coefficient) + "*" + self.operation[:index] + "**("  + str(x) + \
                            "*" + str(self.exponent) + ")"
            else:
                expresion = self.sign + str(self.coefficient) + "*(" + str(x) + "**" + str(self.exponent) + ")"
        return expresion

    def flip_sign(self):
        if self.sign == "":
            self.sign = "-"
        else:
            self.sign = "+"

    def set_exponent(self, x):
        self.exponent = x

    def set_coefficient(self, x):
        self.coefficient = x
