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
        self.pre_string = ""
        self.post_string = ""
        self.update_strings()

    def build_term(self, x):
        index = self.operation.find("(")
        if index != -1:
            expresion = self.pre_string + str(x) + self.post_string
        else:
            index = self.operation.find("[")
            if index != -1:
                expresion = self.pre_string + str(x) + self.post_string
            else:
                expresion = self.pre_string + str(x) + self.post_string

        return expresion

    def update_strings(self):
        index = self.operation.find("(")
        if index != -1:
            self.pre_string = self.sign + str(self.coefficient) + "*(" + self.operation[:(index + 1)]
            self.post_string = "**" + str(self.exponent) + self.operation[(index + 1):] + ")"
        else:
            index = self.operation.find("[")
            if index != -1:
                self.pre_string = self.sign + str(self.coefficient) + "*" + self.operation[:index] + "**("
                self.post_string = "*" + str(self.exponent) + ")"
            else:
                self.pre_string = self.sign + str(self.coefficient) + "*("
                self.post_string = "**" + str(self.exponent) + ")"

    def flip_sign(self):
        if self.sign == "":
            self.sign = "-"
        else:
            self.sign = "+"

    def set_exponent(self, x):
        self.exponent = x

    def set_coefficient(self, x):
        self.coefficient = x
