class PolyTerm:
    def __init__(self, coefficient, exponent, sign):
        self.coefficient = coefficient
        self.exponent = exponent
        if sign:
            self.sign = ""
        else:
            self.sign = "-"

    def flip_sign(self):
        if self.sign == "":
            self.sign = "-"
        else:
            self.sign = ""

    def coefficient_change(self, change):
        self.coefficient += change

    def exponent_change(self, change):
        self.exponent += change

    def build_term(self, x):
        return self.sign + str(self.coefficient) + "*(" + str(x) + "**" + str(self.exponent) + ")"

    def get_value(self, x):
        value = self.coefficient * (x ** self.exponent)
        if not self.sign == "":
            value = value * -1
        return value
