import re
import random
from collections import defaultdict
import itertools
from abc import ABC, abstractmethod
import math


class Operator(ABC):

    @abstractmethod
    def f(self, a, b=None) -> float:
        raise NotImplementedError()
        return f_res

    @abstractmethod
    def df(self, a, b=None) -> list:
        raise NotImplementedError()
        return [df_res]


class Exp(Operator):

    def f(self, a, b=None):
        return math.exp(a)

    def df(self, a, b=None):
        return [math.exp(a)]


class Log(Operator):
    ## natural logarithm

    def f(self, a, b=None):
        ## ToDO: implement
        return math.log(a)

    def df(self, a, b=None):
        ## ToDO: implement
        return [1 / a]


class Mult(Operator):

    def f(self, a, b):
        return a * b

    def df(self, a, b=None):
        return [b, a]


class Div(Operator):

    def f(self, a, b):
        ## ToDO: implement
        return a / b

    def df(self, a, b):
        ## ToDO: implement
        return [1 / b, -a / (b * b)]


class Add(Operator):

    def f(self, a, b):
        ## ToDO: implement
        return a + b

    def df(self, a, b=None):
        ## ToDO: implement
        return [1, 1]


class Sub(Operator):

    def f(self, a, b=None):
        ## ToDO: implement
        return a - b

    def df(self, a, b=None):
        ## ToDO: implement
        return [1, -1]


class Pow(Operator):

    def f(self, a, b):
        return a**b

    def df(self, a, b):
        if a <= 0:  ## work-around: treat as unary operation if -a^b
            return [b * (a ** (b - 1))]
        else:
            return [b * (a ** (b - 1)), (a**b) * math.log(a)]


class Sin(Operator):

    def f(self, a, b=None):
        ## ToDO: implement
        return math.sin(a)

    def df(self, a, b=None):
        ## ToDO: implement
        return [math.cos(a)]


class Cos(Operator):

    def f(self, a, b=None):
        ## ToDO: implement
        return math.cos(a)

    def df(self, a, b=None):
        ## ToDO: implement
        return [-math.sin(a)]


class Executor:

    def __init__(self, graph: dict, in_vars: dict = {}):
        """
        graph: computation graph in a data structure of your choosing
        in_vars: dict of input variables, e.g. {"x": 2.0, "y": -1.0}
        """
        self.graph = graph
        self.in_vars = in_vars
        self.fn_map = {
            "log": Log(),
            "exp": Exp(),
            "+": Add(),
            "-": Sub(),
            "^": Pow(),
            "sin": Sin(),
            "cos": Cos(),
            "*": Mult(),
            "/": Div(),
        }
        self.output = -1
        self.derivative = {}

    ## forward execution____________________________

    def forward(self):
        ## ToDO: implement and set self.output

        # Go through variables in topological order
        for key in self.graph.keys():

            # If the variable isn't yet evaluated, evaluate it based on parent nodes
            if self.graph[key]["value"] is None:

                # Each variable has parent nodes (operands), and the function to apply on the parent nodes (fun_to_use)
                fun_to_use = self.graph[key]["function"]
                operands = self.graph[key]["operands"].copy()

                # Find the values of the parent nodes needed to compute the value of the current variable
                num_operands = len(operands)
                if num_operands == 1:
                    if operands[0] in self.graph.keys():
                        operands[0] = self.graph[operands[0]]["value"]
                elif num_operands == 2:
                    if operands[0] in self.graph.keys():
                        operands[0] = self.graph[operands[0]]["value"]
                    if operands[1] in self.graph.keys():
                        operands[1] = self.graph[operands[1]]["value"]

                computed_val = self.fn_map[fun_to_use].f(*operands)

                self.graph[key]["value"] = computed_val

        self.output = self.graph[list(self.graph.keys())[-1]]["value"]

    ## backward execution____________________________

    def backward(self):
        self.derivative = {}

        # Go through all nodes and set all  backwards values to 0
        for key in self.graph.keys():
            self.graph[key]["backwards"] = 0

        # 1. Go through nodes in reverse topological order (starting from the back). Last variable has backwards value = 1.
        # 2. For all children of the current node, find calculate the product of the corresponding gradient multiplied by the current node value
        # 3. Add the value to the children node backwards_value
        # (No need for recursion - just do it iteratively)

        output_flag = 0

        for key in reversed(list(self.graph.keys())):
            # print('Current key: ', key)

            # Set the derivative of output wrt itself to 1 (backwards value)
            if not output_flag:
                self.graph[key]["backwards"] = 1
                output_flag = 1

            # Calculate the gradient of current node wrt its parent nodes
            fun_to_use = self.graph[key]["function"]

            parent_values = []

            operands = self.graph[key]["operands"]
            if operands is not None:
                for operand in operands:
                    if type(operand) is str:
                        parent_values.append(self.graph[operand]["value"])
                    elif operand is not None:
                        parent_values.append(operand)

            if fun_to_use is not None:
                gradient = self.fn_map[fun_to_use].df(*parent_values)

            # Do the rest of the variables
            if self.graph[key]["operands"] is not None:
                for i, parent in enumerate(self.graph[key]["operands"]):
                    if (parent is not None) and (type(parent) is str):
                        self.graph[parent]["backwards"] += (
                            self.graph[key]["backwards"] * gradient[i]
                        )

        # Final output
        for input_var in self.in_vars:
            self.derivative[input_var] = self.graph[input_var]["backwards"]
