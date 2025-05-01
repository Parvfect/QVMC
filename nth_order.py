

from collections import defaultdict

class Variable:
    def __init__(self, value, local_gradients=()):
        self.value = value
        self.local_gradients = local_gradients

    def __add__(a, b):
        return add(a, b)
    
    def __sub__(a, b):
        return sub(a, b)
    
    def __mul__(a, b):
        return mul(a, b)
    
    def __truediv__(a, b):
        return div(a, b)
    
    def __pow__(a, b):
        return pow(a, b)

# Useful constants:
ONE = Variable(1.)
NEG_ONE = Variable(-1.)

def add(a, b):
    value = a.value + b.value    
    local_gradients = (
        # Note that local_gradients now contains lambda functions.
        (a, lambda path_value: path_value),
        # local gradient is 1, so multiply path_value by 1.
        (b, lambda path_value: path_value)
        # local gradient is 1, so multiply path_value by 1.
    )
    return Variable(value, local_gradients)

def sub(a, b):
    value = a.value - b.value    
    local_gradients = (
        # Note that local_gradients now contains lambda functions.
        (a, lambda path_value: path_value),
        # local gradient is 1, so multiply path_value by 1.
        (b, lambda path_value: NEG_ONE * path_value)
        # local gradient is 1, so multiply path_value by 1.
    )
    return Variable(value, local_gradients)

def mul(a, b):
    value = a.value * b.value
    local_gradients = (
        (a, lambda path_value: path_value * b),
        # local gradient for a is b, so multiply path_value by b.
        (b, lambda path_value : path_value * a)
        # local gradient for b is a, so multiply path_value by a.
    )
    return Variable(value, local_gradients)

def div(a, b):
    value = a.value / b.value
    local_gradients = (
        (a, lambda path_value : path_value * ONE/b),
        (b, lambda path_value : path_value * NEG_ONE * a/(b*b))
    )
    return Variable(value, local_gradients)

def pow(a, b):
    value = np.power(a.value, b.value)
    local_gradients = (
        (a, lambda path_value: path_value * b * a ** (b - ONE)),
        (b, lambda path_value: 0)
    )

    return Variable(value, local_gradients)

def get_gradients(variable):
    """ Compute the first derivatives of `variable` 
    with respect to child variables.
    """
    gradients = defaultdict(lambda: Variable(0))
    
    def compute_gradients(variable, path_value):
        for child_variable, multiply_by_locgrad in variable.local_gradients:
            # "Multiply the edges of a path":
            value_of_path_to_child = multiply_by_locgrad(path_value)  # Now a function is used here.
            # "Add together the different paths":
            gradients[child_variable] += value_of_path_to_child
            # recurse through graph:
            compute_gradients(child_variable, value_of_path_to_child)
    
    compute_gradients(variable, path_value=ONE)  # Now path_value is a Variable.
    # (path_value=1 is from `variable` differentiated w.r.t. itself)
    return gradients

"""
A 2nd derivative example.
y = x*x = x**2
y' = 2x (= 2*3 = 6)
y'' = 2
"""

x = Variable(3)
y = x * x

derivs_1 = get_gradients(y)
dy_by_dx = derivs_1[x]

print('y.value =', y.value)
print("The derivative of y with respect to x =", dy_by_dx.value)

derivs_2 = get_gradients(dy_by_dx)
dy_by_dx2 = derivs_2[x]
print("The 2nd derivative of y with respect to x =", dy_by_dx2.value)