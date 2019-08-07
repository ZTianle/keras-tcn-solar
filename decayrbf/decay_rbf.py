import math

def decay_rbf(x, def_function):

    if def_function == 'Mexican Hat ':
        fai = math.exp(-x**2)

    elif def_function == 'Morlet':
        fai = 2/(3**0.5)*math.pi**(-1/4)*(1-x**2)*math.exp(-x**2/2)

    elif def_function == 'Gaussian':
        fai = 2/(3**0.5)*math.exp(-x**2/2)*math.cos(5*x)
    else:
        raise Exception('wrong function')

    return fai


