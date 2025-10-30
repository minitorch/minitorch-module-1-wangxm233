"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0.0}{(1.0.0 + e^{-x})}$ if x >=0.0 else $\frac{e^x}{(1.0.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.0.1.
def mul(a: float, b: float) -> float:
    """Add two numbers.

    Args:
    ----
        a (float): First operand.
        b (float): Second operand.

    Returns:
    -------
        float: Multipiles two numbers.

    """
    return a * b


def id(a: float) -> float:
    """Returns the input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers"""
    return a + b


def neg(a: float) -> float:
    """Negates a number"""
    return -1.0 * a


def lt(a: float, b: float) -> bool:
    """Checks if one number is less than another"""
    return a < b


def eq(a: float, b: float) -> bool:
    """Checks if two numbers are equal"""
    return a == b


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers"""
    if a > b:
        return a
    else:
        return b


def is_close(a: float, b: float) -> bool:
    """Checks if two numbers are close in value"""
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Calculates the sigmoid function"""
    if a >= 0.0:
        return 1 / (1 + math.exp(-a))
    else:
        return math.exp(a) / (1 + math.exp(a))


def relu(a: float) -> float:
    """Applies the Relu activation function"""
    if a >= 0.0:
        return a
    else:
        return 0.0


def log(a: float) -> float:
    """Calculates the natural logarithm"""
    if a > 0.0:
        return math.log(a)
    else:
        raise ValueError("Cannot compute logarithm of non-positive number.")


def exp(a: float) -> float:
    """Calculates the exponential function"""
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the recipocal"""
    if a == 0.0:
        raise ZeroDivisionError("Cannot calculate the reciprocal of zero.")
    return 1 / a


def log_back(x: float, g: float) -> float:
    """Computes the derivative of log times"""
    return g / x


def inv_back(x: float, g: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -g / (x * x)


def relu_back(x: float, g: float) -> float:
    """Computes the derivative of ReLu times a second arg"""
    if x > 0.0:
        return g
    else:
        return 0.0


# ## Task 0.0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a function to each element an iterable"""

    def apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements from two iterables using a given function"""

    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x1, x2) for x1, x2 in zip(ls1, ls2)]

    return apply


def reduce(
    fn: Callable[[float, float], float], initial: float
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value using a given function"""

    def apply(ls: Iterable[float]) -> float:
        result = initial
        for x in ls:
            result = fn(result, x)
        return result

    return apply


negList: Callable[[Iterable[float]], Iterable[float]] = map(neg)  # noqa: N816
addLists: Callable[[Iterable[float], Iterable[float]], Iterable[float]] = zipWith(add)  # noqa: N816
sum: Callable[[Iterable[float]], float] = reduce(add, 0.0)
prod: Callable[[Iterable[float]], float] = reduce(mul, 1)
