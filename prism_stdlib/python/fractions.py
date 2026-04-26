"""Compact Fraction implementation for Prism's compatibility stdlib."""


def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a or 1


class Fraction:
    def __init__(self, numerator=0, denominator=None):
        if denominator is None:
            denominator = 1
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise TypeError("both arguments should be Rational instances")
        if denominator == 0:
            raise ZeroDivisionError("Fraction(?, 0)")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        divisor = _gcd(numerator, denominator)
        self.numerator = numerator // divisor
        self.denominator = denominator // divisor

    def _compare_value(self, other):
        if isinstance(other, Fraction):
            return (
                self.numerator * other.denominator
                - other.numerator * self.denominator
            )
        if isinstance(other, int):
            return self.numerator - other * self.denominator
        if isinstance(other, float):
            left = float(self)
            if left < other:
                return -1
            if left > other:
                return 1
            return 0
        value = getattr(other, "_prism_decimal_value", None)
        if value is not None:
            left = float(self)
            if left < value:
                return -1
            if left > value:
                return 1
            return 0
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, complex):
            if other.imag != 0:
                return False
            return self == other.real
        result = self._compare_value(other)
        if result is NotImplemented:
            return NotImplemented
        return result == 0

    def __lt__(self, other):
        result = self._compare_value(other)
        if result is NotImplemented:
            return NotImplemented
        return result < 0

    def __le__(self, other):
        result = self._compare_value(other)
        if result is NotImplemented:
            return NotImplemented
        return result <= 0

    def __gt__(self, other):
        result = self._compare_value(other)
        if result is NotImplemented:
            return NotImplemented
        return result > 0

    def __ge__(self, other):
        result = self._compare_value(other)
        if result is NotImplemented:
            return NotImplemented
        return result >= 0

    def __float__(self):
        return self.numerator / self.denominator

    def __repr__(self):
        return "Fraction(%r, %r)" % (self.numerator, self.denominator)
