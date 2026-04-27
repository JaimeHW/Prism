"""Compact Decimal implementation for Prism's compatibility stdlib."""


class Decimal:
    def __init__(self, value="0"):
        if isinstance(value, Decimal):
            self._prism_decimal_value = value._prism_decimal_value
        elif isinstance(value, int) or isinstance(value, float):
            self._prism_decimal_value = float(value)
        elif isinstance(value, str):
            self._prism_decimal_value = float(value)
        else:
            raise TypeError("conversion from %s to Decimal is not supported" % type(value).__name__)

    def _coerce(self, other):
        if isinstance(other, Decimal):
            return other._prism_decimal_value
        if isinstance(other, int) or isinstance(other, float):
            return float(other)
        value = getattr(other, "_prism_decimal_value", None)
        if value is not None:
            return value
        numerator = getattr(other, "numerator", None)
        denominator = getattr(other, "denominator", None)
        if isinstance(numerator, int) and isinstance(denominator, int):
            return numerator / denominator
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, complex):
            if other.imag != 0:
                return False
            other = other.real
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented
        return self._prism_decimal_value == other_value

    def __lt__(self, other):
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented
        return self._prism_decimal_value < other_value

    def __le__(self, other):
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented
        return self._prism_decimal_value <= other_value

    def __gt__(self, other):
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented
        return self._prism_decimal_value > other_value

    def __ge__(self, other):
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented
        return self._prism_decimal_value >= other_value

    def __float__(self):
        return self._prism_decimal_value

    def __repr__(self):
        return "Decimal(%r)" % str(self._prism_decimal_value)
