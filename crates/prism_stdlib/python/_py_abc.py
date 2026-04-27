"""Pure-Python ABC surface backed by Prism's native _abc state."""

from abc import (
    ABCMeta,
    get_cache_token,
    abstractclassmethod,
    abstractmethod,
    abstractproperty,
    abstractstaticmethod,
)


__all__ = (
    "ABCMeta",
    "abstractclassmethod",
    "abstractmethod",
    "abstractproperty",
    "abstractstaticmethod",
    "get_cache_token",
)
