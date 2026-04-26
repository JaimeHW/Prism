"""Abstract base class support backed by Prism's native _abc accelerator."""

from _abc import (
    get_cache_token,
    _abc_init,
    _abc_register,
    _abc_instancecheck,
    _abc_subclasscheck,
    _get_dump,
    _reset_registry,
    _reset_caches,
)


def abstractmethod(funcobj):
    funcobj.__isabstractmethod__ = True
    return funcobj


class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


class abstractstaticmethod(staticmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


class abstractproperty(property):
    __isabstractmethod__ = True


class ABCMeta(type):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = type.__new__(mcls, name, bases, namespace)
        _abc_init(cls)
        return cls

    def register(cls, subclass):
        return _abc_register(cls, subclass)

    def __instancecheck__(cls, instance):
        return _abc_instancecheck(cls, instance)

    def __subclasscheck__(cls, subclass):
        return _abc_subclasscheck(cls, subclass)

    def _dump_registry(cls, file=None):
        registry, cache, negative_cache, version = _get_dump(cls)
        print("_abc_registry:", registry, file=file)
        print("_abc_cache:", cache, file=file)
        print("_abc_negative_cache:", negative_cache, file=file)
        print("_abc_negative_cache_version:", version, file=file)

    def _abc_registry_clear(cls):
        _reset_registry(cls)

    def _abc_caches_clear(cls):
        _reset_caches(cls)


class ABC(metaclass=ABCMeta):
    pass


def update_abstractmethods(cls):
    return cls
