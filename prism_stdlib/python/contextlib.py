"""Small source-backed subset of CPython's contextlib module."""

__all__ = ["contextmanager"]


class _GeneratorContextManager:
    def __init__(self, func, args, kwds):
        self.func = func
        self.args = args
        self.kwds = kwds
        self.gen = func(*args, **kwds)

    def _recreate_cm(self):
        return type(self)(self.func, self.args, self.kwds)

    def __call__(self, func):
        def inner(*args, **kwds):
            with self._recreate_cm():
                return func(*args, **kwds)

        inner.__name__ = getattr(func, "__name__", "inner")
        inner.__qualname__ = getattr(func, "__qualname__", inner.__name__)
        inner.__doc__ = getattr(func, "__doc__", None)
        inner.__module__ = getattr(func, "__module__", None)
        return inner

    def __enter__(self):
        try:
            return next(self.gen)
        except StopIteration:
            raise RuntimeError("generator didn't yield")

    def __exit__(self, typ, value, traceback):
        if typ is not None:
            return False

        try:
            next(self.gen)
        except StopIteration:
            return False
        raise RuntimeError("generator didn't stop")


def contextmanager(func):
    def helper(*args, **kwds):
        return _GeneratorContextManager(func, args, kwds)

    helper.__name__ = getattr(func, "__name__", "helper")
    helper.__qualname__ = getattr(func, "__qualname__", helper.__name__)
    helper.__doc__ = getattr(func, "__doc__", None)
    helper.__module__ = getattr(func, "__module__", None)
    return helper
