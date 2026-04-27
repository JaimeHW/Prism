"""Python bytecode path helpers.

Prism does not consume CPython ``.pyc`` files.  This module keeps the public
``py_compile`` shape available for stdlib helpers and returns the path a
bytecode compiler would have produced while Prism's import pipeline remains
source/native-code oriented.
"""

class PyCompileError(Exception):
    def __init__(self, exc_type, exc_value, file, msg=""):
        self.exc_type_name = getattr(exc_type, "__name__", str(exc_type))
        self.exc_value = exc_value
        self.file = file
        self.msg = msg or ("%s: %s" % (self.exc_type_name, exc_value))
        Exception.__init__(self, self.msg)

    def __str__(self):
        return self.msg


class PycInvalidationMode:
    TIMESTAMP = 1
    CHECKED_HASH = 2
    UNCHECKED_HASH = 3


def _default_cfile(file):
    return file + "c"


def compile(file, cfile=None, dfile=None, doraise=False, optimize=-1,
            invalidation_mode=None, quiet=0):
    if cfile is None:
        cfile = _default_cfile(file)
    return cfile


def main(args=None):
    return 0
