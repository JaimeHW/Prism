"""Public warnings module backed by Prism's native _warnings engine."""

import sys

from _warnings import (
    _defaultaction as defaultaction,
    _filters_mutated,
    _onceregistry as onceregistry,
    filters,
    warn,
    warn_explicit,
)

__all__ = [
    "catch_warnings",
    "filterwarnings",
    "formatwarning",
    "resetwarnings",
    "showwarning",
    "simplefilter",
    "warn",
    "warn_explicit",
]


def showwarning(message, category, filename, lineno, file=None, line=None):
    msg = WarningMessage(message, category, filename, lineno, file, line)
    _showwarnmsg_impl(msg)


def formatwarning(message, category, filename, lineno, line=None):
    msg = WarningMessage(message, category, filename, lineno, None, line)
    return _formatwarnmsg_impl(msg)


def _formatwarnmsg_impl(msg):
    category = msg._category_name
    if category is None:
        category = "Warning"
    text = "%s:%s: %s: %s\n" % (
        msg.filename,
        msg.lineno,
        category,
        msg.message,
    )
    if msg.line:
        text += "  %s\n" % msg.line.strip()
    return text


def _showwarnmsg_impl(msg):
    file = msg.file
    if file is None:
        file = sys.stderr
    if file is None:
        return
    file.write(_formatwarnmsg(msg))


_showwarning_orig = showwarning
_formatwarning_orig = formatwarning


def _showwarnmsg(msg):
    if showwarning is not _showwarning_orig:
        if not callable(showwarning):
            raise TypeError("warnings.showwarning() must be set to a function or method")
        showwarning(
            msg.message,
            msg.category,
            msg.filename,
            msg.lineno,
            msg.file,
            msg.line,
        )
        return
    _showwarnmsg_impl(msg)


def _formatwarnmsg(msg):
    if formatwarning is not _formatwarning_orig:
        return formatwarning(msg.message, msg.category, msg.filename, msg.lineno, msg.line)
    return _formatwarnmsg_impl(msg)


def filterwarnings(action, message="", category=Warning, module="", lineno=0, append=False):
    if action not in ("error", "ignore", "always", "default", "module", "once"):
        raise AssertionError("invalid action: %r" % (action,))
    if not isinstance(message, str):
        raise AssertionError("message must be a string")
    if not isinstance(category, type):
        raise AssertionError("category must be a class")
    if not issubclass(category, Warning):
        raise AssertionError("category must be a Warning subclass")
    if not isinstance(module, str):
        raise AssertionError("module must be a string")
    if not isinstance(lineno, int) or lineno < 0:
        raise AssertionError("lineno must be an int >= 0")

    if message or module:
        import re
    if message:
        message = re.compile(message, re.I)
    else:
        message = None
    if module:
        module = re.compile(module)
    else:
        module = None

    _add_filter(action, message, category, module, lineno, append=append)


def simplefilter(action, category=Warning, lineno=0, append=False):
    if action not in ("error", "ignore", "always", "default", "module", "once"):
        raise AssertionError("invalid action: %r" % (action,))
    if not isinstance(category, type):
        raise AssertionError("category must be a class")
    if not issubclass(category, Warning):
        raise AssertionError("category must be a Warning subclass")
    if not isinstance(lineno, int) or lineno < 0:
        raise AssertionError("lineno must be an int >= 0")

    _add_filter(action, None, category, None, lineno, append=append)


def _add_filter(*item, append):
    if append:
        if item not in filters:
            filters.append(item)
    else:
        try:
            filters.remove(item)
        except ValueError:
            pass
        filters.insert(0, item)
    _filters_mutated()


def resetwarnings():
    filters[:] = []
    _filters_mutated()


class WarningMessage:
    _WARNING_DETAILS = ("message", "category", "filename", "lineno", "file", "line", "source")

    def __init__(self, message, category, filename, lineno, file=None, line=None, source=None):
        self.message = message
        self.category = category
        self.filename = filename
        self.lineno = lineno
        self.file = file
        self.line = line
        self.source = source
        self._category_name = category.__name__ if category else None

    def __str__(self):
        return (
            "{message : %r, category : %r, filename : %r, lineno : %s, line : %r}"
            % (self.message, self._category_name, self.filename, self.lineno, self.line)
        )


class catch_warnings:
    def __init__(
        self,
        *,
        record=False,
        module=None,
        action=None,
        category=Warning,
        lineno=0,
        append=False
    ):
        self._record = record
        self._module = sys.modules["warnings"] if module is None else module
        self._entered = False
        if action is None:
            self._filter = None
        else:
            self._filter = (action, category, lineno, append)

    def __repr__(self):
        args = []
        if self._record:
            args.append("record=True")
        return "%s(%s)" % (type(self).__name__, ", ".join(args))

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._module._filters_mutated()
        self._showwarning = self._module.showwarning
        self._showwarnmsg_impl = self._module._showwarnmsg_impl
        if self._filter is not None:
            simplefilter(*self._filter)
        if self._record:
            log = []
            self._module._showwarnmsg_impl = log.append
            self._module.showwarning = self._module._showwarning_orig
            return log
        return None

    def __exit__(self, *exc_info):
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        self._module.filters = self._filters
        self._module._filters_mutated()
        self._module.showwarning = self._showwarning
        self._module._showwarnmsg_impl = self._showwarnmsg_impl
