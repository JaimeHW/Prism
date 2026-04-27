"""Temporary file name helpers.

This is the small, import-safe core of CPython's tempfile API.  The functions
implemented here are enough for code that needs a platform-appropriate temporary
directory and unique candidate names without creating files.
"""

import os
import sys

__all__ = ["gettempdir", "gettempprefix", "mktemp"]

template = "tmp"
tempdir = None
_counter = 0


def _candidate_tempdir():
    for key in ("TMPDIR", "TEMP", "TMP"):
        value = os.getenv(key)
        if value:
            return value
    if sys.platform == "win32":
        return os.getcwd()
    return "/tmp"


def _join(dir, name):
    if not dir:
        return name
    sep = "\\" if sys.platform == "win32" else "/"
    if dir.endswith("/") or dir.endswith("\\"):
        return dir + name
    return dir + sep + name


def gettempdir():
    global tempdir
    if tempdir is None:
        tempdir = _candidate_tempdir()
    return tempdir


def gettempprefix():
    return template


def mktemp(suffix="", prefix=template, dir=None):
    """Return a unique temporary pathname candidate.

    Like CPython's deprecated ``tempfile.mktemp()``, this function does not
    create the file.  Callers that need race-free creation should use a creating
    API once Prism's file-descriptor tempfile helpers are available.
    """

    global _counter
    if dir is None:
        dir = gettempdir()
    if suffix is None:
        suffix = ""
    if prefix is None:
        prefix = template

    _counter += 1
    name = "%s%d_%d%s" % (prefix, os.getpid(), _counter, suffix)
    return _join(dir, name)
