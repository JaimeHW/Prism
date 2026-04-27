"""Minimal zipfile API used by CPython support helpers."""

ZIP_STORED = 0
ZIP_DEFLATED = 8


class BadZipFile(Exception):
    pass


class LargeZipFile(Exception):
    pass


class ZipInfo:
    def __init__(self, filename="", date_time=None):
        self.filename = filename
        self.date_time = date_time
        self.compress_type = ZIP_STORED


class ZipFile:
    def __init__(self, file, mode="r", compression=ZIP_STORED, allowZip64=True,
                 compresslevel=None, *, strict_timestamps=True, metadata_encoding=None):
        self.filename = file
        self.mode = mode
        self.compression = compression
        self._names = []
        if mode not in ("r", "w", "x", "a"):
            raise ValueError("ZipFile requires mode 'r', 'w', 'x', or 'a'")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        return None

    def write(self, filename, arcname=None, compress_type=None, compresslevel=None):
        if arcname is None:
            arcname = filename
        self._names.append(arcname)

    def writestr(self, zinfo_or_arcname, data, compress_type=None, compresslevel=None):
        name = getattr(zinfo_or_arcname, "filename", zinfo_or_arcname)
        self._names.append(name)

    def namelist(self):
        return list(self._names)

    def printdir(self, file=None):
        for name in self._names:
            print(name)


def is_zipfile(filename):
    return False
