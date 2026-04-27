"""Small subprocess facade backed by Prism's native process runner."""

import _prism_subprocess

PIPE = -1
STDOUT = -2
DEVNULL = -3

CREATE_NEW_PROCESS_GROUP = 0
STARTF_USESHOWWINDOW = 1
STARTF_USESTDHANDLES = 256
SW_HIDE = 0


class SubprocessError(Exception):
    pass


class TimeoutExpired(SubprocessError):
    def __init__(self, cmd, timeout, output=None, stderr=None):
        self.cmd = cmd
        self.timeout = timeout
        self.output = output
        self.stderr = stderr
        SubprocessError.__init__(self, cmd, timeout)


class CalledProcessError(SubprocessError):
    def __init__(self, returncode, cmd, output=None, stderr=None):
        self.returncode = returncode
        self.cmd = cmd
        self.output = output
        self.stdout = output
        self.stderr = stderr
        SubprocessError.__init__(self, returncode, cmd, output, stderr)

    def __str__(self):
        return "Command %r returned non-zero exit status %r." % (
            self.cmd,
            self.returncode,
        )


class CompletedProcess:
    def __init__(self, args, returncode, stdout=None, stderr=None):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def check_returncode(self):
        if self.returncode:
            raise CalledProcessError(
                self.returncode,
                self.args,
                output=self.stdout,
                stderr=self.stderr,
            )


class STARTUPINFO:
    def __init__(self, *, dwFlags=0, hStdInput=None, hStdOutput=None,
                 hStdError=None, wShowWindow=0):
        self.dwFlags = dwFlags
        self.hStdInput = hStdInput
        self.hStdOutput = hStdOutput
        self.hStdError = hStdError
        self.wShowWindow = wShowWindow


class _BytesPipe:
    def __init__(self, data=b""):
        self._data = data
        self._pos = 0
        self.closed = False

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self._data) - self._pos
        end = self._pos + size
        data = self._data[self._pos:end]
        self._pos += len(data)
        return data

    def write(self, data):
        self._data = self._data + bytes(data)
        return len(data)

    def close(self):
        self.closed = True


def _normalize_args(args):
    if isinstance(args, str):
        return [args]
    if isinstance(args, tuple):
        return list(args)
    return args


class Popen:
    def __init__(self, args, stdin=None, stdout=None, stderr=None, env=None,
                 cwd=None, **kwargs):
        self.args = _normalize_args(args)
        self.stdin = _BytesPipe() if stdin == PIPE else None
        self.stdout = None
        self.stderr = None
        self.returncode = None
        self._capture_stdout = stdout == PIPE
        self._capture_stderr = stderr == PIPE
        self._merge_stderr = stderr == STDOUT
        self._env = env
        self._cwd = cwd
        self._stdout_data = b""
        self._stderr_data = b""
        self._completed = False

    def _run(self, input=None):
        if self._completed:
            return
        rc, out, err = _prism_subprocess.run(
            self.args,
            input,
            self._env,
            self._cwd,
            self._merge_stderr,
        )
        self.returncode = rc
        self._stdout_data = out
        self._stderr_data = err
        if self._capture_stdout:
            self.stdout = _BytesPipe(out)
        if self._capture_stderr:
            self.stderr = _BytesPipe(err)
        self._completed = True

    def communicate(self, input=None, timeout=None):
        if input is not None and not isinstance(input, (bytes, bytearray)):
            raise TypeError("communicate() input must be bytes")
        self._run(input)
        out = self._stdout_data if self._capture_stdout or self._merge_stderr else None
        err = self._stderr_data if self._capture_stderr else None
        return out, err

    def wait(self, timeout=None):
        self._run()
        return self.returncode

    def poll(self):
        if not self._completed:
            return None
        return self.returncode

    def kill(self):
        return None

    def terminate(self):
        return self.kill()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.wait()
        return False


def run(*popenargs, **kwargs):
    input = kwargs.pop("input", None)
    check = kwargs.pop("check", False)
    capture_output = kwargs.pop("capture_output", False)
    if capture_output:
        kwargs["stdout"] = PIPE
        kwargs["stderr"] = PIPE
    proc = Popen(*popenargs, **kwargs)
    stdout, stderr = proc.communicate(input)
    completed = CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    if check:
        completed.check_returncode()
    return completed


def check_call(*popenargs, **kwargs):
    proc = Popen(*popenargs, **kwargs)
    rc = proc.wait()
    if rc:
        raise CalledProcessError(rc, proc.args)
    return 0


def check_output(*popenargs, **kwargs):
    kwargs["stdout"] = PIPE
    proc = Popen(*popenargs, **kwargs)
    stdout, stderr = proc.communicate(kwargs.pop("input", None))
    if proc.returncode:
        raise CalledProcessError(proc.returncode, proc.args, output=stdout, stderr=stderr)
    return stdout


def call(*popenargs, **kwargs):
    return Popen(*popenargs, **kwargs).wait()


def _cleanup():
    return None
