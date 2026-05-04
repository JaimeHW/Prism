use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::VirtualMachine;

fn execute(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");
    let mut vm = VirtualMachine::new();
    vm.execute_runtime(code).expect("source should execute");
}

#[test]
fn nonblocking_recv_raises_blocking_io_error() {
    execute(
        r#"
import _socket

server = _socket.socket()
server.bind(("127.0.0.1", 0))
server.listen()

client = _socket.socket()
client.connect(server.getsockname())

fd, peer = server._accept()
accepted = _socket.socket(server.family, server.type, server.proto, fd)
accepted.setblocking(False)

try:
    accepted.recv(1)
except BlockingIOError:
    pass
else:
    raise RuntimeError("nonblocking recv did not raise BlockingIOError")
"#,
    );
}

#[test]
fn getaddrinfo_resolves_numeric_and_named_services() {
    execute(
        r#"
import _socket

for service, expected in (("80", 80), (b"80", 80), ("http", 80), (b"http", 80)):
    infos = _socket.getaddrinfo("127.0.0.1", service, _socket.AF_INET, _socket.SOCK_STREAM)
    sockaddr = infos[0][4]
    if sockaddr != ("127.0.0.1", expected):
        raise RuntimeError(sockaddr)

for service in ("nonsense", b"nonsense"):
    try:
        _socket.getaddrinfo("127.0.0.1", service, _socket.AF_INET, _socket.SOCK_STREAM)
    except OSError:
        pass
    else:
        raise RuntimeError("unknown service should fail")
"#,
    );
}
