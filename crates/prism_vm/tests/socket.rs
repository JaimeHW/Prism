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

#[test]
fn exports_tcp_socket_level_alias() {
    execute(
        r#"
import _socket

if _socket.SOL_TCP != _socket.IPPROTO_TCP:
    raise RuntimeError((_socket.SOL_TCP, _socket.IPPROTO_TCP))
"#,
    );
}

#[test]
fn udp_sendto_and_recvfrom_round_trip() {
    execute(
        r#"
import _socket

server = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
server.bind(("127.0.0.1", 0))
server.settimeout(1.0)

client = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
sent = client.sendto(b"ok", server.getsockname())
if sent != 2:
    raise RuntimeError(sent)

data, peer = server.recvfrom(16)
if data != b"ok":
    raise RuntimeError(data)
if peer[0] != "127.0.0.1":
    raise RuntimeError(peer)
"#,
    );
}

#[test]
fn udp_nonblocking_recvfrom_raises_blocking_io_error() {
    execute(
        r#"
import _socket

sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 0))
sock.setblocking(False)

try:
    sock.recvfrom(1)
except BlockingIOError:
    pass
else:
    raise RuntimeError("nonblocking recvfrom did not raise BlockingIOError")
"#,
    );
}

#[test]
fn udp_broadcast_socket_option_round_trips() {
    execute(
        r#"
import _socket

sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
if sock.getsockopt(_socket.SOL_SOCKET, _socket.SO_BROADCAST):
    raise RuntimeError("SO_BROADCAST should be disabled by default")

sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_BROADCAST, 1)
if not sock.getsockopt(_socket.SOL_SOCKET, _socket.SO_BROADCAST):
    raise RuntimeError("SO_BROADCAST should be enabled")
"#,
    );
}

#[test]
fn getaddrinfo_preserves_ipv6_family_and_scope_id() {
    execute(
        r#"
import _socket

infos = _socket.getaddrinfo("::1", 80, _socket.AF_UNSPEC, _socket.SOCK_STREAM)
if infos[0][0] != _socket.AF_INET6:
    raise RuntimeError(infos[0])
if infos[0][4] != ("::1", 80, 0, 0):
    raise RuntimeError(infos[0][4])

infos = _socket.getaddrinfo("fe80::1%1", b"http", _socket.AF_UNSPEC, _socket.SOCK_STREAM)
if infos[0][0] != _socket.AF_INET6:
    raise RuntimeError(infos[0])
if infos[0][4] != ("fe80::1", 80, 0, 1):
    raise RuntimeError(infos[0][4])
"#,
    );
}
