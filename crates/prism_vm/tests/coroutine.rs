use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::VirtualMachine;

fn execute(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");
    let mut vm = VirtualMachine::new();
    vm.execute_runtime(code).expect("source should execute");
}

#[test]
fn await_preserves_delegate_registers_above_liveness_low_word() {
    execute(
        r#"
class WaitOnce:
    def __await__(self):
        yield None
        return None

async def make_transport(local_addr=None, remote_addr=("127.0.0.1", 0), family=0, proto=0, flags=0):
    addr_infos = {}
    for idx, addr in ((0, local_addr), (1, remote_addr)):
        if addr is not None:
            infos = [(2, 2, 17, "", ("127.0.0.1", 0))]
            if not infos:
                raise OSError("getaddrinfo() returned empty list")
            for fam, _, pro, _, address in infos:
                key = (fam, pro)
                if key not in addr_infos:
                    addr_infos[key] = [None, None]
                addr_infos[key][idx] = address

    addr_pairs_info = [(key, addr_pair) for key, addr_pair in addr_infos.items()]
    exceptions = []
    for ((family, proto), (local_address, remote_address)) in addr_pairs_info:
        sock = None
        r_addr = None
        try:
            sock = object()
            if local_addr:
                pass
            if remote_addr:
                r_addr = remote_address
        except OSError as exc:
            if sock is not None:
                pass
            exceptions.append(exc)
        except:
            if sock is not None:
                pass
            raise
        else:
            break
    else:
        raise exceptions[0]

    protocol = "protocol"
    transport = ("transport", sock, r_addr)
    await WaitOnce()

    if protocol != "protocol":
        raise RuntimeError(protocol)
    if transport[0] != "transport":
        raise RuntimeError(transport)
    if transport[2] != ("127.0.0.1", 0):
        raise RuntimeError(transport)

coroutine = make_transport()
if coroutine.send(None) is not None:
    raise RuntimeError("first coroutine send should yield None")

try:
    coroutine.send(None)
except StopIteration:
    pass
else:
    raise RuntimeError("coroutine should finish")
"#,
    );
}
