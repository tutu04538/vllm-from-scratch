"""在 111 的既有独立进程驱动上安装可选的逐层观察钩子。"""
import argparse
import importlib
import sys


def main(hooks_module):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diag-request',required=True)
    parser.add_argument('--diag-positions',required=True)
    args,remaining=parser.parse_known_args()
    positions=[int(value) for value in args.diag_positions.split(',')]
    sys.argv=[sys.argv[0],*remaining]
    hooks=importlib.import_module(hooks_module)
    from benchmarks import step111_parallel_worker as worker
    install_gate,flush_gate=worker.install_gate,worker.flush_gate

    def install(rpc,upstream,k):
        if k: raise ValueError('逐层诊断只支持 K0')
        install_gate(rpc,upstream,k)
        rpc(hooks.install,args.diag_request,positions)

    def flush(rpc,directory,k):
        rpc(hooks.flush,str(directory)+'_layers')
        return flush_gate(rpc,directory,k)

    worker.install_gate=install
    worker.flush_gate=flush
    return worker.main()
