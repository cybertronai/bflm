import os
import sys


def _info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"


def pdb_on_error():
    sys.excepthook = _info


def find_worker_script() -> str:
    """Assume argv[0] is launch_xyz.py, it must have xyz.py in same directory, return xyz.py"""
    launcher_fn = os.path.abspath(sys.argv[0])
    launcher_fn_rel = os.path.basename(launcher_fn)
    assert launcher_fn_rel.startswith('launch_')
    worker_fn_rel = launcher_fn_rel[len('launch_'):]
    worker_fn = os.path.dirname(launcher_fn)+'/'+worker_fn_rel
    assert os.path.exists(worker_fn), f"{worker_fn} not found"
    return worker_fn
