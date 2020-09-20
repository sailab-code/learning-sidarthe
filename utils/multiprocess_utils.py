import multiprocessing as mp
from typing import Dict

#mp.set_start_method('spawn')

N_PROCESSES = 6
TIMEOUT=10


class ProcessPool:
    def __init__(self, max_n_procs=N_PROCESSES):
        self.max_n_procs = max_n_procs
        self.procs: Dict[int, mp.Process] = {} # id => Process

    def start(self, target, kwargs):
        proc = mp.Process(
            target=target,
            kwargs=kwargs
        )

        proc.start()
        self.procs[proc.pid] = proc

    def wait_for_empty_slot(self, timeout=TIMEOUT):
        # run N_PROCESSES exps at a time
        if len(self.procs) == self.max_n_procs:
            while True:
                for pid, proc in self.procs.items():
                    proc.join(timeout=timeout)
                    if not proc.is_alive():
                        del self.procs[pid]
                        return

    def wait_for_all(self):
        for pid, proc in self.procs.items():
            proc.join()