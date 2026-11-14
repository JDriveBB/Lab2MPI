"""MPI version 2: dynamic master-worker distribution (load balancing).

rank 0 is the master: it broadcasts the query and hands out ONE file at a time
to whichever worker asks for work. Workers (ranks 1..p-1) count locally and send their partial Counter at
the end. Faster workers automatically receive more files, so the unequal
file sizes no longer produce idle time at the end.
"""
import os
from collections import Counter

from mpi4py import MPI

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TOP_N = 10
TAG_REQUEST, TAG_WORK, TAG_RESULT = 1, 2, 3


def count_file(path, targets, freq):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for w in line.lower().split():
                if w in targets:
                    freq[w] += 1


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    comm.Barrier()
    t_start = MPI.Wtime()

    words, files = None, None
    if rank == 0:
        with open(os.path.join(DATASET, "consulta.txt"), "r", encoding="utf-8") as f:
            words = sorted({l.strip().lower() for l in f if l.strip()})
        files = sorted(
            os.path.join(DATASET, n)
            for n in os.listdir(DATASET)
            if n.startswith("file_") and n.endswith(".txt")
        )
        # NOTE: sorting by size (largest first) was tried; os.stat on the Docker
        # bind mount costs ~1.5 ms/file (~4.5 s for 3000 files), more than it saves.
    words = comm.bcast(words, root=0)
    targets = set(words)

    if size == 1:  # degenerate case: no workers, master does everything
        local = Counter()
        t0 = MPI.Wtime()
        for p in files:
            count_file(p, targets, local)
        print("MPI version 2 | processes=1")
        print(f"  rank 0: files={len(files)} local_time={MPI.Wtime() - t0:.6f}s")
        print(f"EXECUTION_TIME={MPI.Wtime() - t_start:.6f}")
        for w, c in local.most_common(TOP_N):
            print(f"  {w}: {c}")
        return

    if rank == 0:
        status = MPI.Status()
        next_idx, stopped = 0, 0
        while stopped < size - 1:
            comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQUEST, status=status)
            worker = status.Get_source()
            if next_idx < len(files):
                comm.send(files[next_idx], dest=worker, tag=TAG_WORK)
                next_idx += 1
            else:
                comm.send(None, dest=worker, tag=TAG_WORK)  # stop signal
                stopped += 1
        results = [comm.recv(source=w, tag=TAG_RESULT) for w in range(1, size)]
        t_total = MPI.Wtime() - t_start

        total = Counter()
        for _, _, _, c in results:
            total.update(c)
        times = [r[2] for r in results]
        mean_t = sum(times) / len(times)
        print(f"MPI version 2 | processes={size} (1 master + {size - 1} workers) | files={len(files)}")
        for r, n, t, _ in sorted(results, key=lambda x: x[0]):
            print(f"  rank {r}: files={n} local_time={t:.6f}s")
        print(f"LI={max(times) / mean_t:.4f}")
        print(f"SPREAD={(max(times) - min(times)) / mean_t:.4f}")
        print(f"EXECUTION_TIME={t_total:.6f}")
        print(f"Total occurrences: {sum(total.values())}")
        print(f"Top {TOP_N} query words:")
        for w, c in total.most_common(TOP_N):
            print(f"  {w}: {c}")
    else:
        local, n_files, t_local = Counter(), 0, 0.0
        while True:
            comm.send(None, dest=0, tag=TAG_REQUEST)
            path = comm.recv(source=0, tag=TAG_WORK)
            if path is None:
                break
            t0 = MPI.Wtime()
            count_file(path, targets, local)
            t_local += MPI.Wtime() - t0
            n_files += 1
        comm.send((rank, n_files, t_local, local), dest=0, tag=TAG_RESULT)


if __name__ == "__main__":
    main()
