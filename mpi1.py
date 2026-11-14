"""MPI version 1: broadcast of the query words + static block distribution of files.

rank 0 reads consulta.txt and lists the file_*.txt files; the query is broadcast,
the files are split in contiguous blocks (one per process), every process counts
locally and the partial Counters are gathered in rank 0, which prints the top 10.
"""
import os
from collections import Counter

from mpi4py import MPI

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TOP_N = 10


def count_files(paths, targets):
    """Same tokenization as baseline_secuencial.py (split + lower)."""
    freq = Counter()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.lower().split():
                    if w in targets:
                        freq[w] += 1
    return freq


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
    words = comm.bcast(words, root=0)
    targets = set(words)

    # Static distribution: contiguous blocks, sizes differ by at most one file.
    if rank == 0:
        base, extra = divmod(len(files), size)
        chunks, start = [], 0
        for r in range(size):
            n = base + (1 if r < extra else 0)
            chunks.append(files[start:start + n])
            start += n
    else:
        chunks = None
    my_files = comm.scatter(chunks, root=0)

    t0 = MPI.Wtime()
    local = count_files(my_files, targets)
    t_local = MPI.Wtime() - t0

    gathered = comm.gather((rank, len(my_files), t_local, local), root=0)
    t_total = MPI.Wtime() - t_start

    if rank == 0:
        total = Counter()
        for _, _, _, c in gathered:
            total.update(c)
        times = [g[2] for g in gathered]
        mean_t = sum(times) / len(times)
        print(f"MPI version 1 | processes={size} | files={len(files)}")
        for r, n, t, _ in sorted(gathered):
            print(f"  rank {r}: files={n} local_time={t:.6f}s")
        print(f"LI={max(times) / mean_t:.4f}")
        print(f"SPREAD={(max(times) - min(times)) / mean_t:.4f}")
        print(f"EXECUTION_TIME={t_total:.6f}")
        print(f"Total occurrences: {sum(total.values())}")
        print(f"Top {TOP_N} query words:")
        for w, c in total.most_common(TOP_N):
            print(f"  {w}: {c}")


if __name__ == "__main__":
    main()
