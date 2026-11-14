# Lab 2 – Parallel Word Counting in a Text Corpus with MPI

## 1. Problem Description
Count how many times each word of `dataset/consulta.txt` (200 words) appears in a corpus of 3000 files `dataset/file_XXXX.txt` (~44.9 M tokens, 262 MB, file sizes 12 KB–609 KB) and report the top 10. Three implementations are compared: the provided sequential baseline, MPI v1 (broadcast + static distribution) and MPI v2 (dynamic master–worker).

## 2. Environment and Execution Instructions
- Host: Windows 11, 8 physical cores / 16 logical threads, Docker Desktop (WSL2), image `augustosalazar/slim-mpi:2` (Python 3.8, Open MPI 4.1.4, mpi4py). Dataset on a bind-mounted OneDrive folder.
- **p = 8** (one process per physical core). The `Detectar_procesos_mpi` script was not available in the repo or the image, so p was chosen as the physical core count (`nproc` inside Docker reports 16 logical threads).
- Run everything: `docker run --rm -e NP_LIST=8 -e RUNS=3 -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh`
- `generator.py`, `baseline_secuencial.py`, `run_all.sh` are unmodified.

### 2a. Experimental plan
Same dataset and same p=8 for all programs; 3 runs per program; correctness checked against the baseline top 10 and total (3,631,778 occurrences); metrics: T_seq (mean of 3 baseline runs), T_p (mean of 3 runs), S_p = T_seq/T_p, E_p = S_p/p, LI = max(t_i)/mean(t_i), spread = (max−min)/mean. Order: baseline → v1 → v2. Raw logs in `results/logs/`, summary in `results/times_summary.csv`.

### 2b. Sequential baseline
Loads the query set, iterates over every `file_*.txt`, lower-cases and splits each line, and counts tokens in the query set with a `Counter`. Prints elapsed time and the top 10.

### 2c. MPI version 1 (`mpi1.py`)
Rank 0 reads the query and lists/sorts the files; the query is sent with `comm.bcast`; files are split into 8 contiguous blocks of 375 files (static) and sent with `comm.scatter`; every rank counts locally (same tokenization as the baseline) and times it; partial Counters are collected with `comm.gather` in rank 0, which merges them and prints top 10, per-rank file count/local time, LI and total time (`EXECUTION_TIME`, measured from a barrier to the end of the gather).

### 2d. Test procedure
Output of v1/v2 (top 10 and total occurrences) compared with the baseline: identical in all runs. Times extracted from `EXECUTION_TIME=` lines by `run_all.sh`.

## 3. Experimental Plan Execution

### 3a. Sequential baseline
| Run | Time (s) |
|---|---|
| 1 (inside run_all.sh, cold cache) | 47.093 |
| 2 | 37.229 |
| 3 | 34.604 |
| **T_seq (mean)** | **39.642** |

Top 10: a 785774, para 392156, sus 228913, otros 105530, ante 99832, unos 88794, otra 83901, vosotros 61617, mios 58420, tuya 56635.

### 3b. MPI v1 timing (p = 8)
| Run | Total time (s) |
|---|---|
| 1 | 6.993 |
| 2 | 7.496 |
| 3 | 13.716 |
| **T_p (mean)** | **9.402** |

S_p = 39.642 / 9.402 = **4.22**, E_p = 4.22 / 8 = **0.53**.
(Without the run-3 outlier, T_p = 7.244 s, S_p = 5.47, E_p = 0.68.)

### 3c. Load imbalance evidence (v1, local times in s; every rank had 375 files)
| Rank | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| 0 | 6.953 | 7.423 | 13.653 |
| 1 | 5.565 | 6.457 | 12.938 |
| 2 | 6.203 | 7.473 | 13.611 |
| 3 | 6.136 | 7.379 | 13.685 |
| 4 | 5.806 | 6.887 | 13.220 |
| 5 | 5.889 | 7.088 | 13.353 |
| 6 | 5.801 | 6.962 | 13.346 |
| 7 | 6.040 | 7.224 | 13.539 |
| **LI** | **1.150** | **1.051** | **1.020** |
| spread | 0.230 | 0.143 | 0.056 |

Mean LI = **1.073**. Rank 0 was consistently among the slowest (it also does the file listing/scatter and shares CPU with Docker/OS work); rank 1 the fastest.

### 3d. MPI v2 (`mpi2.py`): correction and results
**How it corrects imbalance:** rank 0 is a master that broadcasts the query and then hands out one file at a time to whichever worker requests work (`send/recv`, tags REQUEST/WORK/RESULT; `None` = stop). Workers (ranks 1–7) count, accumulate local time, and send their Counter at the end. A worker that finishes early simply receives more files (workers processed 398–465 files), so differences in file size or CPU speed cannot leave a rank idle at the end. Trade-off: only 7 of the 8 processes do counting, and there is one message round-trip per file. (Sorting files by size was tried and discarded: `os.stat` on the Docker bind mount cost ~4.5 s for 3000 files.)

| Run | Total time (s) | LI | spread |
|---|---|---|---|
| 1 | 6.613 | 1.0067 | 0.0095 |
| 2 | 6.573 | 1.0068 | 0.0100 |
| 3 | 5.960 | 1.0045 | 0.0071 |
| **Mean** | **6.382** | **1.006** | 0.0089 |

S_p = 39.642 / 6.382 = **6.21**, E_p = 6.21 / 8 = **0.78** (computed with p = 8 processes launched, to be comparable with v1).

### Comparison
| | T_p (s) | S_p | E_p | LI |
|---|---|---|---|---|
| v1 static | 9.402 (7.244 w/o outlier) | 4.22 (5.47) | 0.53 (0.68) | 1.073 |
| v2 dynamic | 6.382 | 6.21 | 0.78 | 1.006 |

## 4. Analysis

### 4a. Parallel performance
Yes: both MPI versions beat the baseline (v1 4.2×, v2 6.2× with 8 processes), but neither reaches the ideal S_p = 8. Gap causes: (1) the problem is largely I/O-bound and the data sits on a Docker bind mount over OneDrive/WSL2, so 8 processes compete for the same file-system path; (2) 8 processes are launched in a container sharing the host with Windows, Docker and OneDrive; (3) serial parts (listing 3000 files, scatter/gather, merging); (4) in v2 one process only coordinates, so at most 7/8 of the compute capacity is used (ideal ≈ 7); (5) the baseline's own time varies widely (34.6–47.1 s), partly cold cache.

### 4b. Load balance
v1 showed mild imbalance: LI 1.15 / 1.05 / 1.02 (mean 1.07), with run 1 spread of 23 % between the fastest and slowest rank. Static blocks give identical file counts (375) but unequal work (files vary 12–609 KB), plus rank 0's extra duties. v2 brought LI to 1.004–1.007. Its mean time was 32 % lower than v1 (6.38 vs 9.40 s), **but this improvement is not solid**: v1 run 3 (13.7 s) was slow on *all* ranks (LI only 1.02), i.e. system noise, not imbalance. Comparing only the quiet runs, v1 ≈ 7.0–7.5 s vs v2 ≈ 6.0–6.6 s, a smaller (≈10 %) but still consistent advantage that matches the LI reduction. Conclusion: v2 clearly fixes the balance; the real speedup gain is positive but modest and not statistically proven with 3 runs.

### 4c. Limitations
- Only 3 runs per configuration and large run-to-run variance (v1 run 3, baseline 34–47 s): conclusions on small differences are weak.
- Single p (8); no scaling curve. p chosen manually because `Detectar_procesos_mpi` was unavailable.
- Docker Desktop/WSL2 over an OneDrive bind mount: file-system latency and background processes add noise; the host was not dedicated. Hyper-threading (16 logical) not explored.
- T_seq includes a cold-cache first run; using the 2 warm runs (35.9 s) would lower the speedups by ~9 %.
- Synthetic, fairly homogeneous dataset (random Spanish words), so static imbalance is mild; results may differ with highly skewed file sizes.
- Python/mpi4py pickling and the interpreter's speed are not representative of compiled code.

## 5. Conclusions
For this problem I would recommend the dynamic master–worker distribution (v2): with p = 8 it reached S_p = 6.21 and E_p = 0.78 against 4.22 and 0.53 for the static version, and it reduced LI from a mean of 1.073 (up to 1.15) to 1.006, because workers pull files as they finish and never wait for a slow block. The caveat is that the measurement noise (3 runs, shared Docker host, one outlier in v1) means that the true gain is probably nearer 10 % than 32 %, and that v2 spends one process on coordination; if the file sizes were uniform and the I/O stable, the simpler static scheme would be nearly as good.
