from dataclasses import dataclass
import multiprocessing as mp
import time


@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float


def fnv1a_hash(s):
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run_benchmark(fn, warmup_runs=3, timed_runs=9):
    for _ in range(warmup_runs):
        fn()

    samples = []
    for _ in range(timed_runs):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        samples.append((end - start) / 1e6)

    samples.sort()
    middle = timed_runs // 2
    median = (
        samples[middle]
        if timed_runs % 2
        else (samples[middle - 1] + samples[middle]) / 2
    )
    q25 = samples[timed_runs // 4]
    q75 = samples[3 * timed_runs // 4]

    return BenchStats(samples[0], median, q75 - q25, samples[-1])


def emit_verify_tag(task_id, student_seed, primary_metric, secondary_metric):
    logical = mp.cpu_count()
    payload = (
        f"{task_id}|{student_seed}|{logical}|"
        f"{primary_metric:.4f}|{secondary_metric:.4f}"
    )
    signature = fnv1a_hash(payload)
    return (
        f"VERIFY_HASH[{task_id}]: {signature:016X} "
        f"(P_log={logical})"
    )