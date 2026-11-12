import ctypes
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import platform
from pathlib import Path
import statistics
import sys
import time

from harness import run_benchmark, emit_verify_tag


# ---------- ЗАМЕНИ СВОИМИ ДАННЫМИ ----------

STUDENT_ID = 230103222
PPHYS = 6

CPU_MODEL = "Вставь название CPU"
L1D_DESCRIPTION = "Вставь L1 Data на одно ядро или: unavailable"
L2_DESCRIPTION = "Вставь размер и область: на ядро / общий"
L3_DESCRIPTION = "Вставь размер L3"

# Если известно, укажи в байтах. Иначе оставь None.
CACHE_LINE_BYTES = None
L1D_BYTES = None
L2_BYTES = None

# -----------------------------------------

PLOG = mp.cpu_count()
ROOT = Path(__file__).resolve().parent

# Phase 1 требует именно 2 warm-up + 7 измерений.
# Для остальных этапов используем стандартные 3 + 9 из harness.
WARMUP = 3
REPEATS = 9


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [
            f"{value:.6f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        print("| " + " | ".join(cells) + " |")


def verify(phase, first, second):
    print(emit_verify_tag(phase, STUDENT_ID, first, second))


def run_processes(target, arguments):
    processes = [
        mp.Process(target=target, args=args)
        for args in arguments
    ]
    for process in processes:
        process.start()

    for process in processes:
        process.join()

    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("A worker process failed")


def make_shm(size):
    block = shared_memory.SharedMemory(create=True, size=size)
    return block


def destroy(block):
    block.close()
    block.unlink()


# ==================== PHASE 0 ====================

def phase0():
    print("# Phase 0: Hardware and timer")
    table(
        ["Parameter", "Value"],
        [
            ["Student ID", STUDENT_ID],
            ["CPU model", CPU_MODEL],
            ["Physical cores", PPHYS],
            ["Logical processors", PLOG],
            ["Cache line bytes", CACHE_LINE_BYTES],
            ["L1d per core", L1D_DESCRIPTION],
            ["L2", L2_DESCRIPTION],
            ["L3", L3_DESCRIPTION],
            ["Python", platform.python_version()],
            ["OS", platform.platform()],
        ],
    )

    reads = 2_000_000
    clock = time.perf_counter_ns

    start = clock()
    for _ in range(reads):
        clock()
    elapsed = clock() - start

    average_ns = elapsed / reads

    print(f"Clock reads: {reads}")
    print(f"Average clock-read loop latency: {average_ns:.6f} ns")
    print("This estimate includes Python loop overhead.")

    # The PDF does not specify the two Phase 0 metrics.
    # Here they are explicitly defined as latency and PLOG.
    print("Hash metrics: average_ns, PLOG")
    verify("PHASE0", average_ns, float(PLOG))


# ==================== PHASE 1 ====================

def fs_worker(name, slot, iterations):
    block = shared_memory.SharedMemory(name=name)
    values = (ctypes.c_uint64 * 512).from_buffer(block.buf)

    acc = values[slot]
    mask = 0xFFFFFFFFFFFFFFFF
    multiplier = 6364136223846793005

    for i in range(iterations):
        acc = (acc * multiplier + (i | 1)) & mask
        values[slot] = acc

    del values
    block.close()


def phase1():
    print("# Phase 1: False sharing")
    total_iterations = 24_000_000
    strides = [1, 2, 4, 8, 16, 32]

    if (max(PPHYS, PLOG) - 1) * 32 >= 512:
        raise ValueError(
            "4096-byte buffer is too small for this CPU and stride. "
            "Clarify the buffer-size requirement with the instructor."
        )

    block = make_shm(4096)
    cache = {}

    def measure(p, stride):
        key = (p, stride)
        if key in cache:
            return cache[key]

        def trial():
            block.buf[:] = bytes(4096)
            quotient, remainder = divmod(total_iterations, p)
            arguments = [
                (
                    block.name,
                    worker * stride,
                    quotient + int(worker < remainder),
                )
                for worker in range(p)
            ]
            run_processes(fs_worker, arguments)

        result = run_benchmark(trial, warmup_runs=2, timed_runs=7)
        cache[key] = result
        return result

    try:
        stride_rows = []
        for stride in strides:
            stats = measure(PPHYS, stride)
            stride_rows.append([
                stride,
                stride * 8,
                stats.median_ms,
                stats.iqr_ms,
                total_iterations / (stats.median_ms * 1000),
            ])

        print("## Table 1A")
        table(
            ["Stride", "Bytes", "Median ms", "IQR ms", "Mops/s"],
            stride_rows,
        )

        counts = {1, PLOG}
        p = 2
        while p <= PLOG:
            counts.add(p)
            p *= 2

        print("## Table 1B")
        scaling_rows = []

        for stride in [1, 16]:
            baseline = measure(1, stride).median_ms

            for p in sorted(counts):
                stats = measure(p, stride)
                speedup = baseline / stats.median_ms
                efficiency = speedup / p
                karp_flatt = (
                    (1 / speedup - 1 / p) / (1 - 1 / p)
                    if p > 1 else "N/A"
                )
                scaling_rows.append([
                    stride, p, stats.median_ms,
                    stats.iqr_ms, speedup, efficiency, karp_flatt,
                ])

        table(
            ["Stride", "p", "Median ms", "IQR ms", "S", "E", "e(p)"],
            scaling_rows,
        )

        drops = [
            (
                100 * (stride_rows[i - 1][2] - stride_rows[i][2])
                / stride_rows[i - 1][2],
                stride_rows[i][1],
            )
            for i in range(1, len(stride_rows))
        ]
        drop, offset = max(drops)

        print(
            f"Largest adjacent timing reduction: {drop:.2f}% "
            f"at separation {offset} bytes."
        )
        print(
            "This is a candidate transition, not proof of cache-line size. "
            "Compare it with IQR and the OS hardware data."
        )
        print(
            "False sharing can increase e(p) because workers write "
            "different words belonging to the same coherence line. "
            "Padding reduces invalidations. Process startup and Python "
            "execution overhead can mask the effect."
        )

        verify(
            "PHASE1",
            measure(PPHYS, 1).median_ms,
            measure(PPHYS, 16).median_ms,
        )
    finally:
        destroy(block)


# ==================== PHASE 2 ====================

def serial_stage(n):
    # Six sin + six sqrt calls = 12 transcendental operations.
    prefix = 0.0
    filtered = 0.0

    for i in range(n // 20):
        x = 0.001 * (i % 1000 + 1)
        for _ in range(6):
            x = math.sqrt(1.0 + abs(math.sin(x)))
        prefix += x
        if prefix > 0.0:
            filtered += x

    return filtered


def parallel_stage(bounds):
    start, end = bounds
    result = 0.0

    for i in range(start, end):
        x = 0.001 * (i % 1000 + 1)
        for _ in range(35):
            x = x * 0.999999 + 0.000001
            x = math.sin(x)
        result += x

    return result


def scaling_measure(pool, p, n):
    serial_times = []
    checksum = [0.0]

    def trial():
        start = time.perf_counter_ns()
        serial_value = serial_stage(n)
        serial_times.append((time.perf_counter_ns() - start) / 1e6)

        bounds = [
            (worker * n // p, (worker + 1) * n // p)
            for worker in range(p)
        ]
        values = pool.map(parallel_stage, bounds, chunksize=1)
        checksum[0] = serial_value + sum(values)

    stats = run_benchmark(trial, WARMUP, REPEATS)
    serial_ms = statistics.median(serial_times[WARMUP:])

    return stats, serial_ms, checksum[0]


def phase2():
    print("# Phase 2: Strong and weak scaling")
    strong = {}
    weak = {}

    for p in range(1, PLOG + 1):
        # Pool creation is outside the timed region.
        with mp.Pool(processes=p) as pool:
            strong[p] = scaling_measure(pool, p, 1_600_000)
            weak[p] = scaling_measure(pool, p, p * 250_000)

        print(f"Completed p={p}", flush=True)

    strong_base = strong[1][0].median_ms
    weak_base = weak[1][0].median_ms

    strong_rows = []
    weak_rows = []
    speedups = {}

    for p in range(1, PLOG + 1):
        stats, serial_ms, checksum = strong[p]
        speedup = strong_base / stats.median_ms
        speedups[p] = speedup

        strong_rows.append([
            p, 1_600_000, stats.median_ms, stats.iqr_ms,
            serial_ms, speedup, speedup / p, checksum,
        ])

        stats, serial_ms, checksum = weak[p]
        alpha = serial_ms / stats.median_ms
        gustafson = p - alpha * (p - 1)

        weak_rows.append([
            p, p * 250_000, stats.median_ms, stats.iqr_ms,
            serial_ms, weak_base / stats.median_ms,
            alpha, gustafson, checksum,
        ])

    print("## Strong scaling")
    table(
        ["p", "N", "Median ms", "IQR ms", "Stage A ms",
         "S", "E", "Checksum"],
        strong_rows,
    )

    print("## Weak scaling")
    table(
        ["p", "N", "Median ms", "IQR ms", "Stage A ms",
         "Weak E", "Alpha", "Gustafson S", "Checksum"],
        weak_rows,
    )

    serial_fraction = strong[1][1] / strong_base
    maximum = 1 / serial_fraction
    theoretical = 1 / (
        serial_fraction + (1 - serial_fraction) / PLOG
    )
    observed = speedups[PLOG]

    print(f"Measured serial fraction s: {serial_fraction:.8f}")
    print(f"Amdahl asymptotic ceiling: {maximum:.6f}")
    print(f"Amdahl prediction at PLOG: {theoretical:.6f}")
    print(f"Observed speedup at PLOG: {observed:.6f}")
    print(
        "Observed versus predicted difference (%): "
        f"{100 * (observed / theoretical - 1):.3f}"
    )

    if 1 < PPHYS < PLOG:
        before = speedups[PPHYS] - speedups[PPHYS - 1]
        after = speedups[PPHYS + 1] - speedups[PPHYS]

        print(f"Marginal speedup before boundary: {before:.6f}")
        print(f"Marginal speedup after boundary: {after:.6f}")
        if before > 0:
            print(
                "Marginal speedup reduction (%): "
                f"{100 * (before - after) / before:.3f}"
            )
        else:
            print(
                "Percentage reduction is not meaningful because "
                "the preceding marginal gain is non-positive."
            )

    print(
        "Strong scaling keeps N fixed; weak scaling keeps N/p fixed. "
        "Serial work, scheduling, IPC and resource contention limit speedup. "
        "Here Stage A also grows with N in weak scaling."
    )
    print(
        "Without CPU affinity, the physical-core boundary does not prove "
        "an exact transition from one thread per core to SMT. "
        "Hybrid CPUs require separate core-type information."
    )

    verify("PHASE2", observed, weak_rows[-1][7])


# ==================== PHASE 3A ====================

def copy_worker(a_name, b_name, start, end):
    a = shared_memory.SharedMemory(name=a_name)
    b = shared_memory.SharedMemory(name=b_name)

    a.buf[start:end] = b.buf[start:end]

    a.close()
    b.close()


def compute_worker(iterations):
    a = 0.9999999
    b = 0.0000001

    x0 = x1 = x2 = x3 = 0.1
    x4 = x5 = x6 = x7 = 0.2
    x8 = x9 = x10 = x11 = 0.3
    x12 = x13 = x14 = x15 = 0.4

    for _ in range(iterations):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b

    return sum((
        x0, x1, x2, x3, x4, x5, x6, x7,
        x8, x9, x10, x11, x12, x13, x14, x15,
    ))


def roofline_measure():
    size = 96_000_000
    a = make_shm(size)
    b = make_shm(size)

    try:
        # Touch pages before measurement.
        a.buf[:] = b"\x00" * size
        b.buf[:] = b"\x01" * size

        with mp.Pool(PPHYS) as pool:
            arguments = [
                (
                    a.name, b.name,
                    worker * size // PPHYS,
                    (worker + 1) * size // PPHYS,
                )
                for worker in range(PPHYS)
            ]

            copy_stats = run_benchmark(
                lambda: pool.starmap(copy_worker, arguments, chunksize=1),
                WARMUP, REPEATS,
            )

            # 16 multiply-add expressions = 32 conventional FLOPs.
            iterations = 80_000_000 // 32
            compute_stats = run_benchmark(
                lambda: pool.map(
                    compute_worker,
                    [iterations] * PPHYS,
                    chunksize=1,
                ),
                WARMUP, REPEATS,
            )

        bandwidth = 2 * size / (copy_stats.median_ms / 1000) / 1e9
        compute = (
            80_000_000 * PPHYS
            / (compute_stats.median_ms / 1000) / 1e9
        )

        table(
            ["Metric", "Value"],
            [
                ["Copy median ms", copy_stats.median_ms],
                ["Copy IQR ms", copy_stats.iqr_ms],
                ["Effective copy GB/s", bandwidth],
                ["Compute median ms", compute_stats.median_ms],
                ["Compute IQR ms", compute_stats.iqr_ms],
                ["Python GFLOP/s", compute],
                ["Ridge FLOP/byte", compute / bandwidth],
            ],
        )
        print(
            "Copy timing includes Pool dispatch and shared-memory attachment. "
            "2*size counts useful read/write traffic; actual DRAM traffic "
            "may differ because of caches and write allocation."
        )
        print(
            "Python x*a+b is not a hardware fused-FMA instruction. "
            "This is an interpreter compute ceiling."
        )
        return bandwidth, compute

    finally:
        destroy(a)
        destroy(b)


# ==================== PHASE 3B ====================

def gemm_worker(a_name, b_name, c_name, n, lo, hi, variant, bs):
    a_block = shared_memory.SharedMemory(name=a_name)
    b_block = shared_memory.SharedMemory(name=b_name)
    c_block = shared_memory.SharedMemory(name=c_name)

    a = a_block.buf.cast("d")
    b = b_block.buf.cast("d")
    c = c_block.buf.cast("d")

    # Each worker owns distinct output rows.
    for index in range(lo * n, hi * n):
        c[index] = 0.0

    if variant == 1:
        for i in range(lo, hi):
            row = i * n
            for j in range(n):
                value = 0.0
                for k in range(n):
                    value += a[row + k] * b[k * n + j]
                c[row + j] = value

    elif variant == 2:
        for i in range(lo, hi):
            row = i * n
            for k in range(n):
                aik = a[row + k]
                brow = k * n
                for j in range(n):
                    c[row + j] += aik * b[brow + j]

    else:
        for ii in range(lo, hi, bs):
            for jj in range(0, n, bs):
                for kk in range(0, n, bs):
                    jend = min(jj + bs, n)
                    width = jend - jj
                    for i in range(ii, min(ii + bs, hi)):
                        row = i * n
                        c_slice = c[row + jj:row + jend]
                        for k in range(kk, min(kk + bs, n)):
                            aik = a[row + k]
                            b_slice = b[k * n + jj:k * n + jend]
                            for j in range(width):
                                c_slice[j] += aik * b_slice[j]
                            b_slice.release()
                        c_slice.release()

    a.release()
    b.release()
    c.release()

    a_block.close()
    b_block.close()
    c_block.close()


def phase3():
    print("# Phase 3: Roofline and GEMM")
    bandwidth, ceiling = roofline_measure()

    n = 384
    length = n * n
    blocks = [make_shm(length * 8) for _ in range(3)]
    a_block, b_block, c_block = blocks

    a = a_block.buf.cast("d")
    b = b_block.buf.cast("d")
    c = c_block.buf.cast("d")

    for index in range(length):
        a[index] = (index % 97) / 97.0
        b[index] = (index % 89) / 89.0
        c[index] = 0.0

    results = []

    try:
        with mp.Pool(PPHYS) as pool:
            def measure(variant, bs=0):
                arguments = [
                    (
                        a_block.name, b_block.name, c_block.name,
                        n, worker * n // PPHYS,
                        (worker + 1) * n // PPHYS,
                        variant, bs,
                    )
                    for worker in range(PPHYS)
                ]
                stats = run_benchmark(
                    lambda: pool.starmap(
                        gemm_worker, arguments, chunksize=1
                    ),
                    WARMUP, REPEATS,
                )
                gflops = 2 * n ** 3 / (stats.median_ms / 1000) / 1e9
                return stats, gflops

            s1, g1 = measure(1)
            naive_output = list(c)

            s2, g2 = measure(2)
            reference = list(c)
            error = max(
                abs(x - y)
                for x, y in zip(naive_output, reference)
            )
            assert error < 1e-9, error

            results.extend([
                ["Naive", "-", s1.median_ms, s1.iqr_ms, g1, "-", error],
                ["Reordered", "-", s2.median_ms, s2.iqr_ms, g2, "-", 0.0],
            ])

            tiled_results = []
            for bs in [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
                stats, gflops = measure(3, bs)
                error = max(
                    abs(c[index] - reference[index])
                    for index in range(length)
                )
                assert error < 1e-9, (bs, error)

                footprint = 3 * bs * bs * 8
                results.append([
                    "Tiled", bs, stats.median_ms, stats.iqr_ms,
                    gflops, footprint, error,
                ])
                tiled_results.append((gflops, bs, footprint))

        table(
            ["Variant", "BS", "Median ms", "IQR ms",
             "GFLOP/s", "Tile bytes", "Max error"],
            results,
        )

        best_gflops, best_bs, footprint = max(tiled_results)
        print(f"Best BS: {best_bs}")
        print(f"Best tile footprint: {footprint} bytes")

        if L1D_BYTES:
            print(f"Tile footprint / L1d: {footprint / L1D_BYTES:.4f}")
        if L2_BYTES:
            print(f"Tile footprint / L2: {footprint / L2_BYTES:.4f}")

        # The PDF gives no operational-intensity formula for Variant 2.
        # State an explicit approximate assumption rather than hide it.
        mappings = [
            ("Naive", g1, 0.125),
            ("Reordered", g2, 1 / 12),
            ("Best tiled", best_gflops, best_bs / 12),
        ]

        roof_rows = []
        for name, measured, intensity in mappings:
            attainable = min(ceiling, bandwidth * intensity)
            roof_rows.append([
                name, intensity, attainable,
                measured, 100 * measured / attainable,
            ])

        table(
            ["Variant", "Assumed FLOP/byte", "Roof GFLOP/s",
             "Measured GFLOP/s", "Percent of roof"],
            roof_rows,
        )
        print(
            "Reordered intensity 1/12 is a simplified inner-loop model: "
            "8-byte B read + 8-byte C read + 8-byte C write per 2 FLOPs. "
            "It is not measured DRAM intensity. "
            "The three mappings are approximate, not hardware-counter results."
        )
        print(
            "Tiling trades data locality against extra Python loops, "
            "indexing and memoryview operations. The best tile does not "
            "have to fit exactly into a nominal cache capacity."
        )

        verify("PHASE3", g1, best_gflops)

    finally:
        a.release()
        b.release()
        c.release()
        for block in blocks:
            destroy(block)


# ==================== PHASE 4 ====================

def skew_item(i):
    iterations = 1 + i * i // 3600
    value = 0.001 * (i + 1)

    for _ in range(iterations):
        value = math.sqrt(
            1.0 + abs(math.sin(value) + math.cos(value))
        )

    return value


def scheduling_worker(
    worker, p, strategy, chunk, counter, active, sums, ready, start_event
):
    n = 12_000
    total = 0.0

    ready.release()
    start_event.wait()
    start = time.perf_counter_ns()

    if strategy == "block":
        indices = range(worker * n // p, (worker + 1) * n // p)
        for i in indices:
            total += skew_item(i)

    elif strategy == "cyclic":
        for i in range(worker, n, p):
            total += skew_item(i)

    else:
        while True:
            with counter.get_lock():
                first = counter.value
                counter.value += chunk

            if first >= n:
                break

            for i in range(first, min(first + chunk, n)):
                total += skew_item(i)

    active[worker] = (time.perf_counter_ns() - start) / 1e6
    sums[worker] = total


def schedule_trial(strategy, chunk):
    p = PPHYS
    counter = mp.Value("i", 0)
    active = mp.Array("d", p, lock=False)
    sums = mp.Array("d", p, lock=False)
    ready = mp.Semaphore(0)
    event = mp.Event()

    workers = [
        mp.Process(
            target=scheduling_worker,
            args=(
                worker, p, strategy, chunk,
                counter, active, sums, ready, event,
            ),
        )
        for worker in range(p)
    ]

    # End-to-end wall time includes process creation and joining.
    start = time.perf_counter_ns()

    for worker in workers:
        worker.start()
    for _ in workers:
        ready.acquire()

    event.set()

    for worker in workers:
        worker.join()

    wall_ms = (time.perf_counter_ns() - start) / 1e6
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError("Scheduling worker failed")

    times = list(active)
    imbalance = max(times) / statistics.mean(times)

    return wall_ms, imbalance, sum(sums), times


def phase4():
    print("# Phase 4: Scheduling")
    configurations = [
        ("block", 0),
        ("cyclic", 0),
        *[("dynamic", size) for size in [1, 4, 16, 64, 256, 1024]],
    ]

    results = []
    reference_checksum = None

    for strategy, chunk in configurations:
        for _ in range(WARMUP):
            schedule_trial(strategy, chunk)

        samples = [
            schedule_trial(strategy, chunk)
            for _ in range(REPEATS)
        ]

        print(f"## Raw runs: {strategy}, C={chunk}")
        for index, sample in enumerate(samples, start=1):
            wall, imbalance, checksum, times = sample
            print(
                f"run={index}, wall_ms={wall:.6f}, "
                f"lambda={imbalance:.6f}, checksum={checksum:.9f}, "
                f"worker_ms={times}"
            )

            if reference_checksum is None:
                reference_checksum = checksum
            assert math.isclose(
                checksum, reference_checksum,
                rel_tol=1e-10, abs_tol=1e-8,
            )

        ordered = sorted(samples, key=lambda item: item[0])
        representative = ordered[len(ordered) // 2]
        wall, imbalance, checksum, times = representative
        iqr = (
            ordered[3 * len(ordered) // 4][0]
            - ordered[len(ordered) // 4][0]
        )

        # Lambda and worker times belong to the actual median-wall run.
        results.append([
            strategy, chunk, wall, iqr, imbalance, checksum
        ])

    print("## Summary: median-wall run")
    table(
        ["Strategy", "C", "Wall ms", "IQR ms", "Lambda", "Checksum"],
        results,
    )

    best = min(
        (row for row in results if row[0] == "dynamic"),
        key=lambda row: row[2],
    )

    print(f"Best dynamic chunk C*: {best[1]}")
    print(f"Static block wall time: {results[0][2]:.6f} ms")
    print(f"Best dynamic wall time: {best[2]:.6f} ms")
    print(f"Static/best-dynamic time ratio: {results[0][2] / best[2]:.6f}")

    print(
        "Block scheduling assigns the expensive high-index items "
        "to the last workers. Cyclic scheduling spreads them across workers. "
        "Dynamic C=1 balances the tail but requires many lock acquisitions. "
        "C=1024 reduces synchronization but leaves large indivisible chunks. "
        "C* is the chunk with the lowest measured median wall time."
    )
    print(
        "Worker active time includes computation and queue-lock waiting, "
        "but excludes waiting at the initial start event."
    )

    verify("PHASE4", results[0][2], best[2])


# ==================== REPORT ====================

class Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def build_report():
    required = [ROOT / f"phase{i}.txt" for i in range(5)]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Run all phases first. Missing: {missing}")

    report = ROOT / f"lab_report_{STUDENT_ID}.md"

    with report.open("w", encoding="utf-8") as output:
        output.write(f"# Parallel Computing Lab — {STUDENT_ID}\n\n")

        for path in required:
            output.write(path.read_text(encoding="utf-8"))
            output.write("\n\n")

        answers = ROOT / "answers.txt"
        if answers.exists():
            output.write("# Additional analytical answers\n\n")
            output.write(answers.read_text(encoding="utf-8"))
            output.write("\n\n")

        output.write("# Complete Python source code\n\n")
        for filename in ["harness.py", "lab.py"]:
            output.write(f"## {filename}\n\n```python\n")
            output.write((ROOT / filename).read_text(encoding="utf-8"))
            output.write("\n```\n\n")

    print(f"Report saved: {report}")


def main():
    if not 1 <= PPHYS <= PLOG:
        raise ValueError("Check PPHYS against your actual hardware")

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python lab.py 0|1|2|3|4|report")

    choice = sys.argv[1]

    if choice == "report":
        build_report()
        return

    phases = {
        "0": phase0,
        "1": phase1,
        "2": phase2,
        "3": phase3,
        "4": phase4,
    }
    if choice not in phases:
        raise SystemExit("Choose 0, 1, 2, 3, 4 or report")

    log_path = ROOT / f"phase{choice}.txt"
    terminal = sys.stdout

    with log_path.open("w", encoding="utf-8") as log:
        sys.stdout = Tee(terminal, log)
        try:
            phases[choice]()
            print("STATUS: COMPLETED")
        finally:
            sys.stdout = terminal


if __name__ == "__main__":
    mp.freeze_support()
    main()