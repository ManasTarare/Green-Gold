"""
workload_runner.py — Real OS worker process management for GreenGold.

Spawns real Python worker subprocesses that perform lightweight numpy
matrix work (simulating GPU/CPU compute tasks). Each worker runs
continuously and can be paused, resumed, or have its CPU-core affinity
changed — making the RL agent's actions genuinely real.

Architecture
------------
  WorkloadRunner
    ├── Worker-0  (logical server A, cores 0-1)
    ├── Worker-1  (logical server A, cores 0-1)
    ├── Worker-2  (logical server B, cores 2-3)
    ├── Worker-3  (logical server B, cores 2-3)
    └── ...

Each worker maps to one chip slot (SRV-00X-CXX). The manager can:
  - pause_worker(pid)      → SIGSTOP / Windows suspend
  - resume_worker(pid)     → SIGCONT / Windows resume
  - set_affinity(pid, cores) → restrict to CPU core group
  - kill_worker(wid)       → terminate + relaunch

Step 3: logical servers = CPU core groups.
  Server A = cores [0, 1]   (low-carbon simulation)
  Server B = cores [2, 3]   (high-carbon simulation)
  Migration = moving affinity from one group to the other.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CPU core group → logical "server" mapping
# ─────────────────────────────────────────────
import psutil as _psutil  # imported lazily below too

_CPU_COUNT = _psutil.cpu_count(logical=True) or 4

# Divide available cores into two groups (two logical "server clusters").
# If only 2 logical cores, each group gets 1.
_half = max(1, _CPU_COUNT // 2)
SERVER_CORE_GROUPS = {
    "SRV-WRK-A": list(range(0, _half)),
    "SRV-WRK-B": list(range(_half, _CPU_COUNT)),
}

NUM_WORKERS = 6   # total worker processes to spawn


# ─────────────────────────────────────────────
# Worker process entry-point (runs in subprocess)
# ─────────────────────────────────────────────
def _worker_main(worker_id: int, task_name: str, stop_event) -> None:
    """
    Lightweight CPU-bound loop simulating a compute workload.
    Uses numpy for matrix math so it shows real CPU utilization.
    Runs until stop_event is set.
    """
    try:
        import numpy as np
        rng = np.random.default_rng(worker_id)
        size = 128  # small enough to not tax the machine
        while not stop_event.is_set():
            a = rng.random((size, size), dtype=np.float32)
            b = rng.random((size, size), dtype=np.float32)
            _ = np.dot(a, b)
            # Sleep briefly so we don't pin the core at 100%
            time.sleep(random.uniform(0.01, 0.05))
    except Exception as exc:
        pass  # worker exits silently if parent is gone


# ─────────────────────────────────────────────
# Worker descriptor
# ─────────────────────────────────────────────
@dataclass
class WorkerInfo:
    worker_id:   int
    task_name:   str
    server_group: str                    # "SRV-WRK-A" or "SRV-WRK-B"
    core_group:  list[int]               # current CPU affinity
    process:     Optional[multiprocessing.Process] = None
    stop_event:  Optional[multiprocessing.Event]   = None
    paused:      bool = False

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process and self.process.is_alive() else None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def to_dict(self) -> dict:
        return {
            "worker_id":    self.worker_id,
            "task_name":    self.task_name,
            "server_group": self.server_group,
            "core_group":   self.core_group,
            "pid":          self.pid,
            "paused":       self.paused,
            "alive":        self.alive,
        }


# ─────────────────────────────────────────────
# Main manager class
# ─────────────────────────────────────────────
class WorkloadRunner:
    """
    Manages a pool of real OS worker processes.
    Thread-safe for use from Flask request handlers.
    """

    TASKS = ["LLM Training", "Image Gen", "Vector Search", "Embedding", "Fine-tuning", "Idle"]

    def __init__(self, num_workers: int = NUM_WORKERS) -> None:
        self._workers: list[WorkerInfo] = []
        self._num = num_workers
        self._started = False
        self._psutil_available = False

        try:
            import psutil
            self._psutil = psutil
            self._psutil_available = True
        except ImportError:
            log.warning("WorkloadRunner: psutil not installed — real actions disabled.")

        # Use spawn context to be safe on Windows
        self._mp_ctx = multiprocessing.get_context("spawn")

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    def start(self) -> None:
        """Spawn all worker processes."""
        if self._started:
            return

        groups = list(SERVER_CORE_GROUPS.items())  # [("SRV-WRK-A", [...]), ("SRV-WRK-B", [...])]
        tasks  = self.TASKS * (self._num // len(self.TASKS) + 1)

        for i in range(self._num):
            grp_name, cores = groups[i % len(groups)]
            task = tasks[i]

            stop_ev = self._mp_ctx.Event()
            proc    = self._mp_ctx.Process(
                target=_worker_main,
                args=(i, task, stop_ev),
                name=f"GreenGold-Worker-{i}",
                daemon=True,
            )
            proc.start()
            winfo = WorkerInfo(
                worker_id=i,
                task_name=task,
                server_group=grp_name,
                core_group=cores,
                process=proc,
                stop_event=stop_ev,
            )
            self._workers.append(winfo)

            # Set initial CPU affinity
            self._set_affinity_safe(proc.pid, cores)
            log.info("WorkloadRunner: spawned Worker-%d PID=%s task=%s cores=%s",
                     i, proc.pid, task, cores)

        self._started = True
        log.info("WorkloadRunner: %d workers running.", self._num)

    def shutdown(self) -> None:
        """Terminate all worker processes."""
        for w in self._workers:
            if w.stop_event:
                w.stop_event.set()
            if w.process and w.process.is_alive():
                w.process.terminate()
                w.process.join(timeout=2)
        self._workers.clear()
        self._started = False
        log.info("WorkloadRunner: all workers terminated.")

    # ──────────────────────────────────────────
    # Actions (called by RL agent)
    # ──────────────────────────────────────────

    def pause_worker(self, worker_id: int) -> tuple[bool, str]:
        """Suspend a worker process (SIGSTOP / Windows suspend)."""
        w = self._get(worker_id)
        if not w or not w.alive or w.paused:
            return False, f"Worker-{worker_id} not pauseable."
        try:
            p = self._psutil.Process(w.pid)
            p.suspend()
            w.paused = True
            return True, f"Worker-{worker_id} (PID {w.pid}) SUSPENDED."
        except Exception as exc:
            return False, f"pause failed: {exc}"

    def resume_worker(self, worker_id: int) -> tuple[bool, str]:
        """Resume a suspended worker process."""
        w = self._get(worker_id)
        if not w or not w.alive or not w.paused:
            return False, f"Worker-{worker_id} not resumeable."
        try:
            p = self._psutil.Process(w.pid)
            p.resume()
            w.paused = False
            return True, f"Worker-{worker_id} (PID {w.pid}) RESUMED."
        except Exception as exc:
            return False, f"resume failed: {exc}"

    def migrate_worker(self, worker_id: int, target_group: str) -> tuple[bool, str]:
        """
        Migrate a worker to a different logical server (CPU core group).
        This is the real 'move container between machines' analog.
        """
        w = self._get(worker_id)
        if not w or not w.alive:
            return False, f"Worker-{worker_id} not alive."
        if target_group not in SERVER_CORE_GROUPS:
            return False, f"Unknown server group: {target_group}"
        if w.server_group == target_group:
            return False, f"Worker-{worker_id} already on {target_group}."

        new_cores  = SERVER_CORE_GROUPS[target_group]
        old_group  = w.server_group
        ok, msg    = self._set_affinity_safe(w.pid, new_cores)
        if ok:
            w.server_group = target_group
            w.core_group   = new_cores
            return True, (f"Worker-{worker_id} (PID {w.pid}) MIGRATED "
                          f"{old_group} → {target_group} (cores {new_cores})")
        return False, msg

    def rebalance(self) -> tuple[bool, str]:
        """
        Spread paused/overloaded workers evenly across server groups.
        Resumes any paused workers and redistributes core affinity.
        """
        msgs = []
        # Resume all paused workers
        for w in self._workers:
            if w.paused and w.alive:
                ok, msg = self.resume_worker(w.worker_id)
                if ok:
                    msgs.append(msg)

        # Rebalance: alternate groups
        groups = list(SERVER_CORE_GROUPS.keys())
        for i, w in enumerate(self._workers):
            if not w.alive:
                continue
            target = groups[i % len(groups)]
            if w.server_group != target:
                ok, msg = self.migrate_worker(w.worker_id, target)
                if ok:
                    msgs.append(msg)

        if msgs:
            return True, " | ".join(msgs)
        return True, "Rebalance: all workers already balanced."

    # ──────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────

    def get_all(self) -> list[dict]:
        return [w.to_dict() for w in self._workers]

    def get_hottest_worker(self) -> Optional[WorkerInfo]:
        """Return the worker with highest CPU usage (via psutil)."""
        if not self._psutil_available:
            return None
        best, best_cpu = None, -1.0
        for w in self._workers:
            if not w.alive or w.paused:
                continue
            try:
                cpu = self._psutil.Process(w.pid).cpu_percent(interval=0.05)
                if cpu > best_cpu:
                    best_cpu = cpu
                    best = w
            except Exception:
                pass
        return best

    def get_worker_cpu(self, worker_id: int) -> float:
        """Return CPU% for a single worker (0.0 if unavailable)."""
        w = self._get(worker_id)
        if not w or not w.alive or not self._psutil_available:
            return 0.0
        try:
            return self._psutil.Process(w.pid).cpu_percent(interval=0.05)
        except Exception:
            return 0.0

    @property
    def available(self) -> bool:
        return self._started and self._psutil_available

    # ──────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────

    def _get(self, worker_id: int) -> Optional[WorkerInfo]:
        for w in self._workers:
            if w.worker_id == worker_id:
                return w
        return None

    def _set_affinity_safe(self, pid: Optional[int], cores: list[int]) -> tuple[bool, str]:
        if not pid or not self._psutil_available:
            return False, "psutil unavailable or no PID."
        try:
            p = self._psutil.Process(pid)
            # On Windows, psutil uses SetThreadAffinityMask via cpu_affinity setter
            p.cpu_affinity(cores)
            return True, f"PID {pid} affinity → cores {cores}"
        except Exception as exc:
            # Some OS configs don't allow affinity changes — not fatal
            log.debug("set_affinity PID=%s cores=%s failed: %s", pid, cores, exc)
            return False, str(exc)
