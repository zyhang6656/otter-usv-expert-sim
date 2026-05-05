from __future__ import annotations

import argparse
import os
import json
import math
import mimetypes
import sys
import time
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from .config import ControllerConfig, SimConfig
from .io import load_sample_json, save_dataset_npz
from .planner import theta_star_path
from .scenario import obstacle_clearance, sample_scenario
from .simulator import simulate_scenario
from .types import Scenario, TrajectorySample


DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_LABELS = {"easy": "简单", "medium": "中等", "hard": "困难"}
MIN_START_GOAL_DISTANCE = 80.0
MIN_TRAJECTORY_CLEARANCE = 0.2
MAX_SINGLE_GENERATE_COUNT = 50
MAX_BALANCED_GENERATE_COUNT = 30000
BATCH_SAVE_INTERVAL = 250
MAX_BALANCED_WORKERS = 6
BATCH_CONTROLLER_CONFIG = ControllerConfig(candidates=160, iterations=2, horizon=8, elites=24)
STATIC_DIR = Path(__file__).with_name("dashboard_static")


@dataclass
class SampleRecord:
    sample_id: str
    source: str
    index: int
    sample: TrajectorySample
    difficulty: str


@dataclass
class GenerationJob:
    job_id: str
    requested: int
    targets: dict[str, int]
    seed: int
    output: str
    created: dict[str, int]
    attempts: dict[str, int]
    failures: dict[str, dict[str, int]]
    status: str = "running"
    message: str = ""
    started_at: str = ""
    finished_at: str = ""
    saved_at: str = ""
    workers: int = 1
    total_created: int = 0
    saved_count: int = 0
    generated_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "message": self.message,
            "requested": self.requested,
            "created": self.total_created,
            "targets": dict(self.targets),
            "created_by_difficulty": dict(self.created),
            "attempts": dict(self.attempts),
            "failures": {key: dict(value) for key, value in self.failures.items()},
            "seed": self.seed,
            "output": self.output,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "saved_at": self.saved_at,
            "workers": self.workers,
            "saved_count": self.saved_count,
            "first_generated_id": self.generated_ids[0] if self.generated_ids else "",
            "generated_ids": self.generated_ids,
        }


class DashboardState:
    def __init__(self, data_dir: Path, runs_dir: Path, generated_path: Path):
        self.data_dir = data_dir
        self.runs_dir = runs_dir
        self.generated_path = generated_path
        self._lock = Lock()
        self._records: dict[str, SampleRecord] = {}
        self._jobs: dict[str, GenerationJob] = {}
        self.reload()

    def reload(self) -> None:
        records: dict[str, SampleRecord] = {}
        for path in sorted(self.data_dir.glob("*.npz")):
            for record in _load_npz_records(path):
                records[record.sample_id] = record
        for path in sorted(self.runs_dir.glob("*.json")):
            record = _load_json_record(path)
            if record is not None:
                records[record.sample_id] = record
        with self._lock:
            self._records = records

    def summaries(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records.values())
        records = [record for record in records if _is_displayable_expert(record.sample)]
        summaries = [_summary_payload(record) for record in records]
        counts = {difficulty: 0 for difficulty in DIFFICULTIES}
        for item in summaries:
            counts[item["difficulty"]] = counts.get(item["difficulty"], 0) + 1
        return {"samples": summaries, "counts": counts, "total": len(summaries)}

    def sample_detail(self, sample_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(sample_id)
        if record is None or not _is_displayable_expert(record.sample):
            return None
        return _detail_payload(record)

    def generate(self, difficulty: str, count: int, seed: int | None) -> dict[str, Any]:
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty: {difficulty}")
        if count < 1:
            raise ValueError("count must be positive")
        if count > MAX_SINGLE_GENERATE_COUNT:
            raise ValueError(f"count must be <= {MAX_SINGLE_GENERATE_COUNT}")

        base_seed = int(seed if seed is not None else time.time_ns() % (2**31 - 1))
        rng = np.random.default_rng(base_seed)
        config = SimConfig(rng_seed=base_seed)
        samples: list[TrajectorySample] = []
        failures: dict[str, int] = {"sample": 0, "not_reached": 0, "collided": 0, "saturated": 0}
        max_attempts = max(60, count * 90)
        attempt = 0

        while len(samples) < count and attempt < max_attempts:
            attempt += 1
            sim_seed = base_seed + attempt
            try:
                scenario = sample_scenario(
                    rng,
                    difficulty=difficulty,
                    config=config,
                    min_distance=MIN_START_GOAL_DISTANCE,
                )
                sample = simulate_scenario(scenario, config, seed=sim_seed)
            except Exception:
                failures["sample"] += 1
                continue

            accepted, failure = _prepare_generated_sample(sample, difficulty, sim_seed, config, keep_trace=True)
            if accepted:
                samples.append(sample)
            else:
                failures[failure] = failures.get(failure, 0) + 1

        if not samples:
            self.reload()
            return {
                "requested": count,
                "created": 0,
                "difficulty": difficulty,
                "difficulty_label": DIFFICULTY_LABELS[difficulty],
                "seed": base_seed,
                "attempts": attempt,
                "failures": failures,
                "generated_ids": [],
                "output": str(self.generated_path),
            }

        existing = _load_npz_samples(self.generated_path) if self.generated_path.exists() else []
        existing = [
            sample
            for sample in existing
            if sample.states.shape == samples[0].states.shape and _is_displayable_expert(sample)
        ]
        first_new_index = len(existing)
        all_samples = [*existing, *samples]
        save_dataset_npz(
            all_samples,
            self.generated_path,
            {
                "generated_by": "usv_sim.dashboard",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "latest_generation": {
                    "requested": count,
                    "created": len(samples),
                    "difficulty": difficulty,
                    "seed": base_seed,
                    "attempts": attempt,
                    "failures": failures,
                },
            },
        )
        self.reload()
        generated_ids = [
            _make_npz_id(self.generated_path, first_new_index + i)
            for i in range(len(samples))
        ]
        return {
            "requested": count,
            "created": len(samples),
            "difficulty": difficulty,
            "difficulty_label": DIFFICULTY_LABELS[difficulty],
            "seed": base_seed,
            "attempts": attempt,
            "failures": failures,
            "generated_ids": generated_ids,
            "output": str(self.generated_path),
        }

    def start_balanced_generation(self, count: int, seed: int | None) -> dict[str, Any]:
        if count < 3:
            raise ValueError("count must be >= 3")
        if count > MAX_BALANCED_GENERATE_COUNT:
            raise ValueError(f"count must be <= {MAX_BALANCED_GENERATE_COUNT}")

        with self._lock:
            running = next((job for job in self._jobs.values() if job.status == "running"), None)
            if running is not None:
                return running.to_json()

        base_seed = int(seed if seed is not None else time.time_ns() % (2**31 - 1))
        targets = _balanced_targets(count)
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = self.data_dir / f"balanced_{count}_{base_seed}_{run_stamp}.npz"
        job_id = f"balanced-{base_seed}-{run_stamp}"
        job = GenerationJob(
            job_id=job_id,
            requested=count,
            targets=targets,
            seed=base_seed,
            output=str(output),
            workers=_balanced_worker_count(count),
            created={difficulty: 0 for difficulty in DIFFICULTIES},
            attempts={difficulty: 0 for difficulty in DIFFICULTIES},
            failures={
                difficulty: {"sample": 0, "not_reached": 0, "collided": 0, "saturated": 0}
                for difficulty in DIFFICULTIES
            },
            started_at=datetime.now().isoformat(timespec="seconds"),
            message="批量生成中",
        )
        with self._lock:
            self._jobs[job_id] = job
        worker = Thread(target=self._run_balanced_generation, args=(job, output), daemon=True)
        worker.start()
        return job.to_json()

    def job_status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else job.to_json()

    def latest_job_status(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._jobs:
                return None
            job = next((item for item in self._jobs.values() if item.status == "running"), None)
            if job is None:
                job = list(self._jobs.values())[-1]
            return job.to_json()

    def _run_balanced_generation(self, job: GenerationJob, output: Path) -> None:
        max_attempts = {difficulty: max(300, target * 120) for difficulty, target in job.targets.items()}
        samples: list[TrajectorySample] = []
        next_attempt = 0

        multiprocess_failed = False
        multiprocess_error = ""
        try:
            next_attempt = self._run_balanced_with_executor(
                ProcessPoolExecutor(max_workers=job.workers),
                job,
                output,
                max_attempts,
                samples,
                next_attempt,
            )
        except Exception as exc:
            multiprocess_failed = True
            multiprocess_error = str(exc)

        if multiprocess_failed and not samples:
            with self._lock:
                job.message = f"多进程不可用，已切换到线程生成：{multiprocess_error}"
                self._jobs[job.job_id] = job
            try:
                next_attempt = self._run_balanced_with_executor(
                    ThreadPoolExecutor(max_workers=job.workers),
                    job,
                    output,
                    max_attempts,
                    samples,
                    next_attempt,
                )
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.message = str(exc)
        elif multiprocess_failed:
            with self._lock:
                job.status = "failed"
                job.message = multiprocess_error

        try:
            if job.status != "failed":
                self._save_balanced_samples(samples, output, job, status="completed")
                with self._lock:
                    job.generated_ids = [_make_npz_id(output, i) for i in range(len(samples))]
                    job.status = "completed"
                    job.message = f"已生成 {job.total_created}/{job.requested}"
        finally:
            with self._lock:
                job.finished_at = datetime.now().isoformat(timespec="seconds")
            self.reload()
            self._publish_job(job)

    def _run_balanced_with_executor(
        self,
        executor: Executor,
        job: GenerationJob,
        output: Path,
        max_attempts: dict[str, int],
        samples: list[TrajectorySample],
        next_attempt: int,
    ) -> int:
        with executor as pool:
            futures: dict[Any, str] = {}
            while job.total_created < job.requested or futures:
                while len(futures) < job.workers:
                    difficulty = _next_needed_difficulty(job)
                    if difficulty is None:
                        break
                    if job.attempts[difficulty] >= max_attempts[difficulty]:
                        raise RuntimeError(
                            f"{DIFFICULTY_LABELS[difficulty]}轨迹只生成了 "
                            f"{job.created[difficulty]}/{job.targets[difficulty]}，已达到最大尝试次数"
                        )
                    next_attempt += 1
                    attempt_seed = job.seed + next_attempt
                    with self._lock:
                        job.attempts[difficulty] += 1
                        job.message = _balanced_job_message(job)
                        self._jobs[job.job_id] = job
                    future = pool.submit(_generate_balanced_attempt, difficulty, attempt_seed)
                    futures[future] = difficulty

                if not futures:
                    break

                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    difficulty = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        result = {"accepted": False, "failure": "sample", "sample": None}
                    accepted = bool(result.get("accepted", False))
                    failure = str(result.get("failure") or "sample")
                    sample = result.get("sample")

                    with self._lock:
                        if accepted and isinstance(sample, TrajectorySample):
                            if job.created[difficulty] < job.targets[difficulty]:
                                samples.append(sample)
                                job.created[difficulty] += 1
                                job.total_created += 1
                            else:
                                failure = "overflow"
                                job.failures[difficulty][failure] = job.failures[difficulty].get(failure, 0) + 1
                        else:
                            job.failures[difficulty][failure] = job.failures[difficulty].get(failure, 0) + 1
                        job.message = _balanced_job_message(job)
                        self._jobs[job.job_id] = job

                    save_interval = min(BATCH_SAVE_INTERVAL, max(1, job.requested // 20))
                    if samples and (
                        len(samples) - job.saved_count >= save_interval
                        or job.total_created >= job.requested
                    ):
                        self._save_balanced_samples(samples, output, job, status="running")

        return next_attempt

    def _publish_job(self, job: GenerationJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def _save_balanced_samples(
        self,
        samples: list[TrajectorySample],
        output: Path,
        job: GenerationJob,
        status: str,
    ) -> None:
        save_dataset_npz(
            samples,
            output,
            {
                "generated_by": "usv_sim.dashboard.balanced",
                "status": status,
                "requested": job.requested,
                "created": job.total_created,
                "difficulty_ratio": "easy:medium:hard=1:1:1",
                "targets": job.targets,
                "created_by_difficulty": job.created,
                "attempts": job.attempts,
                "seed": job.seed,
                "workers": job.workers,
                "failures": job.failures,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        with self._lock:
            job.saved_count = len(samples)
            job.saved_at = datetime.now().isoformat(timespec="seconds")
            self._jobs[job.job_id] = job


class DashboardHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: DashboardState):
        super().__init__(server_address, DashboardRequestHandler)
        self.state = state


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        _safe_print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/samples":
            self._send_json(self.server.state.summaries())
            return
        if path == "/api/generate/balanced/latest":
            status = self.server.state.latest_job_status()
            if status is None:
                self._send_json({"status": "idle"})
            else:
                self._send_json(status)
            return
        if path.startswith("/api/generate/balanced/"):
            job_id = unquote(path.removeprefix("/api/generate/balanced/"))
            status = self.server.state.job_status(job_id)
            if status is None:
                self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
            else:
                self._send_json(status)
            return
        if path.startswith("/api/samples/"):
            sample_id = unquote(path.removeprefix("/api/samples/"))
            detail = self.server.state.sample_detail(sample_id)
            if detail is None:
                self._send_json({"error": "sample not found"}, HTTPStatus.NOT_FOUND)
            else:
                self._send_json(detail)
            return
        if path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        candidate = (STATIC_DIR / path.lstrip("/")).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_json({"error": "invalid static path"}, HTTPStatus.BAD_REQUEST)
            return
        if candidate.is_file():
            mime = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            self._send_file(candidate, mime)
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate/balanced":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                count = int(payload.get("count", 9000))
                raw_seed = payload.get("seed", None)
                seed = None if raw_seed in (None, "") else int(raw_seed)
                result = self.server.state.start_balanced_generation(count, seed)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(result, HTTPStatus.ACCEPTED)
            return
        if parsed.path != "/api/generate":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            difficulty = str(payload.get("difficulty", "medium"))
            count = int(payload.get("count", 1))
            raw_seed = payload.get("seed", None)
            seed = None if raw_seed in (None, "") else int(raw_seed)
            result = self.server.state.generate(difficulty, count, seed)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(result)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _load_npz_records(path: Path) -> list[SampleRecord]:
    samples = _load_npz_samples(path)
    return [
        SampleRecord(
            sample_id=_make_npz_id(path, i),
            source=str(path),
            index=i,
            sample=sample,
            difficulty=_infer_difficulty(sample.metadata, sample.scenario),
        )
        for i, sample in enumerate(samples)
    ]


def _load_npz_samples(path: Path) -> list[TrajectorySample]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as npz:
        states = np.asarray(npz["states"], dtype=float)
        controls = np.asarray(npz["controls"], dtype=float)
        env = np.asarray(npz["env"], dtype=float)
        metadata = _read_metadata(npz)

    samples_meta = metadata.get("samples", [])
    scenarios = metadata.get("scenarios", [])
    paths = metadata.get("paths", [])
    out: list[TrajectorySample] = []
    for i in range(states.shape[0]):
        if i >= len(scenarios):
            continue
        scenario = Scenario.from_json(scenarios[i])
        sample_meta = dict(samples_meta[i]) if i < len(samples_meta) else {}
        if i < len(paths) and len(paths[i]) >= 2:
            path_arr = np.asarray(paths[i], dtype=float)
        else:
            path_arr = _path_for_scenario(scenario)
        out.append(
            TrajectorySample(
                states=np.asarray(states[i], dtype=float),
                controls=np.asarray(controls[i], dtype=float),
                env57=np.asarray(env[i], dtype=float),
                path=path_arr,
                metadata=sample_meta,
                scenario=scenario,
            )
        )
    return out


def _read_metadata(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in npz.files:
        return {}
    raw = npz["metadata"]
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw)) if raw is not None and str(raw) else {}


def _load_json_record(path: Path) -> SampleRecord | None:
    try:
        sample = load_sample_json(path)
    except Exception:
        return None
    return SampleRecord(
        sample_id=f"json:{path.parent.name}:{path.name}",
        source=str(path),
        index=0,
        sample=sample,
        difficulty=_infer_difficulty(sample.metadata, sample.scenario),
    )


def _make_npz_id(path: Path, index: int) -> str:
    return f"npz:{path.parent.name}:{path.name}:{index}"


def _path_for_scenario(scenario: Scenario) -> np.ndarray:
    try:
        return theta_star_path(scenario, SimConfig())
    except Exception:
        return np.vstack([scenario.start[:2], scenario.goal])


def _infer_difficulty(metadata: dict[str, Any], scenario: Scenario) -> str:
    difficulty = str(metadata.get("difficulty", "")).lower()
    if difficulty in DIFFICULTIES:
        return difficulty

    name = scenario.name.lower()
    for candidate in DIFFICULTIES:
        if candidate in name:
            return candidate

    static_count = int(metadata.get("static_obstacle_count", len(scenario.static_obstacles)))
    dynamic_count = int(metadata.get("dynamic_obstacle_count", len(scenario.dynamic_obstacles)))
    initial_distance = float(metadata.get("initial_distance", np.linalg.norm(scenario.goal - scenario.start[:2])))
    if static_count <= 4 and dynamic_count <= 2 and initial_distance <= 50.0:
        return "easy"
    if static_count >= 8 or dynamic_count >= 4 or initial_distance >= 60.0:
        return "hard"
    return "medium"


def _start_goal_distance(sample: TrajectorySample) -> float:
    return float(np.linalg.norm(sample.scenario.goal - sample.scenario.start[:2]))


def _metadata_min_clearance(sample: TrajectorySample) -> float | None:
    value = sample.metadata.get("min_clearance")
    try:
        clearance = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(clearance) else clearance


def _trajectory_min_clearance(
    sample: TrajectorySample,
    config: SimConfig | None = None,
    *,
    use_cached: bool = False,
) -> float:
    if use_cached:
        cached = _metadata_min_clearance(sample)
        if cached is not None:
            return cached

    config = config or SimConfig()
    dt = float(sample.metadata.get("dt", config.dt))
    obstacles = sample.scenario.all_obstacles()
    if not obstacles:
        return float("inf")
    min_clearance = float("inf")
    for step, state in enumerate(np.asarray(sample.states, dtype=float)):
        t = step * dt
        clearance = obstacle_clearance(state[:2], obstacles, t, config.obstacle_clearance_margin)
        min_clearance = min(min_clearance, clearance)
    return float(min_clearance)


def _prepare_generated_sample(
    sample: TrajectorySample,
    difficulty: str,
    seed: int,
    config: SimConfig,
    keep_trace: bool,
) -> tuple[bool, str]:
    sample.metadata["difficulty"] = difficulty
    sample.metadata["generated_at"] = datetime.now().isoformat(timespec="seconds")
    sample.metadata["generation_seed"] = seed
    strict_clearance = _trajectory_min_clearance(sample, config)
    if (
        sample.metadata.get("success", False)
        and sample.metadata.get("control_saturation_ratio", 1.0) <= 0.9
        and _start_goal_distance(sample) >= MIN_START_GOAL_DISTANCE
        and strict_clearance >= MIN_TRAJECTORY_CLEARANCE
    ):
        sample.metadata["min_clearance"] = strict_clearance
        if not keep_trace:
            sample.metadata.pop("trace", None)
        return True, ""
    if sample.metadata.get("collided", False) or strict_clearance < MIN_TRAJECTORY_CLEARANCE:
        return False, "collided"
    if not sample.metadata.get("reached", False):
        return False, "not_reached"
    return False, "saturated"


def _balanced_targets(count: int) -> dict[str, int]:
    if count % len(DIFFICULTIES) != 0:
        raise ValueError("count must be divisible by 3 for a 1:1:1 difficulty ratio")
    per_difficulty = count // len(DIFFICULTIES)
    return {difficulty: per_difficulty for difficulty in DIFFICULTIES}


def _balanced_worker_count(count: int) -> int:
    cpu_count = os.cpu_count() or 1
    if count < len(DIFFICULTIES) * 4:
        return 1
    return max(1, min(MAX_BALANCED_WORKERS, cpu_count))


def _next_needed_difficulty(job: GenerationJob) -> str | None:
    needed = [
        difficulty
        for difficulty in DIFFICULTIES
        if job.created[difficulty] < job.targets[difficulty]
    ]
    if not needed:
        return None
    return min(
        needed,
        key=lambda difficulty: (
            job.created[difficulty] / max(1, job.targets[difficulty]),
            job.attempts[difficulty],
        ),
    )


def _generate_balanced_attempt(difficulty: str, seed: int) -> dict[str, Any]:
    config = SimConfig(rng_seed=seed)
    rng = np.random.default_rng(seed)
    try:
        scenario = sample_scenario(
            rng,
            difficulty=difficulty,
            config=config,
            min_distance=MIN_START_GOAL_DISTANCE,
        )
        sample = simulate_scenario(
            scenario,
            config,
            BATCH_CONTROLLER_CONFIG,
            seed=seed,
            record_trace=False,
            stop_on_reach=True,
        )
    except Exception:
        return {"accepted": False, "failure": "sample", "sample": None}

    accepted, failure = _prepare_generated_sample(sample, difficulty, seed, config, keep_trace=False)
    return {"accepted": accepted, "failure": failure, "sample": sample if accepted else None}


def _balanced_job_message(job: GenerationJob) -> str:
    parts = [
        f"{DIFFICULTY_LABELS[difficulty]} {job.created[difficulty]}/{job.targets[difficulty]}"
        for difficulty in DIFFICULTIES
    ]
    return "，".join(parts)


def _trace_for_sample(sample: TrajectorySample) -> list[dict[str, Any]]:
    trace = sample.metadata.get("trace", [])
    if trace:
        return trace
    config = SimConfig()
    dt = float(sample.metadata.get("dt", config.dt))
    obstacles = sample.scenario.all_obstacles()
    out: list[dict[str, Any]] = []
    goal = np.asarray(sample.scenario.goal, dtype=float)
    path = np.asarray(sample.path, dtype=float)
    subtarget_idx = 0
    for step, state in enumerate(np.asarray(sample.states, dtype=float)):
        pos = state[:2]
        if len(path) > 1:
            while subtarget_idx < len(path) - 1 and np.linalg.norm(pos - path[subtarget_idx]) < config.subtarget_radius:
                subtarget_idx += 1
            subtarget = path[subtarget_idx]
        else:
            subtarget = goal
        t = step * dt
        out.append(
            {
                "step": step,
                "time": t,
                "subtarget_idx": int(subtarget_idx),
                "subtarget": np.asarray(subtarget, dtype=float).tolist(),
                "clearance": obstacle_clearance(pos, obstacles, t, config.obstacle_clearance_margin),
                "dist_goal": float(np.linalg.norm(pos - goal)),
                "control_cost": 0.0,
            }
        )
    return out


def _is_displayable_expert(sample: TrajectorySample) -> bool:
    return (
        bool(sample.metadata.get("success", False))
        and _start_goal_distance(sample) >= MIN_START_GOAL_DISTANCE
        and _trajectory_min_clearance(sample, use_cached=True) >= MIN_TRAJECTORY_CLEARANCE
    )


def _summary_payload(record: SampleRecord) -> dict[str, Any]:
    sample = record.sample
    states_xy = np.asarray(sample.states[:, 0:2], dtype=float)
    step_distances = np.linalg.norm(np.diff(states_xy, axis=0), axis=1) if len(states_xy) > 1 else np.asarray([])
    metadata = sample.metadata
    dt = float(metadata.get("dt", SimConfig().dt))
    min_clearance = _trajectory_min_clearance(sample, use_cached=True)
    return {
        "id": record.sample_id,
        "source": record.source,
        "index": record.index + 1,
        "difficulty": record.difficulty,
        "difficulty_label": DIFFICULTY_LABELS[record.difficulty],
        "name": sample.scenario.name,
        "success": bool(metadata.get("success", False)),
        "reached": bool(metadata.get("reached", False)),
        "collided": bool(metadata.get("collided", False)),
        "static_obstacle_count": int(metadata.get("static_obstacle_count", len(sample.scenario.static_obstacles))),
        "dynamic_obstacle_count": int(metadata.get("dynamic_obstacle_count", len(sample.scenario.dynamic_obstacles))),
        "initial_distance": _start_goal_distance(sample),
        "final_distance": _finite_float(metadata.get("final_distance")),
        "min_clearance": min_clearance,
        "control_saturation_ratio": _finite_float(metadata.get("control_saturation_ratio")),
        "duration": (len(sample.states) - 1) * dt,
        "steps": len(sample.states),
        "trajectory_length": float(np.sum(step_distances)) if len(step_distances) else 0.0,
        "generated_at": metadata.get("generated_at", ""),
        "polyline": _round_array(states_xy),
    }


def _detail_payload(record: SampleRecord) -> dict[str, Any]:
    summary = _summary_payload(record)
    sample = record.sample
    metadata = dict(sample.metadata)
    trace = _trace_for_sample(sample)
    metadata.pop("trace", None)
    return {
        **summary,
        "states": _round_array(sample.states),
        "controls": _round_array(sample.controls),
        "path": _round_array(sample.path),
        "scenario": sample.scenario.to_json(),
        "metadata": metadata,
        "trace": trace,
        "workspace": list(sample.scenario.workspace),
        "dt": float(sample.metadata.get("dt", SimConfig().dt)),
        "safety_margin": SimConfig().safety_margin,
        "vessel_collision_radius": SimConfig().vessel_collision_radius,
        "obstacle_clearance_margin": SimConfig().obstacle_clearance_margin,
    }


def _round_array(values: np.ndarray) -> list[Any]:
    return np.round(np.asarray(values, dtype=float), 4).tolist()


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_print(*args: Any) -> None:
    if sys.stdout is None:
        return
    try:
        print(*args, flush=True)
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local USV expert trajectory dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--generated-out", type=Path, default=Path("data/dashboard_generated.npz"))
    args = parser.parse_args()

    state = DashboardState(args.data_dir, args.runs_dir, args.generated_out)
    server = DashboardHTTPServer((args.host, args.port), state)
    url = f"http://{args.host}:{server.server_port}"
    _safe_print(f"USV dashboard running at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
