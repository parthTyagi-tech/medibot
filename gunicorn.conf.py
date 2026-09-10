import os
import sys
import time
import subprocess
import threading

# Gunicorn performance & memory optimization for 512MB RAM containers
port = os.getenv("PORT", "10000")
bind = f"0.0.0.0:{port}"
workers = 1
threads = 2
worker_class = "gthread"
timeout = 120
graceful_timeout = 30
keepalive = 5
max_requests = 100
max_requests_jitter = 25


_worker_process = None
_supervisor_thread = None
_keep_running = True


def _supervise_worker():
    global _worker_process, _keep_running

    # Check if voice worker is enabled or if credentials are configured
    enable_voice = os.getenv("ENABLE_VOICE_WORKER", "true").lower() in ("true", "1", "yes")
    livekit_url = os.getenv("LIVEKIT_URL", "").strip()
    livekit_key = os.getenv("LIVEKIT_API_KEY", "").strip()

    if not enable_voice or not livekit_url or not livekit_key:
        print(
            "[gunicorn.conf.py] Voice worker is disabled (ENABLE_VOICE_WORKER=false or missing LIVEKIT credentials). "
            "Skipping voice worker subprocess to conserve RAM.",
            file=sys.stderr,
        )
        return

    print("[gunicorn.conf.py] Starting LiveKit agent worker supervisor thread...")
    consecutive_failures = 0

    while _keep_running:
        if _worker_process is None or _worker_process.poll() is not None:
            if _worker_process is not None and _worker_process.poll() is not None:
                exit_code = _worker_process.poll()
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print(
                        f"[gunicorn.conf.py] Voice worker crashed 5 consecutive times (last exit {exit_code}). "
                        "Halting supervisor to prevent container memory exhaustion.",
                        file=sys.stderr,
                    )
                    break

                backoff = min(60, 5 * (2 ** (consecutive_failures - 1)))
                print(
                    f"[gunicorn.conf.py] Voice worker process (PID {_worker_process.pid}) "
                    f"exited with code {exit_code}. Retrying in {backoff} seconds...",
                    file=sys.stderr,
                )
                time.sleep(backoff)

            if not _keep_running:
                break

            try:
                print("[gunicorn.conf.py] Spawning voice_worker.py start...")
                _worker_process = subprocess.Popen(
                    [sys.executable, "voice_worker.py", "start"],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
                print(
                    f"[gunicorn.conf.py] LiveKit agent worker launched successfully (PID={_worker_process.pid})."
                )
            except Exception as e:
                print(
                    f"[gunicorn.conf.py] Failed to launch voice worker: {e}",
                    file=sys.stderr,
                )
                time.sleep(10)
        else:
            consecutive_failures = 0
        time.sleep(3)


_task_worker_process = None
_task_supervisor_thread = None


def _supervise_task_worker():
    global _task_worker_process, _keep_running

    enable_task_worker = os.getenv("ENABLE_TASK_WORKER", "true").lower() in ("true", "1", "yes")
    if not enable_task_worker:
        print("[gunicorn.conf.py] Task worker disabled (ENABLE_TASK_WORKER=false).", file=sys.stderr)
        return

    print("[gunicorn.conf.py] Starting Redis task worker supervisor thread...")
    consecutive_failures = 0

    while _keep_running:
        if _task_worker_process is None or _task_worker_process.poll() is not None:
            if _task_worker_process is not None and _task_worker_process.poll() is not None:
                exit_code = _task_worker_process.poll()
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print(
                        f"[gunicorn.conf.py] Task worker crashed 5 times (exit {exit_code}). Halting supervisor.",
                        file=sys.stderr,
                    )
                    break
                backoff = min(60, 5 * (2 ** (consecutive_failures - 1)))
                print(
                    f"[gunicorn.conf.py] Task worker (PID {_task_worker_process.pid}) exited with code {exit_code}. Retrying in {backoff}s...",
                    file=sys.stderr,
                )
                time.sleep(backoff)

            if not _keep_running:
                break

            try:
                print("[gunicorn.conf.py] Spawning task_worker.py...")
                _task_worker_process = subprocess.Popen(
                    [sys.executable, "task_worker.py"],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
                print(f"[gunicorn.conf.py] Task worker launched successfully (PID={_task_worker_process.pid}).")
            except Exception as e:
                print(f"[gunicorn.conf.py] Failed to launch task worker: {e}", file=sys.stderr)
                time.sleep(10)
        else:
            consecutive_failures = 0
        time.sleep(3)


def on_starting(server):
    """
    Run exactly once by the master Gunicorn process before workers are spawned.
    This starts and supervises both LiveKit agent worker and Redis task worker.
    """
    global _supervisor_thread, _task_supervisor_thread, _keep_running
    _keep_running = True
    _supervisor_thread = threading.Thread(target=_supervise_worker, daemon=True)
    _supervisor_thread.start()

    _task_supervisor_thread = threading.Thread(target=_supervise_task_worker, daemon=True)
    _task_supervisor_thread.start()


def on_exit(server):
    """
    Run when the master Gunicorn process exits.
    This ensures we cleanly terminate background worker processes and do not leak orphans.
    """
    global _worker_process, _task_worker_process, _keep_running
    _keep_running = False

    for proc_name, proc in [("LiveKit voice worker", _worker_process), ("Redis task worker", _task_worker_process)]:
        if proc:
            print(f"Gunicorn on_exit: Terminating {proc_name}...")
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"Gunicorn on_exit: {proc_name} terminated cleanly.")
            except Exception as e:
                print(f"Gunicorn on_exit: Failed to terminate {proc_name}: {e}. Killing...", file=sys.stderr)
                try:
                    proc.kill()
                except Exception:
                    pass

