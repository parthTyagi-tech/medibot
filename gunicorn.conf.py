import os
import sys
import time
import subprocess
import threading

# Gunicorn performance & memory optimization for 512MB RAM containers
workers = 1
threads = 4
worker_class = "gthread"
timeout = 120
graceful_timeout = 30
keepalive = 5
max_requests = 500
max_requests_jitter = 50


_worker_process = None
_supervisor_thread = None
_keep_running = True


def _supervise_worker():
    global _worker_process, _keep_running
    print("[gunicorn.conf.py] Starting LiveKit agent worker supervisor thread...")
    while _keep_running:
        if _worker_process is None or _worker_process.poll() is not None:
            if _worker_process is not None and _worker_process.poll() is not None:
                exit_code = _worker_process.poll()
                print(
                    f"[gunicorn.conf.py] Voice worker process (PID {_worker_process.pid}) "
                    f"exited with code {exit_code}. Restarting in 3 seconds...",
                    file=sys.stderr,
                )
                time.sleep(3)

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
                time.sleep(5)
        time.sleep(2)


def on_starting(server):
    """
    Run exactly once by the master Gunicorn process before workers are spawned.
    This starts and supervises the LiveKit agent worker in the background.
    """
    global _supervisor_thread, _keep_running
    _keep_running = True
    _supervisor_thread = threading.Thread(target=_supervise_worker, daemon=True)
    _supervisor_thread.start()


def on_exit(server):
    """
    Run when the master Gunicorn process exits.
    This ensures we cleanly terminate the background worker process and do not leak orphans.
    """
    global _worker_process, _keep_running
    _keep_running = False
    if _worker_process:
        print("Gunicorn on_exit: Terminating LiveKit agent worker...")
        try:
            _worker_process.terminate()
            _worker_process.wait(timeout=5)
            print("Gunicorn on_exit: LiveKit agent worker terminated cleanly.")
        except Exception as e:
            print(
                f"Gunicorn on_exit: Failed to terminate worker: {e}. Killing process...",
                file=sys.stderr,
            )
            try:
                _worker_process.kill()
            except Exception:
                pass

