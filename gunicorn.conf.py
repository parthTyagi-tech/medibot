import os
import subprocess
import sys

# Keep track of the worker process to terminate it cleanly on exit
_worker_process = None

def on_starting(server):
    """
    Run exactly once by the master Gunicorn process before workers are spawned.
    This starts the LiveKit agent worker in the background.
    """
    global _worker_process
    print("Gunicorn on_starting: Launching LiveKit agent worker in background...")
    try:
        # Launch voice_worker.py in production 'start' mode
        _worker_process = subprocess.Popen(
            [sys.executable, "voice_worker.py", "start"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print(f"Gunicorn on_starting: LiveKit agent worker launched successfully (PID={_worker_process.pid}).")
    except Exception as e:
        print(f"Gunicorn on_starting: Failed to launch voice worker: {e}", file=sys.stderr)

def on_exit(server):
    """
    Run when the master Gunicorn process exits.
    This ensures we cleanly terminate the background worker process and do not leak orphans.
    """
    global _worker_process
    if _worker_process:
        print("Gunicorn on_exit: Terminating LiveKit agent worker...")
        try:
            _worker_process.terminate()
            _worker_process.wait(timeout=5)
            print("Gunicorn on_exit: LiveKit agent worker terminated cleanly.")
        except Exception as e:
            print(f"Gunicorn on_exit: Failed to terminate worker: {e}. Killing process...", file=sys.stderr)
            try:
                _worker_process.kill()
            except Exception:
                pass
