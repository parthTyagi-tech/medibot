import os
import subprocess
import sys

def on_starting(server):
    """
    Run exactly once by the master Gunicorn process before workers are spawned.
    This starts the LiveKit agent worker in the background.
    """
    print("Gunicorn on_starting: Launching LiveKit agent worker in background...")
    try:
        # Launch voice_worker.py in production 'start' mode
        subprocess.Popen(
            [sys.executable, "voice_worker.py", "start"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print("Gunicorn on_starting: LiveKit agent worker launched successfully.")
    except Exception as e:
        print(f"Gunicorn on_starting: Failed to launch voice worker: {e}", file=sys.stderr)
