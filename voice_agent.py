"""Backward-compatible alias — run voice_worker.py instead."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("voice_worker", run_name="__main__")
