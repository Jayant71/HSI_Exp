import os
import sys
import re
from datetime import datetime


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


class _TeeStream:
    """Write to both the original stream and a file simultaneously."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, msg):
        self._original.write(msg)
        self._log_file.write(msg)
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()


def _next_run_number(dataset_dir):
    """Scan *dataset_dir* for run<N> folders and return N+1."""
    if not os.path.isdir(dataset_dir):
        return 1
    existing = []
    for name in os.listdir(dataset_dir):
        m = re.match(r"^run(\d+)$", name)
        if m:
            existing.append(int(m.group(1)))
    return max(existing, default=0) + 1


class RunLogger:
    """Manages a single experiment run directory and logging.

    Usage
    -----
    >>> logger = RunLogger(dataset="ip", model="svm")
    >>> logger.log("Training started")
    >>> logger.savefig(fig, "confusion_matrix.png")
    >>> logger.close()

    Or as a context manager:
    >>> with RunLogger(dataset="ip", model="svm") as logger:
    ...     logger.log("hello")
    """

    def __init__(self, dataset, model="experiment"):
        self.dataset = dataset
        self.model = model

        dataset_dir = os.path.join(RESULTS_ROOT, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        self.run_number = _next_run_number(dataset_dir)
        self.run_dir = os.path.join(dataset_dir, f"run{self.run_number}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Set up file logging
        log_path = os.path.join(self.run_dir, "results.log")
        self._log_file = open(log_path, "w")

        # Tee stdout/stderr so all print() calls are captured
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self._log_file)
        sys.stderr = _TeeStream(self._orig_stderr, self._log_file)

        self.log(f"=== Run {self.run_number} | dataset={dataset} | model={model} ===")
        self.log(f"Run directory: {self.run_dir}")
        self.log(f"Started at {datetime.now().isoformat()}")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def log(self, msg, level="info"):
        """Write a structured log line to console and log file (via tee)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} [{level.upper()}] {msg}")

    def get_path(self, filename):
        """Return the full path for *filename* inside the run directory."""
        return os.path.join(self.run_dir, filename)

    def savefig(self, fig, filename, dpi=150):
        """Save a matplotlib figure to the run directory."""
        path = self.get_path(filename)
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        self.log(f"Saved figure: {filename}")

    def save_array(self, arr, filename):
        """Save a numpy array (.npy) to the run directory."""
        import numpy as np
        path = self.get_path(filename)
        np.save(path, arr)
        self.log(f"Saved array: {filename}")

    def save_text(self, text, filename):
        """Save arbitrary text to a file in the run directory."""
        path = self.get_path(filename)
        with open(path, "w") as f:
            f.write(text)
        self.log(f"Saved text: {filename}")

    def close(self):
        """Restore stdout/stderr and close the log file."""
        self.log(f"Finished at {datetime.now().isoformat()}")
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._log_file.close()

    # ----------------------------------------------------------
    # Context manager
    # ----------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log(f"Run aborted with {exc_type.__name__}: {exc_val}", level="error")
        self.close()
        return False
