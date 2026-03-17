import azure.functions as func
import datetime
import logging
import os
import subprocess
import sys
from pathlib import Path

app = func.FunctionApp()


@app.function_name(name="run_pipeline")
@app.schedule(schedule="0 0 6 * * 0", arg_name="mytimer", run_on_startup=False, use_monitor=False)
def run_pipeline(mytimer: func.TimerRequest) -> None:
    utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
    if mytimer.past_due:
        logging.warning("Timer is past due at %s", utc_now)

    repo_root = Path(__file__).resolve().parents[1]
    run_pipeline_py = repo_root / "run_pipeline.py"

    if not run_pipeline_py.exists():
        raise FileNotFoundError(f"Could not find run_pipeline.py at: {run_pipeline_py}")

    logging.info("Starting UFC pipeline at %s", utc_now)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [sys.executable, str(run_pipeline_py), "--quick", "--storage", "azure"]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    if proc.stdout:
        logging.info("Pipeline stdout:\n%s", proc.stdout)
    if proc.stderr:
        logging.warning("Pipeline stderr:\n%s", proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"Pipeline failed with exit code {proc.returncode}")

    logging.info("Pipeline completed successfully at %s", utc_now)