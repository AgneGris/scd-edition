"""
generate_decomp_jobs.py
───────────────────────
Generate (and optionally submit) one PBS job per decomposition group
defined in a file-list YAML.

Usage
─────
    python jobs/generate_decomp_jobs.py jobs/file_lists/sub01_day01.yaml
    python jobs/generate_decomp_jobs.py jobs/file_lists/sub02_day01.yaml

Each decomposition group in the YAML becomes one .sh file in jobs/.
To actually submit, uncomment the qsub line at the bottom.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# ── HPC / environment settings (edit as needed) ──────────────────────────────

CONDA_ENV    = "scd"                                        # conda environment
PATH_TO_LOGS = Path(os.environ.get("EPHEMERAL", "/tmp"), "thinfilm_logs_dir")

# ── decomposition parameters ─────────────────────────────────────────────────

SIL_THRESHOLD = 0.85
ITERATIONS    = 150

# ── paths ────────────────────────────────────────────────────────────────────

TEMPLATE_PATH = Path("jobs/batch_decomp_template.sh")
JOB_OUTPUTS   = Path("jobs/job_outputs")

# ─────────────────────────────────────────────────────────────────────────────


def generate_jobs(yaml_path: Path) -> None:
    with open(yaml_path) as f:
        fl = yaml.safe_load(f)

    subject      = fl["subject"]
    data_dir     = Path(fl["data_dir"])
    channel_cfg  = fl["channel_config"]
    rejections   = fl["output_rejections"]
    output_dir   = fl["output_dir"]

    with open(TEMPLATE_PATH) as f:
        template = f.read()

    JOB_OUTPUTS.mkdir(parents=True, exist_ok=True)

    generated = []
    for group in fl["decompositions"]:
        name        = group["name"]
        files       = group["files"]
        concatenate = group["concatenate"]

        run_name = f"{subject.replace('-', '')}_{name}"  # e.g. sub01_5ext_I

        # Full paths
        full_paths = [str(data_dir / fname) for fname in files]

        # bash array of quoted file paths for the copy loop
        files_bash_array = " ".join(f'"{p}"' for p in full_paths)

        # echo lines for logging
        files_echo = "\n".join(f'    echo "  {p}"' for p in full_paths)

        # --concat flag (with line continuation if present)
        concat_flag = "        --concat \\\n" if concatenate else ""
        concat_bool = str(concatenate)

        job_script = template
        job_script = job_script.replace("%RUN_NAME%",        run_name)
        job_script = job_script.replace("%CHANNEL_CONFIG%",  channel_cfg)
        job_script = job_script.replace("%FILES_BASH_ARRAY%",files_bash_array)
        job_script = job_script.replace("%FILES_ECHO%",      files_echo)
        job_script = job_script.replace("%CONCAT_FLAG%",    concat_flag)
        job_script = job_script.replace("%CONCAT_BOOL%",    concat_bool)
        job_script = job_script.replace("%REJECTIONS_FILE%",rejections)
        job_script = job_script.replace("%OUTPUT_DIR%",     output_dir)
        job_script = job_script.replace("%PATH_TO_LOGS%",   str(PATH_TO_LOGS))
        job_script = job_script.replace("%ENVIRONMENT%",    CONDA_ENV)
        job_script = job_script.replace("%SIL_THRESHOLD%",  str(SIL_THRESHOLD))
        job_script = job_script.replace("%ITERATIONS%",     str(ITERATIONS))

        out_path = Path("jobs") / f"batch_decomp_{run_name}.sh"
        with open(out_path, "w") as f:
            f.write(job_script)

        generated.append(out_path)
        print(f"  Generated: {out_path}")

    print(f"\n{len(generated)} job script(s) written for {subject}.")
    print("\nTo submit, run:")
    for p in generated:
        print(f"  qsub {p}")

    # ── submission ────────────────────────────────────────────────────────────
    # Uncomment to submit automatically:
    # for p in generated:
    #     os.system(f"qsub {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("yaml", help="Path to file-list YAML (e.g. jobs/file_lists/sub01_day01.yaml)")
    args = ap.parse_args()

    generate_jobs(Path(args.yaml))
