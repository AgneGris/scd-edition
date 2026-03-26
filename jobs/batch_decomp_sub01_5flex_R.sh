#!/bin/bash
#PBS -N sub01_5flex_R
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=64gb:ngpus=1
#PBS -o /rds/general/user/ag4916/ephemeral/thinfilm_logs_dir
#PBS -e /rds/general/user/ag4916/ephemeral/thinfilm_logs_dir

# Activate conda environment
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate scd

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Live log file — written during the job so you can tail it while it runs
LIVE_LOG=/rds/general/user/ag4916/ephemeral/thinfilm_logs_dir/sub01_5flex_R_live.txt

# Print GPU and config info to live log immediately
{
echo "=== GPU / environment check ==="
python -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA compiled : {torch.version.cuda}')
print(f'GPU available : {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
if not torch.cuda.is_available():
    print('  WARNING: running on CPU!')
"
echo "iterations     : 150"
echo "sil_threshold  : 0.85"
echo "==============================="
} | tee "$LIVE_LOG"

# Print job parameters
{
    echo "=== Job: sub01_5flex_R ==="
    echo "Channel config : /rds/general/project/thinfilmdata/live/forearm/raw_data/sub-01/channel_config_sub01_day01.json"
    echo "Concatenate    : True"
    echo "Rejections     : /rds/general/project/thinfilmdata/live/forearm/raw_data/sub-01/channel_rejections_sub01_day01.json"
    echo "Output dir     : /rds/general/project/thinfilmdata/live/forearm/output/sub-01"
    echo "sil_threshold  : 0.85"
    echo "iterations     : 150"
    echo "Files:"
    echo "  /rds/general/project/thinfilmdata/live/forearm/raw_data/sub-01/day-01/sub-01_day-01_task-trap_ang-0_mvc-5flex_fing-R_time-1-1-20_run-01.otb+"
    echo "  /rds/general/project/thinfilmdata/live/forearm/raw_data/sub-01/day-01/sub-01_day-01_task-trap_ang-0_mvc-5flex_fing-R_time-1-1-20_run-02.otb+"
    echo ""

    # Copy input files to local SSD for fast I/O
    echo "=== Copying input files to local SSD ==="
    t_copy=$SECONDS
    ORIG_FILES=("/rds/general/project/thinfilmdata/live/forearm/raw_data/sub-01/day-01/sub-01_day-01_task-trap_ang-0_mvc-5flex_fing-R_time-1-1-20_run-01.otb+" "/rds/general/project/thinfilmdata/live/forearm/raw_data/sub-01/day-01/sub-01_day-01_task-trap_ang-0_mvc-5flex_fing-R_time-1-1-20_run-02.otb+")
    LOCAL_FILES=()
    for f in "${ORIG_FILES[@]}"; do
        dest="$TMPDIR/$(basename "$f")"
        echo "  Copying $(basename "$f") ..."
        cp "$f" "$dest"
        LOCAL_FILES+=("$dest")
    done
    echo "  All files copied in $((SECONDS - t_copy))s"
    echo ""

    # Decompose using local copies
    python scripts/batch_decompose.py \
        --channel-config "/rds/general/project/thinfilmdata/live/forearm/raw_data/sub-01/channel_config_sub01_day01.json" \
        --files "${LOCAL_FILES[@]}" \
        --concat \
        --rejections-file "/rds/general/project/thinfilmdata/live/forearm/raw_data/sub-01/channel_rejections_sub01_day01.json" \
        --output "/rds/general/project/thinfilmdata/live/forearm/output/sub-01" \
        --params sil_threshold=0.85 iterations=150

} 2>&1 | tee -a "$LIVE_LOG" > $TMPDIR/sub01_5flex_R.txt

cp $TMPDIR/sub01_5flex_R.txt $PBS_O_WORKDIR/jobs/job_outputs/sub01_5flex_R.txt
