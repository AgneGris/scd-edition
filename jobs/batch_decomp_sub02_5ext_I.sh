#!/bin/bash
#PBS -N sub02_5ext_I
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
LIVE_LOG=/rds/general/user/ag4916/ephemeral/thinfilm_logs_dir/sub02_5ext_I_live.txt

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
    echo "=== Job: sub02_5ext_I ==="
    echo "Channel config : /rds/general/project/thinfilmdata/live/forearm/raw_data/sub-02/channel_config_sub02_day01.json"
    echo "Concatenate    : False"
    echo "Rejections     : /rds/general/project/thinfilmdata/live/forearm/raw_data/sub-02/channel_rejections_sub02_day01.json"
    echo "Output dir     : /rds/general/project/thinfilmdata/live/forearm/output/sub-02"
    echo "sil_threshold  : 0.85"
    echo "iterations     : 150"
    echo "Files:"
    echo "  /rds/general/project/thinfilmdata/live/forearm/raw_data/sub-02/day-01/sub-02_day-01_task-trap_ang-0_mvc-5ext_fing-I_time-1-1-20_reps-02_run-01.otb+"
    echo ""

    # Copy input files to local SSD for fast I/O
    echo "=== Copying input files to local SSD ==="
    t_copy=$SECONDS
    ORIG_FILES=("/rds/general/project/thinfilmdata/live/forearm/raw_data/sub-02/day-01/sub-02_day-01_task-trap_ang-0_mvc-5ext_fing-I_time-1-1-20_reps-02_run-01.otb+")
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
        --channel-config "/rds/general/project/thinfilmdata/live/forearm/raw_data/sub-02/channel_config_sub02_day01.json" \
        --files "${LOCAL_FILES[@]}" \
        --rejections-file "/rds/general/project/thinfilmdata/live/forearm/raw_data/sub-02/channel_rejections_sub02_day01.json" \
        --output "/rds/general/project/thinfilmdata/live/forearm/output/sub-02" \
        --params sil_threshold=0.85 iterations=150

} 2>&1 | tee -a "$LIVE_LOG" > $TMPDIR/sub02_5ext_I.txt

cp $TMPDIR/sub02_5ext_I.txt $PBS_O_WORKDIR/jobs/job_outputs/sub02_5ext_I.txt
