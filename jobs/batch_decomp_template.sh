#!/bin/bash
#PBS -N %RUN_NAME%
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=64gb:ngpus=1
#PBS -o %PATH_TO_LOGS%
#PBS -e %PATH_TO_LOGS%

# Activate conda environment
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate %ENVIRONMENT%

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Live log file — written during the job so you can tail it while it runs
LIVE_LOG=%PATH_TO_LOGS%/%RUN_NAME%_live.txt

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
echo "iterations     : %ITERATIONS%"
echo "sil_threshold  : %SIL_THRESHOLD%"
echo "==============================="
} | tee "$LIVE_LOG"

# Print job parameters
{
    echo "=== Job: %RUN_NAME% ==="
    echo "Channel config : %CHANNEL_CONFIG%"
    echo "Concatenate    : %CONCAT_BOOL%"
    echo "Rejections     : %REJECTIONS_FILE%"
    echo "Output dir     : %OUTPUT_DIR%"
    echo "sil_threshold  : %SIL_THRESHOLD%"
    echo "iterations     : %ITERATIONS%"
    echo "Files:"
%FILES_ECHO%
    echo ""

    # Copy input files to local SSD for fast I/O
    echo "=== Copying input files to local SSD ==="
    t_copy=$SECONDS
    ORIG_FILES=(%FILES_BASH_ARRAY%)
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
        --channel-config "%CHANNEL_CONFIG%" \
        --files "${LOCAL_FILES[@]}" \
%CONCAT_FLAG%        --rejections-file "%REJECTIONS_FILE%" \
        --output "%OUTPUT_DIR%" \
        --params sil_threshold=%SIL_THRESHOLD% iterations=%ITERATIONS%

} 2>&1 | tee -a "$LIVE_LOG" > $TMPDIR/%RUN_NAME%.txt

cp $TMPDIR/%RUN_NAME%.txt $PBS_O_WORKDIR/jobs/job_outputs/%RUN_NAME%.txt
