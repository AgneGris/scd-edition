#!/bin/bash
#PBS -N %RUN_NAME%
#PBS -l walltime=15:00:00
#PBS -l select=1:ncpus=1:mem=64gb:ngpus=1
#PBS -o %PATH_TO_LOGS%
#PBS -e %PATH_TO_LOGS%

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate %ENVIRONMENT%

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

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

    # Decompose
    python scripts/batch_decompose.py \
        --channel-config "%CHANNEL_CONFIG%" \
        --files \
%FILES_ARGS%
%CONCAT_FLAG%        --rejections-file "%REJECTIONS_FILE%" \
        --output "%OUTPUT_DIR%" \
        --params sil_threshold=%SIL_THRESHOLD% iterations=%ITERATIONS%

} > $TMPDIR/%RUN_NAME%.txt 2>&1

cp $TMPDIR/%RUN_NAME%.txt $PBS_O_WORKDIR/jobs/job_outputs/%RUN_NAME%.txt
