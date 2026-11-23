#!/bin/bash
# sjob_multi_train.sh - Submit multiple training jobs with different parameters.

set -euo pipefail

# Shared cluster configuration
CONDA_ENV=fognet
QOS="gpu"
MEM=64000
CPUS=32
GPU_PARTITION="gpu"
CUDA_MODULE="cuda/12.2"

# Where Slurm files will be written
JOB_DIR="jobs"
mkdir -p "$JOB_DIR"

# Each entry is: job_name|command_line_arguments
# Edit this block to launch the parameter sweep you want.
RUN_SPECS=(
  "featurenet_posenc_fc32|--dir-name Fluo-N2DH-GOWT1 --max-epochs 400 --feat-channels 8 --transformer"
  "featurenet_no_posenc_fc32|--dir-name Fluo-N2DH-GOWT1 --max-epochs 400 --feat-channels 8 --unet" #no transformer
)


for spec in "${RUN_SPECS[@]}"; do
  job_name="${spec%%|*}"
  job_args="${spec#*|}"

  datetime=$(date +"%y%m%d-%H%M%S")
  log_file="${datetime}_${job_name}"
  job_file="${JOB_DIR}/${log_file}.sbatch"

  cat <<EOF > "$job_file"
#!/bin/bash
#SBATCH -J "${job_name}"
#SBATCH -o ${JOB_DIR}/${log_file}.out
#SBATCH -e ${JOB_DIR}/${log_file}.err
#SBATCH --qos=${QOS}
#SBATCH -c ${CPUS}
#SBATCH -p ${GPU_PARTITION}
#SBATCH --mem ${MEM}
#SBATCH --gres=gpu

module load ${CUDA_MODULE}

eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

srun python ./train_cli.py ${job_args} || exit 1

exit 0
EOF

  sbatch "$job_file"
  echo "submitted ${job_name}"
done