#!/bin/bash
# sjob_wrapper.sh - Dynamically generate an SBATCH script with modifiable parameters and launch command

# Configurable variables

JOB_NAME=featurenet

DATETIME=$(date +"%y%m%d-%H%M%S")
LOG_FILE="${DATETIME}_${JOB_NAME}"


CONDA_ENV=fognet

EMAIL="trmannev@pasteur.fr"
HOST_PATH="smb://helix.pasteur.fr/projects/bia/trmannev/FeatureNet"
QOS="gpu"
MEM=64000
CPUS=32

LAUNCH_COMMAND="python ./train.py"

echo $LAUNCH_COMMAND

JOB_FILE=jobs/${LOG_FILE}.sbatch

# Generate the SBATCH script with variable expansion
cat <<EOF > $JOB_FILE
#!/bin/bash
#SBATCH -J "${JOB_NAME}"
#SBATCH -o jobs/${LOG_FILE}.out
#SBATCH -e jobs/${LOG_FILE}.err
#SBATCH --qos=${QOS}
#SBATCH -c ${CPUS}
#SBATCH -p gpu
#SBATCH --mem ${MEM}
#SBATCH --gres=gpu

module load cuda/12.2

eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

srun ${LAUNCH_COMMAND} || exit 1

exit 0
EOF

# Submit the generated job script
sbatch $JOB_FILE
echo $JOB_NAME "submitted"
