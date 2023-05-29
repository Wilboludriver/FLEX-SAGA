#SBATCH -n 16                              # Number of cores
#SBATCH --time=16:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                         # per node!!
#SBATCH --gpus=1                           # Ohter options are: --gpus=gtx_1080_ti:1  --gpus=rtx_2080_ti:1
#SBATCH --job-name=gen_job
#SBATCH --output=./jobs/gen_job.out
#SBATCH --error=./jobs/gen_job.err
srun --gpus=rtx_2080_ti:1 -n 8 --mem-per-cpu=32000 --time=2:00:00 --pty bash 