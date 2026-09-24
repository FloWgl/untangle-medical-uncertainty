"""Submits a Weights & Biases sweep to Slurm.

This script builds a small Slurm bash script that copies dataset exports
to local scratch (when appropriate) and runs `wandb agent` inside a
Singularity container.
"""

import argparse
import re
import subprocess
from pathlib import Path
from shutil import which

parser = argparse.ArgumentParser(description="Submit Weights & Biases sweeps to Slurm")
parser.add_argument("sweep-id", type=str, help="The Weights & Biases sweep ID")
parser.add_argument(
    "--username", type=str, default="bmucsanyi", help="The Weights & Biases username"
)
parser.add_argument(
    "--project", type=str, default="untangle", help="The Weights & Biases project"
)
parser.add_argument(
    "--count", type=int, default=1, help="Number of runs to query from the sweep"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="pathmnist",
    choices=["pathmnist", "pathmnist-c", "hard/pathmnist", "hard/pathmnist-c"],
    help="Dataset name",
)
parser.add_argument(
    "--simg-path",
    type=Path,
    default=Path("/mnt/lustre/work/oh/owl569/repos/untangle/untangle.simg"),
    help="Path to Singularity image",
)
parser.add_argument(
    "--repo-path",
    type=Path,
    default=Path("/mnt/lustre/work/oh/owl569/repos/untangle"),
    help="Path to repository",
)
parser.add_argument(
    "--datasets-root-path",
    type=Path,
    default=Path("/mnt/lustre/work/oh/owl569/datasets"),
    help="Root path of datasets",
)
parser.add_argument("--job-name", type=str, default=None, help="Job name")
parser.add_argument(
    "--partition",
    type=str,
    default="2080-galvani",
    choices=["2080-galvani", "a100-galvani"],
    help="Slurm partition",
)
parser.add_argument("--cpus-per-task", type=int, default=12, help="Number of CPUs")
parser.add_argument("--mem-per-cpu", type=str, default="4G", help="Available RAM per CPU")
parser.add_argument("--gres", type=str, default="gpu:1", help="GPU resources to allocate")
parser.add_argument("--time", type=str, default="3-00:00:00", help="Maximum runtime D-HH:MM:SS")
parser.add_argument(
    "--log-path",
    type=Path,
    default=Path("/mnt/lustre/work/oh/owl569/logs"),
    help="The output file will be stored in this folder",
)
parser.add_argument("--constraint", type=str, default=None, help="Target node constraint")
parser.add_argument("--exclude", type=str, default=None, help="Exclude specific nodes")
parser.add_argument("--mail-type", type=str, default=None, help="Event type(s) for email notification")
parser.add_argument("--mail-user", type=str, default=None, help="Email address for email notification")


class SlurmJob:
    """Submits jobs to the Slurm cluster by writing and calling `sbatch`.

    The class mirrors the previous functionality but is kept compact here.
    """

    def __init__(
        self,
        cmd_str: str,
        job_name: str,
        partition: str,
        cpus_per_task: int,
        mem_per_cpu: str | None,
        mem: str | None,
        gres: str,
        time: str,
        log_path: Path,
        constraint: str | None,
        exclude: str | None,
        mail_type: str | None,
        mail_user: str | None,
    ) -> None:
        self._cmd_str = cmd_str
        self._job_name = job_name
        self._partition = partition
        self._cpus_per_task = cpus_per_task
        self._mem_per_cpu = mem_per_cpu
        self._mem = mem
        self._gres = gres
        self._time = time
        self._log_path = log_path
        self._constraint = constraint
        self._exclude = exclude
        self._mail_type = mail_type
        self._mail_user = mail_user
        self._output_file_path = self._create_file_paths()

        if mem_per_cpu is None and mem is None:
            raise ValueError("Either mem_per_cpu or mem must be set")
        if mail_type is not None and mail_user is None:
            raise ValueError("mail_user must be set when mail_type is provided")

    def _create_file_paths(self) -> Path:
        return self._log_path / f"%j_{self._job_name}.out"

    def _create_sbatch_str(self) -> str:
        mem_option = f"--mem={self._mem}" if self._mem is not None else f"--mem-per-cpu={self._mem_per_cpu}"
        sbatch_str = (
            f"#SBATCH --job-name={self._job_name}\n"
            f"#SBATCH --partition={self._partition}\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks=1\n"
            f"#SBATCH --cpus-per-task={self._cpus_per_task}\n"
            f"#SBATCH {mem_option}\n"
            f"#SBATCH --gres={self._gres}\n"
            f"#SBATCH --time={self._time}\n"
            f"#SBATCH --output={self._output_file_path}\n"
        )
        if self._constraint is not None:
            sbatch_str += f"#SBATCH --constraint={self._constraint}\n"
        if self._exclude is not None:
            sbatch_str += f"#SBATCH --exclude={self._exclude}\n"
        if self._mail_type is not None:
            sbatch_str += f"#SBATCH --mail-type={self._mail_type}\n"
            sbatch_str += f"#SBATCH --mail-user={self._mail_user}\n"
        return sbatch_str

    def _create_bash_str(self) -> str:
        return f"#!/bin/bash\n\n{self._create_sbatch_str()}\n{self._cmd_str}"

    @staticmethod
    def _sbatch_exists() -> bool:
        return which("sbatch") is not None

    def submit(self) -> None:
        if not self._sbatch_exists():
            raise RuntimeError("No 'sbatch' command found on the system")
        self._log_path.mkdir(parents=True, exist_ok=True)
        bash_file_path = Path(f"{self._job_name}.sh")
        with bash_file_path.open("w") as f:
            f.write(self._create_bash_str())
        bash_file_path.chmod(0o700)
        try:
            subprocess.run(["/usr/bin/sbatch", str(bash_file_path)], check=True)
        finally:
            bash_file_path.unlink()


def get_cmd_str(args: argparse.Namespace) -> str:
    """Builds the setup shell commands and the run string.

    When running on the `2080-galvani` partition, this copies local dataset
    exports into `/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets` so that
    experiments can run without network.
    """
    setup_str = ""
    if args.partition == "2080-galvani":
        setup_str = "mkdir /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets\n"
        if args.dataset == "imagenet":
            setup_str += (
                f"cp {args.datasets_root_path}/raters.npz "
                "/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/raters.npz\n"
                f"cp {args.datasets_root_path}/real.json "
                "/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/real.json\n"
            )
        elif "pathmnist" in args.dataset:
            setup_str += (
                f"cp -r {args.datasets_root_path}/pathmnist "
                "/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/pathmnist\n"
            )
            # Canonical PathMNIST-C location: <datasets_root>/pathmnist_c/pathmnist/*.npz
            setup_str += (
                "mkdir -p /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/pathmnist_c\n"
                f"if [ -d \"{args.datasets_root_path}/pathmnist_c/pathmnist\" ]; then\n"
                f"  cp -r \"{args.datasets_root_path}/pathmnist_c/pathmnist\" \"/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/pathmnist_c/pathmnist\"\n"
                "fi\n"
            )
        else:
            setup_str += (
                f"cp -r {args.datasets_root_path}/cifar-10-batches-py "
                "/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/cifar-10-batches-py\n"
                f"cp -r {args.datasets_root_path}/CIFAR10H "
                "/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/CIFAR10H\n"
            )

    run_str = (
        rf"singularity exec --bind /:/host --nv "
        rf'--pwd "/host/{args.repo_path}" {args.simg_path} '
        rf'bash -c "wandb agent --count {args.count} '
        rf'{args.username}/{args.project}/{getattr(args, "sweep-id")}"'
    )

    cmd_str = run_str
    if setup_str:
        cmd_str = f"{setup_str}\n\n{cmd_str}"
    return cmd_str


def main() -> None:
    args = parser.parse_args()
    cmd_str = get_cmd_str(args)
    if args.job_name is None:
        args.job_name = getattr(args, "sweep-id")

    slurm_job = SlurmJob(
        cmd_str=cmd_str,
        job_name=args.job_name,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu=args.mem_per_cpu,
        mem=None,
        gres=args.gres,
        time=args.time,
        log_path=args.log_path,
        constraint=args.constraint,
        exclude=args.exclude,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
    )
    slurm_job.submit()


if __name__ == "__main__":
    main()
