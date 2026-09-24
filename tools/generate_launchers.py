#!/usr/bin/env python3
"""Generate per-sweep launcher scripts (nohup and SLURM) from saved W&B YAMLs.

Creates:
- tools/launchers/nohup_run_<ID>.sh    (runs command with nohup, writes pid)
- tools/launchers/run_<ID>.sh           (direct executable wrapper)
- tools/launchers/sbatch_<ID>.sh        (SLURM sbatch script)

Usage: python3 tools/generate_launchers.py
Edit the generated scripts to map placeholder paths (e.g., SLURM_TUE) to your local paths.
"""
import shlex
from pathlib import Path
import yaml


def build_command_from_yaml(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    cfg = data.get('config', {})
    program = cfg.get('program', 'train.py')
    params = cfg.get('parameters', {})
    cmd = [shlex.quote(str(__import__('sys').executable)), shlex.quote(program)]
    for key in sorted(params.keys()):
        entry = params[key]
        val = None
        if isinstance(entry, dict):
            if 'value' in entry:
                val = entry['value']
            elif 'values' in entry:
                val = ','.join(map(str, entry['values']))
        else:
            val = entry

        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(shlex.quote(flag))
        elif val is None:
            continue
        else:
            cmd.extend([shlex.quote(flag), shlex.quote(str(val))])

    return cmd


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--map-slurm', type=str, default='SLURM_TUE=./data', help='Mapping for placeholder, format PLACEHOLDER=PATH')
    p.add_argument('--disable-wandb', action='store_true', help='Remove --log-wandb flags from generated commands')
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sweeps_dir = repo_root / 'docs' / 'wandb_sweeps'
    out_dir = repo_root / 'tools' / 'launchers'
    logs_dir = repo_root / 'logs'
    pids_dir = repo_root / 'pids'
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pids_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = sorted(sweeps_dir.glob('*.yaml'))
    if not yaml_files:
        print('No YAML sweep files found in', sweeps_dir)
        return

    print(f'Found {len(yaml_files)} sweep YAMLs; generating launchers in {out_dir}')

    # parse mapping
    placeholder, replacement = args.map_slurm.split('=',1)
    for y in yaml_files:
        sweep_id = y.stem
        cmd_list = build_command_from_yaml(y)
        # join and apply placeholder replacement
        cmd = ' '.join(cmd_list)
        if placeholder in cmd:
            cmd = cmd.replace(placeholder, replacement)
        # optionally remove --log-wandb flags
        if args.disable_wandb:
            cmd = cmd.replace(' --log-wandb', '')

        # Writable wrapper
        run_sh = out_dir / f'run_{sweep_id}.sh'
        run_sh.write_text(f"#!/usr/bin/env bash\n# Run sweep {sweep_id}\n# Replace placeholders like SLURM_TUE with your local paths before running.\n{cmd}\n")
        run_sh.chmod(0o755)

        # nohup runner
        nohup_sh = out_dir / f'nohup_run_{sweep_id}.sh'
        nohup_cmd = f"nohup {cmd} > {logs_dir / (sweep_id + '.log')} 2>&1 & echo $! > {pids_dir / (sweep_id + '.pid')}"
        nohup_sh.write_text(f"#!/usr/bin/env bash\n# NoHUP runner for sweep {sweep_id}\n# Edit placeholders then run: bash {nohup_sh.name}\n{nohup_cmd}\n")
        nohup_sh.chmod(0o755)

        # SLURM sbatch script
        sbatch_sh = out_dir / f'sbatch_{sweep_id}.sh'
        sbatch_content = f"""#!/usr/bin/env bash
#SBATCH --job-name=untangle_{sweep_id}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output={logs_dir / (sweep_id + '.out')}

echo "Starting sweep {sweep_id}"
{cmd}
echo "Finished"
"""
        sbatch_sh.write_text(sbatch_content)
        sbatch_sh.chmod(0o755)

    print('Generator finished. Use the nohup scripts or sbatch scripts to run sweeps detached.')


if __name__ == '__main__':
    main()
