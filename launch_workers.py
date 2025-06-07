import os
import subprocess
import sys
import time

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from config.config import register_configs


@ray.remote(num_gpus=1)
def run_worker(worker_id: int, worker_total: int, base_config: dict) -> None:
    """Run a single worker process with specified ID and total workers."""
    # Update config with worker-specific settings
    config = OmegaConf.create(base_config)
    config.worker_id = worker_id
    config.worker_total = worker_total

    # Save temporary config for this worker
    config_path = f"/tmp/worker_config_{worker_id}.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config=config, f=f)

    # Run the main script with this config
    cmd = [sys.executable, "main_grads.py", f"--config-path={config_path}"]

    subprocess.run(cmd, check=True)
    os.remove(config_path)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Launch multiple workers across GPUs."""
    num_workers = cfg.get("workers", 1)
    # Unset RAY_ADDRESS to ensure local cluster initialization
    if "RAY_ADDRESS" in os.environ:
        del os.environ["RAY_ADDRESS"]

    # Initialize Ray with dashboard
    ray.init(
        runtime_env={"working_dir": "."},
        dashboard_host="0.0.0.0",  # Makes dashboard accessible from other machines
        dashboard_port=8265,  # Default Ray dashboard port
        include_dashboard=True,
        address=None,
    )

    # Convert config to dict for serialization
    base_config = OmegaConf.to_container(cfg, resolve=True)

    # Launch workers
    print(f"Launching {num_workers} workers...")
    futures = [
        run_worker.remote(
            worker_id=i, worker_total=num_workers, base_config=base_config
        )
        for i in range(num_workers)
    ]

    # Wait for all workers to complete while showing status
    pending = futures
    while pending:
        done, pending = ray.wait(pending, timeout=1.0)
        num_completed = len(futures) - len(pending)
        print(
            f"\rProgress: {num_completed}/{len(futures)} workers completed",
            end="",
        )
        time.sleep(1)

    print("\nAll workers completed!")


if __name__ == "__main__":
    register_configs()
    main()
