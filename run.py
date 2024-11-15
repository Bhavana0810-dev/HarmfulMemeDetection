import os
import copy
import pytorch_lightning as pl
from src.config import ex
from src.modules import MMTransformerSS
from src.datamodules.multitask_datamodule import MTDataModule

# Setting environment variable for NCCL debug messages to help troubleshoot multi-GPU setup
os.environ["NCCL_DEBUG"] = "INFO"

# Resource management to handle file descriptor limits (optional, you can uncomment if needed)
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))  # Uncomment if system allows

# Main entry point for the experiment
@ex.automain
def main(_config):
    # Deep copy the configuration to avoid changes to the original
    _config = copy.deepcopy(_config)

    # Initialize seed for reproducibility
    pl.seed_everything(_config["seed"])

    # Data module setup, enabling distributed training
    dm = MTDataModule(_config, dist=True)

    # Model initialization
    model = MMTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'  # Experiment name used in logs

    # Create directory to save logs and checkpoints
    os.makedirs(_config["log_dir"], exist_ok=True)

    # Setting up ModelCheckpoint callback to save best model and last checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/the_metric",  # The metric to monitor (e.g., validation accuracy)
        mode="max" if _config["mode"] != "rationale" else "min",  # "max" or "min" based on problem type
        save_top_k=1,  # Save the best model based on validation metric
        save_last=True,  # Save the last checkpoint
        verbose=True
    )

    # TensorBoard logger to log training metrics
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',  # Logging experiment name with seed and model details
    )

    # Learning rate monitor callback to log learning rate at each step
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # List of callbacks to pass to the trainer
    callbacks = [checkpoint_callback, lr_callback]

    # Determine number of GPUs to use
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])  # If multiple GPUs specified
    )

    # Calculate gradient accumulation steps based on batch size and number of GPUs
    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)

    # Define the maximum number of steps to run. If None, no limit on steps
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # Trainer setup: Multi-GPU, DDP, and other configurations
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],  # Number of GPUs
        num_nodes=_config["num_nodes"],  # Number of nodes in distributed setup
        precision=_config["precision"],  # Precision (16 or 32 bit)
        accelerator="ddp",  # Distributed Data Parallel setup
        benchmark=True,  # Enable CUDA benchmarking for better performance
        deterministic=True,  # For reproducibility
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,  # Maximum number of epochs
        max_steps=max_steps,  # Maximum number of training steps
        callbacks=callbacks,  # Pass the callbacks
        logger=logger,  # TensorBoard logger
        prepare_data_per_node=False,  # Do not prepare data per node
        replace_sampler_ddp=False,  # Do not replace sampler for DDP
        accumulate_grad_batches=grad_steps,  # Gradients will accumulate over these steps
        log_every_n_steps=10,  # Log every 10 steps
        flush_logs_every_n_steps=10,  # Flush logs every 10 steps
        resume_from_checkpoint=_config["resume_from"],  # Resume training from checkpoint
        weights_summary="top",  # Display only the top of the model summary
        fast_dev_run=_config["fast_dev_run"],  # Fast debugging (1 batch run)
        val_check_interval=_config["val_check_interval"],  # Validation check interval
    )

    # Start training or testing based on the configuration
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)  # Fit the model
    else:
        trainer.test(model, datamodule=dm)  # Only test the model if test_only is True
