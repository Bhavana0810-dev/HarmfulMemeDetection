import unittest
from unittest.mock import patch, MagicMock
import pytorch_lightning as pl
import os
from src.config import ex
from src.modules import MMTransformerSS
from src.datamodules.multitask_datamodule import MTDataModule
import torch

class TestMainFunction(unittest.TestCase):

    @patch('src.config.ex.automain')
    def test_main_function(self, mock_automain):
        # Mocking the config dictionary
        mock_config = {
            'seed': 42,
            'exp_name': 'TestExperiment',
            'log_dir': './logs',
            'mode': 'rationale',
            'num_gpus': 1,
            'num_nodes': 1,
            'per_gpu_batchsize': 32,
            'batch_size': 32,
            'max_steps': None,
            'max_epoch': 10,
            'precision': 32,
            'resume_from': None,
            'fast_dev_run': False,
            'val_check_interval': 1.0,
            'test_only': False,
            'load_path': './model_checkpoint.pth'
        }
        
        # Mocking the MTDataModule and MMTransformerSS classes
        mock_dm = MagicMock(spec=MTDataModule)
        mock_model = MagicMock(spec=MMTransformerSS)

        # Patching necessary parts of pytorch_lightning to avoid actual training
        with patch('pytorch_lightning.Trainer') as mock_trainer, patch('os.makedirs') as mock_makedirs:
            # Return mock trainer object when Trainer is initialized
            mock_trainer.return_value = MagicMock(spec=pl.Trainer)

            # Call the main function with the mock configuration
            ex.automain(mock_config)

            # Assert that the model was correctly initialized with the config
            mock_model.assert_called_once_with(mock_config)

            # Assert that the data module was initialized with the config
            mock_dm.assert_called_once_with(mock_config, dist=True)

            # Check if the Trainer was called with the correct parameters
            mock_trainer.assert_called_once_with(
                gpus=mock_config["num_gpus"],
                num_nodes=mock_config["num_nodes"],
                precision=mock_config["precision"],
                accelerator="ddp",
                benchmark=True,
                deterministic=True,
                max_epochs=mock_config["max_epoch"],
                max_steps=mock_config["max_steps"],
                callbacks=[mock.ANY],  # Checking if callbacks are passed
                logger=mock.ANY,  # Checking if logger is passed
                prepare_data_per_node=False,
                replace_sampler_ddp=False,
                accumulate_grad_batches=mock_config["batch_size"] // (mock_config["per_gpu_batchsize"] * mock_config["num_gpus"] * mock_config["num_nodes"]),
                log_every_n_steps=10,
                flush_logs_every_n_steps=10,
                resume_from_checkpoint=mock_config["resume_from"],
                weights_summary="top",
                fast_dev_run=mock_config["fast_dev_run"],
                val_check_interval=mock_config["val_check_interval"]
            )

            # Check that the log directory is being created
            mock_makedirs.assert_called_once_with(mock_config["log_dir"], exist_ok=True)

    @patch('src.config.ex.automain')
    def test_callbacks_and_logger(self, mock_automain):
        # Mock the configuration
        mock_config = {
            'seed': 42,
            'exp_name': 'TestExperiment',
            'log_dir': './logs',
            'mode': 'rationale',
            'num_gpus': 1,
            'num_nodes': 1,
            'per_gpu_batchsize': 32,
            'batch_size': 32,
            'max_steps': None,
            'max_epoch': 10,
            'precision': 32,
            'resume_from': None,
            'fast_dev_run': False,
            'val_check_interval': 1.0,
            'test_only': False,
            'load_path': './model_checkpoint.pth'
        }

        # Mocking the MTDataModule and MMTransformerSS classes
        mock_dm = MagicMock(spec=MTDataModule)
        mock_model = MagicMock(spec=MMTransformerSS)

        with patch('pytorch_lightning.Trainer') as mock_trainer, patch('pytorch_lightning.loggers.TensorBoardLogger') as mock_logger:
            # Mocking TensorBoardLogger initialization
            mock_logger.return_value = MagicMock(spec=pl.loggers.TensorBoardLogger)
            mock_trainer.return_value = MagicMock(spec=pl.Trainer)

            # Call main function with mock configuration
            ex.automain(mock_config)

            # Assert that the TensorBoardLogger was initialized with the correct directory
            mock_logger.assert_called_once_with(
                mock_config["log_dir"],
                name=f'{mock_config["exp_name"]}_seed{mock_config["seed"]}_from_{mock_config["load_path"].split("/")[-1][:-5]}'
            )

            # Assert that ModelCheckpoint and LearningRateMonitor callbacks are passed to the trainer
            mock_trainer.assert_called_once_with(
                callbacks=[mock.ANY],  # Ensure that the callbacks are passed
                logger=mock_logger.return_value
            )

    @patch('src.config.ex.automain')
    def test_trainer_execution(self, mock_automain):
        # Mock the configuration
        mock_config = {
            'seed': 42,
            'exp_name': 'TestExperiment',
            'log_dir': './logs',
            'mode': 'rationale',
            'num_gpus': 1,
            'num_nodes': 1,
            'per_gpu_batchsize': 32,
            'batch_size': 32,
            'max_steps': None,
            'max_epoch': 10,
            'precision': 32,
            'resume_from': None,
            'fast_dev_run': False,
            'val_check_interval': 1.0,
            'test_only': False,
            'load_path': './model_checkpoint.pth'
        }

        # Mock the MTDataModule and MMTransformerSS
        mock_dm = MagicMock(spec=MTDataModule)
        mock_model = MagicMock(spec=MMTransformerSS)

        with patch('pytorch_lightning.Trainer') as mock_trainer:
            # Simulate the trainer's fit method
            mock_trainer.return_value.fit = MagicMock(return_value=None)

            # Call the main function with the mock config
            ex.automain(mock_config)

            # Ensure the `fit` method is called on the trainer
            mock_trainer.return_value.fit.assert_called_once_with(mock_model, datamodule=mock_dm)

    @patch('src.config.ex.automain')
    def test_test_only_mode(self, mock_automain):
        # Mock test-only mode
        mock_config = {
            'seed': 42,
            'exp_name': 'TestExperiment',
            'log_dir': './logs',
            'mode': 'rationale',
            'num_gpus': 1,
            'num_nodes': 1,
            'per_gpu_batchsize': 32,
            'batch_size': 32,
            'max_steps': None,
            'max_epoch': 10,
            'precision': 32,
            'resume_from': None,
            'fast_dev_run': False,
            'val_check_interval': 1.0,
            'test_only': True,  # Set test_only to True
            'load_path': './model_checkpoint.pth'
        }

        # Mock the MTDataModule and MMTransformerSS
        mock_dm = MagicMock(spec=MTDataModule)
        mock_model = MagicMock(spec=MMTransformerSS)

        with patch('pytorch_lightning.Trainer') as mock_trainer:
            # Simulate the trainer's test method
            mock_trainer.return_value.test = MagicMock(return_value=None)

            # Call the main function with test-only config
            ex.automain(mock_config)

            # Ensure the `test` method is called on the trainer
            mock_trainer.return_value.test.assert_called_once_with(mock_model, datamodule=mock_dm)

if __name__ == "__main__":
    unittest.main()
