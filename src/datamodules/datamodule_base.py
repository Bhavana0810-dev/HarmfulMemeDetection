import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    AutoTokenizer,
)

# Utility function to get pretrained tokenizer
def get_pretrained_tokenizer(from_pretrained):
    """
    Retrieves a pretrained tokenizer from the specified model.
    Ensures the tokenizer is loaded properly in distributed training.
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            AutoTokenizer.from_pretrained(from_pretrained)
        torch.distributed.barrier()  # Synchronize across processes

    return AutoTokenizer.from_pretrained(from_pretrained)


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        """
        Initialize the data module. The configuration dictionary is expected to contain all
        necessary parameters like image size, batch size, tokenizer, etc.
        """
        super().__init__()

        self.data_dir = _config["data_root"]  # Directory for the dataset
        self.num_workers = _config["num_workers"]  # Number of workers for data loading
        self.batch_size = _config["per_gpu_batchsize"]  # Batch size per GPU
        self.eval_batch_size = self.batch_size  # Batch size for evaluation

        self.image_size = _config["image_size"]  # Image size for resizing
        self.patch_size = _config["patch_size"]  # Patch size for vision models
        self.max_text_len = _config["max_text_len"]  # Max length of the text input

        # Define transformation keys based on provided config or default ones
        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        # Initialize tokenizer using the pretrained model specified in the config
        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size  # Store the vocabulary size

        # Choose the appropriate data collator for MLM (Masked Language Model) or Whole Word Masking
        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        # Initialize MLM collator with the probability of MLM
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=False, mlm_probability=_config["mlm_prob"]
        )
        self.setup_flag = False  # To track whether the setup has been done

    # Abstract properties for datasets. These need to be implemented in subclass.
    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        """
        Initializes the training dataset by creating an instance of the dataset class
        with necessary transformations and configurations.
        """
        self.train_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            transform_keys=self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            patch_size=self.patch_size,
            max_text_len=self.max_text_len,
            tokenizer=self.tokenizer,
        )

    def set_val_dataset(self):
        """
        Initializes the validation dataset similarly to the training dataset, but with a
        validation split.
        """
        self.val_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            transform_keys=self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            patch_size=self.patch_size,
            max_text_len=self.max_text_len,
            tokenizer=self.tokenizer,
        )

    def set_test_dataset(self):
        """
        Initializes the test dataset, typically using the validation transformation keys.
        """
        self.test_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            transform_keys=self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            patch_size=self.patch_size,
            max_text_len=self.max_text_len,
            tokenizer=self.tokenizer,
        )
    
    def make_val_dset(self, image_only=False):
        """
        Creates a validation dataset with an optional flag for image-only datasets.
        """
        return self.dataset_cls(
            data_dir=self.data_dir,
            transform_keys=self.val_transform_keys,
            split="test",  # 'test' can also be used as a validation dataset
            image_size=self.image_size,
            patch_size=self.patch_size,
            max_text_len=self.max_text_len,
            image_only=image_only,  # Flag to load only images without text
            tokenizer=self.tokenizer,
        )

    def setup(self, stage):
        """
        Initializes datasets only once, ensuring they're set up for all stages (train, val, test).
        This method is called by the Lightning framework before training and evaluation.
        """
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            # Assign the tokenizer to datasets
            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True  # Mark setup as done

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset. Handles batching, shuffling,
        and parallel data loading.
        """
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,  # Custom collate function for batching
            drop_last=False,  # Avoid dropping the last batch
        )
        return loader

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset. Ensures no shuffling for validation.
        """
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,  # No shuffle for evaluation
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,  # Custom collate function
            drop_last=False
        )
        return loader

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset, used similarly to validation.
        """
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
            drop_last=False
        )
        return loader
