import random
import torch
import io
import pandas as pd
import os
import jsonlines
from PIL import Image
from ..transforms import keys_to_transforms

def jsonl_reader(path):
    """
    Reads a JSON Lines file and returns a list of objects (records).
    Each line in the file corresponds to a separate JSON object.
    """
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return data

class JsonDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        input_filename,
        transform_keys,
        image_size,
        patch_size,
        img_key,
        text_key,
        label_key,
        rationale_key,
        tokenizer=None,
        max_text_len=50,
        image_only=False,
    ):
        """
        Initializes the dataset by reading data and setting up transformations.
        
        Args:
            data_dir (str): Path to the directory containing dataset files.
            input_filename (str): Name of the input JSONL file.
            transform_keys (list): List of keys used to apply image transformations.
            image_size (int): The target size for the images.
            patch_size (int): The patch size for vision models.
            img_key (str): Key to access image paths in the dataset.
            text_key (str): Key to access text input in the dataset.
            label_key (str): Key to access the label in the dataset.
            rationale_key (str): Key to access the rationale (explanation) in the dataset.
            tokenizer (transformers tokenizer): Tokenizer for processing text data.
            max_text_len (int): Maximum length for tokenized text.
            image_only (bool): Flag to indicate whether to process images only.
        """
        assert len(transform_keys) >= 1  # Ensure at least one transformation key is provided.
        super().__init__()
        
        self.data_dir = data_dir
        self.image_only = image_only
        self.data = jsonl_reader(f"{data_dir}/{input_filename}")  # Read JSONL file
        self.img_key = img_key
        self.text_key = text_key
        self.label_key = label_key
        self.rationale_key = rationale_key
        self.transforms = keys_to_transforms(transform_keys, size=image_size)  # Set up image transformations
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.patch_size = patch_size
        self.tokenizer = tokenizer

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data) if self.data else 0
    
    def get_image(self, idx):
        """Loads and transforms the image for the given index."""
        image_path = f"{self.data_dir}/images/{str(self.data[idx][self.img_key])}"
        image_features = self.transforms[0](Image.open(image_path)).unsqueeze(0)  # Apply transformation
        return {
            "image_features": image_features,  # [1, 3, H, W]
            "raw_index": idx,
            "img_path": image_path,
            "img_index": self.data[idx]["id"],
        }

    def get_text(self, idx):
        """Tokenizes and returns the text for the given index."""
        text = str(self.data[idx][self.text_key]).lower()
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "raw_index": idx,
        }
    
    def get_label(self, idx):
        """Tokenizes and returns the label for the given index."""
        text = "The answer is: " + str(self.data[idx][self.label_key][0]).lower()
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_special_tokens_mask=True,
        )
        return {
            "label": (text, encoding),
            "raw_index": idx,
        }
    
    def get_rationale(self, idx):
        """Tokenizes and returns the rationale for the given index."""
        text = "Output: " + str(self.data[idx][self.rationale_key][0]).lower()
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "rationale": (text, encoding),
            "raw_index": idx,
        }

    def get_suite(self, idx):
        """
        Retrieves all relevant information (image, text, label, rationale) for a given index.
        Handles exceptions gracefully by retrying until valid data is retrieved.
        """
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(idx))  # Get image features
                if not self.image_only:
                    ret.update(self.get_text(idx))  # Get text data
                    ret.update(self.get_label(idx))  # Get label data
                    try:
                        ret.update(self.get_rationale(idx))  # Get rationale if available
                    except:
                        pass
                result = True  # Break the loop once data is retrieved successfully
            except Exception as e:
                print(f"Error while reading file idx {idx} -> {e}")
                idx = random.randint(0, len(self.data) - 1)  # Retry with a random index if error occurs

        return ret

    def collate(self, batch, mlm_collator):
        """
        Collates the batch of data into tensors for training.
        This function is crucial for handling variable-length sequences and performing MLM.
        """
        batch_size = len(batch)
        
        # Collect all the keys in the batch
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # Combine image features in a single tensor
        batch_image_features = torch.cat(dict_batch["image_features"], dim=0)  # [bs, 3, H, W]
        dict_batch["image_features"] = batch_image_features

        # Process text-related keys
        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]

            # Prepare text for MLM
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_masks"] = attention_mask

        # Process label-related keys
        label_keys = [k for k in list(dict_batch.keys()) if "label" in k]

        if len(label_keys) != 0:
            labels = [[d[0] for d in dict_batch[label_key]] for label_key in label_keys]
            encodings = [[d[1] for d in dict_batch[label_key]] for label_key in label_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]

            flatten_mlms = mlm_collator(flatten_encodings)

            for i, label_key in enumerate(label_keys):
                labels, encodings = (
                    [d[0] for d in dict_batch[label_key]],
                    [d[1] for d in dict_batch[label_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[label_key] = labels
                dict_batch[f"{label_key}_ids"] = input_ids
                dict_batch[f"{label_key}_masks"] = attention_mask

        # Process rationale-related keys
        rationale_keys = [k for k in list(dict_batch.keys()) if "rationale" in k]

        if len(rationale_keys) != 0:
            rationales = [[d[0] for d in dict_batch[rationale_key]] for rationale_key in rationale_keys]
            encodings = [[d[1] for d in dict_batch[rationale_key]] for rationale_key in rationale_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]

            flatten_mlms = mlm_collator(flatten_encodings)

            for i, rationale_key in enumerate(rationale_keys):
                rationales, encodings = (
                    [d[0] for d in dict_batch[rationale_key]],
                    [d[1] for d in dict_batch[rationale_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[rationale_key] = rationales
                dict_batch[f"{rationale_key}_ids"] = input_ids
                dict_batch[f"{rationale_key}_masks"] = attention_mask

        return dict_batch
