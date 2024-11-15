import torch
import pytorch_lightning as pl
import json
from transformers import CLIPVisionModel, T5Tokenizer
from . import mm_utils
from . import heads, objectives
from . import dist_utils
from .t5_model import T5ForMultimodalGeneration

# Disable certain optimizations to improve reproducibility
torch.backends.cudnn.enabled = False

class MMTransformerSS(pl.LightningModule):
    """
    MMTransformerSS is a multimodal transformer model for multimodal classification or generation tasks.
    It combines a vision transformer (CLIP model) for image understanding and a T5 model for text generation.
    """
    def __init__(self, config):
        """
        Initializes the multimodal transformer model.
        
        Args:
            config (dict): Configuration dictionary that contains the necessary parameters and paths.
        """
        super().__init__()
        self.save_hyperparameters()
        self.mode = self.hparams.config["mode"]
        self.out_path = self.hparams.config["out_path"]

        # Initialize distributed computing resources if running in a multi-GPU setup
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                CLIPVisionModel.from_pretrained(config["vit"])  # Load Vision Transformer (CLIP model)
                T5ForMultimodalGeneration.from_pretrained(config['tokenizer'])  # Load T5 tokenizer
            torch.distributed.barrier()

        # Initialize the models for vision (CLIP) and text (T5)
        self.image_transformer = CLIPVisionModel.from_pretrained(config["vit"])
        self.text_transformer = T5ForMultimodalGeneration.from_pretrained(
            config['tokenizer'],
            config["input_image_embed_size"],
        )
        self.tokenizer = T5Tokenizer.from_pretrained(config['tokenizer'])
        
        # Freeze the image transformer weights to prevent updates during training
        for param in self.image_transformer.parameters():
            param.requires_grad = False

        # Set up metrics and tasks for the model
        mm_utils.set_metrics(self)
        self.current_tasks = list()

        # Load model checkpoint if specified
        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        
        # Initialize dictionaries for storing predicted and ground truth results
        self.pred_result = {}
        self.gold_result = {}
        
    def encode_image(self, image_features):
        """
        Encodes image features using the vision transformer (CLIP model).

        Args:
            image_features (tensor): The image features to be passed through the model.
        
        Returns:
            tensor: The encoded image features (last hidden state).
        """
        last_hidden_state = self.image_transformer(
            pixel_values=image_features,
        ).last_hidden_state
        return last_hidden_state

    def infer(self, batch):
        """
        Performs inference (forward pass) on a batch of data.
        
        Args:
            batch (dict): The input batch containing text and image data.
        
        Returns:
            dict: A dictionary containing the model's output for the text.
        """
        text_ids = batch[f"text_ids"]
        label_ids = batch[f"label_ids"] if self.mode != "rationale" or "rationale_ids" not in batch else batch[f"rationale_ids"]
        label_ids[label_ids == 0] = -100  # Mask padding tokens in labels
        text_masks = batch[f"text_masks"]
        image_features = batch[f"image_features"]

        # Get encoded image features
        image_features = self.encode_image(image_features)
        
        # Pass text and image features through the text transformer
        text_outputs = self.text_transformer(
            input_ids=text_ids,
            attention_mask=text_masks,
            image_ids=image_features,
            labels=label_ids,
        )

        return {"text_outputs": text_outputs}

    def forward(self, batch):
        """
        The forward pass of the model. Computes the text output and applies objectives.

        Args:
            batch (dict): The input batch containing text and image data.
        
        Returns:
            dict: The output including text predictions and associated loss.
        """
        ret = self.infer(batch)
        ret.update(objectives.compute_clm(self, ret))  # Add any classification or loss objectives
        return ret

    def training_step(self, batch, batch_idx):
        """
        Performs one step of training. This computes the loss for a batch.
        
        Args:
            batch (dict): The input batch containing text and image data.
            batch_idx (int): The index of the batch.
        
        Returns:
            tensor: The total loss for the batch.
        """
        mm_utils.set_task(self)  # Set the current task (e.g., classification, generation)
        output = self(batch)  # Run the forward pass
        total_loss = sum([v for k, v in output.items() if "loss" in k])  # Calculate total loss from output
        return total_loss

    def training_epoch_end(self, outs):
        """
        Executes after the end of each epoch during training. Here, we wrap up the training metrics.
        
        Args:
            outs (list): The outputs from the training steps (not used here).
        """
        mm_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        """
        Performs one step of validation. Computes the outputs for evaluation.
        
        Args:
            batch (dict): The input batch containing text and image data.
            batch_idx (int): The index of the batch.
        
        Returns:
            dict: The model's output for the current batch.
        """
        mm_utils.set_task(self)  # Set the current task for validation
        if self.mode != "rationale":
            text_ids = batch[f"text_ids"]
            image_features = batch[f"image_features"]
            image_features = self.encode_image(image_features)
            self.text_transformer.encoder.update_image_ids(image_features)  # Update image features in the encoder
            self.text_transformer.update_image_ids(image_features)  # Update image features in the model

            # Generate predictions for the current batch
            outputs = self.text_transformer.generate(text_ids, max_length=256)
            pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Store predictions and ground truth for evaluation
            for iid in range(len(pred)):
                self.pred_result[batch["img_index"][iid]] = pred[iid]
                self.gold_result[batch["img_index"][iid]] = batch["label"][iid].split("The answer is: ")[-1].strip()

            ret = dict()  # Return empty dictionary for now (not needed for inference)
        else:
            ret = self(batch)  # For "rationale" mode, perform the normal inference

        return ret

    def validation_epoch_end(self, outs):
        """
        Executes after the end of each validation epoch. Here, we compute accuracy based on predictions.
        
        Args:
            outs (list): The outputs from the validation steps (not used here).
        """
        if self.mode != "rationale":
            correct = 0
            # Compare predicted and ground truth values
            for iid in self.gold_result:
                if iid not in self.pred_result:
                    correct = 0  # If no prediction exists, reset the counter
                    break
                label = self.gold_result[iid]
                pred = self.pred_result[iid].split("The answer is: ")[-1].strip()  # Clean up the prediction
                if pred == label:
                    correct += 1  # Increment correct if prediction matches ground truth
            self.acc = correct / len(self.gold_result)  # Compute accuracy
            self.pred_result = {}  # Reset prediction results for next epoch
        mm_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        """
        Performs one step of testing. This is similar to validation but without ground truth labels.
        
        Args:
            batch (dict): The input batch containing text and image data.
            batch_idx (int): The index of the batch.
        
        Returns:
            dict: The model's output for the current batch.
        """
        mm_utils.set_task(self)

        text_ids = batch[f"text_ids"]
        image_features = batch[f"image_features"]
        image_features = self.encode_image(image_features)
        self.text_transformer.encoder.update_image_ids(image_features)
        self.text_transformer.update_image_ids(image_features)
        
        # Generate predictions for the test batch
        outputs = self.text_transformer.generate(text_ids, max_length=256)
        pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.pred_result[batch["img_index"][0]] = pred  # Store prediction

        ret = dict()  # Return empty dictionary for now (no additional output needed)

        return ret

    def test_epoch_end(self, outs):
        """
        Executes after the end of each test epoch. This saves the predictions to a file.
        
        Args:
            outs (list): The outputs from the test steps (not used here).
        """
        # Save the predictions to a JSON file
        with open(self.out_path, "w") as fout:
            json.dump(self.pred_result, fout)
        mm_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            tuple: The optimizer and scheduler for the model.
        """
        return mm_utils.set_schedule(self)
