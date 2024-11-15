from sacred import Experiment

# Initialize the experiment with the name "Meme"
ex = Experiment("Meme", save_git_info=False)

# Helper function to add additional loss names to the default loss
def _loss_names(d):
    ret = {
        "car": 0,  # Example loss entry for 'car'
    }
    ret.update(d)  # Update with additional losses provided by input `d`
    return ret


# Default configuration for the experiment
@ex.config
def config():
    exp_name = "Meme"  # Experiment name
    seed = 0  # Random seed for reproducibility
    datasets = ["meme"]  # Datasets to be used
    loss_names = _loss_names({"clm": 1})  # Add 'clm' loss to the list of loss names
    batch_size = 4096  # Desired batch size; will accumulate gradients if per-step batch is smaller
    temperature = 0.05  # Temperature setting for softmax scaling (for classification tasks)

    # Image preprocessing settings
    train_transform_keys = ["vit"]  # Transformation keys for training data
    val_transform_keys = ["vit"]  # Transformation keys for validation data
    image_size = 224  # Input image size (224x224)
    patch_size = 16  # Size of the patches for the image (16x16)

    # Text processing settings
    max_text_len = 40  # Maximum text length (in terms of token count)
    tokenizer = "t5-small"  # Pretrained tokenizer to be used
    vocab_size = 32128  # Vocabulary size for the tokenizer
    whole_word_masking = False  # Whether to apply whole-word masking (not compatible with RoBERTa)
    mlm_prob = 0.3  # Probability for masked language model (MLM) training

    # Transformer model settings
    input_image_embed_size = 768  # Image embedding size (for Vision Transformer)
    input_text_embed_size = 768  # Text embedding size (for T5 model)
    vit = 'google/vit-base-patch32-224-in21k'  # Pretrained ViT model to use
    hidden_size = 768  # Hidden size for the transformer models
    num_heads = 12  # Number of attention heads in the transformer
    mlp_ratio = 4  # MLP ratio in the transformer
    drop_rate = 0.1  # Dropout rate for regularization

    # Optimizer settings
    optim_type = "adamw"  # Type of optimizer to use
    learning_rate = 1e-5  # Learning rate
    weight_decay = 0.01  # Weight decay regularization
    decay_power = 1  # Power for learning rate decay
    max_epoch = 100  # Maximum number of epochs
    max_steps = 100000  # Maximum number of training steps
    warmup_steps = 10000  # Number of warm-up steps for learning rate
    end_lr = 0  # Final learning rate after training

    # PyTorch Lightning Trainer settings
    resume_from = None  # Path to resume the training from a checkpoint
    fast_dev_run = False  # Whether to run a quick development cycle
    val_check_interval = 1.0  # Validation check interval (after every epoch)
    test_only = False  # Whether to only run testing (skip training)
    get_recall_metric = False  # Whether to compute recall as a metric during training

    # Environment-specific parameters
    data_root = ""  # Root directory for the dataset
    log_dir = "result"  # Directory to save logs
    per_gpu_batchsize = 0  # Per-GPU batch size (will be calculated automatically)
    num_gpus = 8  # Number of GPUs to use for training
    num_nodes = 1  # Number of nodes in the distributed setup
    load_path = ""  # Path to load a pre-trained model
    num_workers = 8  # Number of workers for data loading
    precision = 32  # Precision to use for training (32-bit floating point)
    mode = "rationale"  # Mode of operation (e.g., "rationale" for specific tasks)
    out_path = ""  # Path to save output results


# Config for training the model with task-specific settings
@ex.named_config
def task_train():
    exp_name = "MEME"  # Name of the experiment
    datasets = ["meme"]  # Dataset to be used
    loss_names = _loss_names({
        "clm": 1,  # Causal Language Modeling loss
    })
    batch_size = 256  # Adjusted batch size for training
    temperature = 0.05  # Temperature setting for softmax
    max_epoch = 30  # Maximum number of epochs for this task
    max_steps = None  # Set to None for unlimited steps (based on epochs)
    warmup_steps = 0.1  # Fraction of total steps for warm-up

    vocab_size = 32128  # Vocabulary size for the tokenizer
    max_text_len = 40  # Maximum text length
    image_size = 224  # Image size for the model
    tokenizer = "bert-base-uncased"  # Tokenizer to use for this task
    train_transform_keys = ["vit"]  # Transformations to apply during training
    val_transform_keys = ["vit"]  # Transformations to apply during validation
    learning_rate = 5e-5  # Adjusted learning rate for the task
    val_check_interval = 1.0  # Interval to check validation
    hidden_size = 768  # Size of the hidden layer in transformer


# Config for Vision Transformer (ViT) with patch size 32
@ex.named_config
def vit32_base224():
    vit = "google/vit-base-patch32-224-in21k"  # Pretrained ViT model with patch size 32
    patch_size = 32  # Patch size for the transformer
    image_size = 224  # Image size for the transformer
    train_transform_keys = ["vit"]  # Transformations to apply during training
    val_transform_keys = ["vit"]  # Transformations to apply during validation
    input_image_embed_size = 768  # Image embedding size for ViT


# Config for ViT with patch size 16
@ex.named_config
def vit16_base224():
    vit = "google/vit-base-patch16-224-in21k"  # Pretrained ViT model with patch size 16
    patch_size = 16  # Patch size for the transformer
    image_size = 224  # Image size for the transformer
    train_transform_keys = ["vit"]  # Transformations to apply during training
    val_transform_keys = ["vit"]  # Transformations to apply during validation
    input_image_embed_size = 768  # Image embedding size for ViT


# Config for ViT with patch size 16 and image size 384
@ex.named_config
def vit16_base384():
    vit = "google/vit-base-patch16-384"  # Pretrained ViT model with patch size 16 and image size 384
    patch_size = 16  # Patch size for the transformer
    image_size = 384  # Image size for the transformer
    train_transform_keys = ["vit"]  # Transformations to apply during training
    val_transform_keys = ["vit"]  # Transformations to apply during validation
    input_image_embed_size = 768  # Image embedding size for ViT


# Config for CLIP model with patch size 32
@ex.named_config
def clip32_base224():
    vit = "openai/clip-vit-base-patch32"  # Pretrained CLIP model with patch size 32
    patch_size = 32  # Patch size for the transformer
    image_size = 224  # Image size for the transformer
    train_transform_keys = ["vit"]  # Transformations to apply during training
    val_transform_keys = ["vit"]  # Transformations to apply during validation
    input_image_embed_size = 768  # Image embedding size for CLIP


# Config for CLIP model with patch size 16
@ex.named_config
def clip16_base224():
    vit = "openai/clip-vit-base-patch16"  # Pretrained CLIP model with patch size 16
    patch_size = 16  # Patch size for the transformer
    image_size = 224  # Image size for the transformer
    train_transform_keys = ["vit"]  # Transformations to apply during training
    val_transform_keys = ["vit"]  # Transformations to apply during validation
    input_image_embed_size = 768  # Image embedding size for CLIP


# Text encoder using BERT tokenizer
@ex.named_config
def text_bert():
    tokenizer = "bert-base-uncased"  # BERT tokenizer
    vocab_size = 30522  # Vocabulary size for BERT
    input_text_embed_size = 768  # Text embedding size for BERT


# Text encoder using T5-small tokenizer
@ex.named_config
def text_t5_small():
    tokenizer = "google/flan-t5-small"  # T5-small tokenizer
    vocab_size = 32128  # Vocabulary size for T5
    input_text_embed_size = 512  # Text embedding size for T5-small


# Text encoder using T5-base tokenizer
@ex.named_config
def text_t5_base():
    tokenizer = "google/flan-t5-base"  # T5-base tokenizer
    vocab_size = 32128  # Vocabulary size for T5
    input_text_embed_size = 768  # Text embedding size for T5-base


# Text encoder using T5-large tokenizer
@ex.named_config
def text_t5_large():
    tokenizer = "google/flan-t5-large"  # T5-large tokenizer
    vocab_size = 32128  # Vocabulary size for T5
    input_text_embed_size = 1024  # Text embedding size for T5-large


# Config for applying random augmentations to ViT
@ex.named_config
def vit_randaug():
    train_transform_keys = ["vit_randaug"]  # Use random augmentations during training
