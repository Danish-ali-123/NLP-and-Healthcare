import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    DistilBertForSequenceClassification,
    XLMRobertaForSequenceClassification,
    AutoModelForSequenceClassification
)
from typing import Dict, Any, Optional, Tuple
import logging
from src.models.indicbert_hpa import IndicBertHPAClassifier
'''
# Use relative import for same-package module
try:
    from .indicbert_hpa import IndicBertHPAClassifier
except ImportError:
    # Fallback for direct execution or when running as script
    from src.models.indicbert_hpa import IndicBertHPAClassifier
'''
logger = logging.getLogger(__name__)
class EnhancedAdapterHead(nn.Module):
    """
    Enhanced adapter head with intermediate feature projection.
    
    Similar to IndicBERT-HPA but for baseline models:
    - Dense layer: hidden_size → adapter_hidden_size
    - ReLU activation
    - Dropout (increased for better regularization)
    - Final classification: adapter_hidden_size → num_labels
    """
    def __init__(
        self,
        hidden_size: int,
        adapter_hidden_size: int,
        num_labels: int,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.adapter_dense = nn.Linear(hidden_size, adapter_hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(adapter_hidden_size, num_labels)
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter head."""
        adapter_output = self.adapter_dense(pooled_output)
        adapter_output = self.activation(adapter_output)
        adapter_output = self.dropout(adapter_output)
        logits = self.classifier(adapter_output)
        return logits


class MultilingualClassifier(nn.Module):
    """
    Multilingual transformer model for clinical text classification.
    
    Supports two head types:
    - 'linear': Standard HuggingFace linear head (default)
    - 'adapter': Enhanced adapter head with intermediate feature projection
    """
    
    def __init__(
        self, 
        model_name: str,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        dropout_rate: float = 0.1,
        head_type: str = 'linear',
        adapter_hidden_size: Optional[int] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.head_type = head_type
        
        # Load encoder only (without classification head)
        if "distilbert" in model_name:
            from transformers import DistilBertModel
            self.encoder = DistilBertModel.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.dim
        elif "xlm-roberta" in model_name:
            from transformers import XLMRobertaModel
            self.encoder = XLMRobertaModel.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
            # Get hidden size from config
            config = AutoConfig.from_pretrained(model_name)
            self.hidden_size = getattr(config, 'hidden_size', 768)
        
        # Build classification head
        if head_type == 'adapter':
            # Enhanced adapter head with intermediate projection
            adapter_hidden_size = adapter_hidden_size or 512
            self.classifier_head = EnhancedAdapterHead(
                hidden_size=self.hidden_size,
                adapter_hidden_size=adapter_hidden_size,
                num_labels=num_labels,
                dropout_rate=dropout_rate
            )
            logger.info(f"Using enhanced adapter head: {self.hidden_size} → {adapter_hidden_size} → {num_labels}")
        else:
            # Standard linear head
            self.classifier_head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.hidden_size, num_labels)
            )
            logger.info(f"Using standard linear head with dropout({dropout_rate})")
        
        logger.info(f"Initialized model: {model_name} with {num_labels} labels, head_type={head_type}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through encoder and classification head.
        
        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask
            labels: Optional ground truth labels
        
        Returns:
            Model output with logits and optional loss
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get pooled representation (CLS token for BERT-based, pooler_output for RoBERTa)
        if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
            pooled_output = encoder_outputs.pooler_output
        else:
            # Use CLS token (first token)
            pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification head
        logits = self.classifier_head(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # Return in HuggingFace format (SequenceClassifierOutput-like)
        # Use a simple class to mimic HuggingFace's output format
        class ModelOutput:
            def __init__(self, logits, loss=None):
                self.logits = logits
                self.loss = loss
        
        return ModelOutput(logits=logits, loss=loss)
    
    def get_param_groups(self, base_lr: float) -> list:
        """
        Return optimizer parameter groups for discriminative learning rates.
        
        - Encoder gets lower LR: base_lr * 0.5
        - Classifier head gets full base_lr
        
        This enables better fine-tuning by keeping encoder weights more stable
        while allowing the classifier to adapt quickly.
        """
        encoder_params = list(self.encoder.parameters())
        classifier_params = list(self.classifier_head.parameters())
        
        return [
            {"params": encoder_params, "lr": base_lr * 0.5},
            {"params": classifier_params, "lr": base_lr}
        ]
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save encoder
        self.encoder.save_pretrained(save_directory)
        
        # Save classifier head
        classifier_state = {
            'classifier_head': self.classifier_head.state_dict(),
            'num_labels': self.num_labels,
            'id2label': self.id2label,
            'label2id': self.label2id,
            'hidden_size': self.hidden_size,
            'head_type': self.head_type
        }
        
        import torch
        classifier_path = os.path.join(save_directory, 'classifier_head.pt')
        torch.save(classifier_state, classifier_path)
        
        logger.info(f"Saved model to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_directory: str):
        """Load model from directory."""
        import os
        import torch
        
        config = AutoConfig.from_pretrained(model_directory)
        
        # Try to load classifier head state
        classifier_path = os.path.join(model_directory, 'classifier_head.pt')
        if os.path.exists(classifier_path):
            classifier_state = torch.load(classifier_path, map_location='cpu')
            head_type = classifier_state.get('head_type', 'linear')
            adapter_hidden_size = None
            if head_type == 'adapter':
                # Infer adapter_hidden_size from saved state
                adapter_dense_state = classifier_state['classifier_head']['adapter_dense.weight']
                adapter_hidden_size = adapter_dense_state.shape[0]
        else:
            head_type = 'linear'
            adapter_hidden_size = None
        
        model = cls(
            model_name=model_directory,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id,
            head_type=head_type,
            adapter_hidden_size=adapter_hidden_size
        )
        
        # Load classifier head if available
        if os.path.exists(classifier_path):
            model.classifier_head.load_state_dict(classifier_state['classifier_head'])
        
        return model

def build_model_and_tokenizer(
    model_config: Dict[str, Any],
    label_info: Dict[str, Any],
    device: torch.device
) -> Tuple[nn.Module, Any]:
    """
    Build model and tokenizer from configuration.
    
    Supports:
    - DistilBERT (English): Linear head with dropout(0.1)
    - IndicBERT-HPA (Hindi/Punjabi): Custom adapter head (768 → 512 → ReLU → Dropout(0.2) → num_labels)
    - Other models: Standard AutoModelForSequenceClassification
    """
    
    model_name = model_config['model']['name']
    tokenizer_name = model_config['model']['tokenizer']['name']
    max_length = model_config['model']['tokenizer'].get('max_length', 256)
    head_type = model_config['model'].get('head_type', 'linear')
    
    logger.info(f"Building model: {model_name}")
    logger.info(f"Head type: {head_type}")
    logger.info(f"Using tokenizer: {tokenizer_name}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
            
    except Exception as e:
        logger.error(f"Failed to load tokenizer {tokenizer_name}: {str(e)}")
        raise
    
    # Build model based on head_type
    try:
        # Check if this is IndicBERT-HPA model
        if head_type == 'hpa' or 'indicbert_hpa' in model_name.lower():
            logger.info("Building IndicBERT-HPA model with custom adapter head")
            
            # Get HPA configuration from model config
            head_config = model_config['model'].get('head', {})
            backbone_config = model_config['model'].get('backbone', {})
            
            # Get backbone checkpoint (prefer backbone.checkpoint, fallback to model.name)
            backbone_checkpoint = backbone_config.get('checkpoint', model_name)
            
            # Get adapter dimensions
            hidden_size = head_config.get('hidden_size', 768)
            # Read adapter_hidden_sizes (list) or fallback to adapter_hidden_size (single value) for backward compatibility
            adapter_hidden_sizes = head_config.get('adapter_hidden_sizes', None)
            if adapter_hidden_sizes is None:
                # Backward compatibility: if adapter_hidden_size is provided, convert to list
                adapter_hidden_size = head_config.get('adapter_hidden_size', 512)
                adapter_hidden_sizes = [adapter_hidden_size]
            adapter_dropout = head_config.get('adapter_dropout', 0.2)
            
            logger.info(f"IndicBERT-HPA config: hidden_size={hidden_size}, "
                       f"adapter_hidden_sizes={adapter_hidden_sizes}, "
                       f"adapter_dropout={adapter_dropout}")
            
            # Build IndicBERT-HPA model
            model = IndicBertHPAClassifier(
                backbone_checkpoint=backbone_checkpoint,
                num_labels=len(label_info['labels']),
                id2label=label_info['id2label'],
                label2id=label_info['label2id'],
                hidden_size=hidden_size,
                adapter_hidden_sizes=adapter_hidden_sizes,  # Pass list of hidden sizes
                adapter_dropout=adapter_dropout,
                class_weights=None  # Can be set later if needed
            )
            
        else:
            # Standard model (DistilBERT, XLM-RoBERTa, etc.)
            logger.info(f"Building standard model: {model_name}")
            
            # Get head configuration
            head_config = model_config['model'].get('head', {})
            dropout_rate = head_config.get('dropout', 0.2)  # Increased default from 0.1 to 0.2
            adapter_hidden_size = head_config.get('adapter_hidden_size', None)
            
            # Check if enhanced adapter head is requested
            use_adapter = head_config.get('use_adapter', False) or head_type == 'adapter'
            
            if use_adapter:
                logger.info(f"Using enhanced adapter head: {model_name}")
                logger.info(f"Adapter config: dropout={dropout_rate}, adapter_hidden_size={adapter_hidden_size or 512}")
            else:
                logger.info(f"Using standard linear head with dropout({dropout_rate})")
            
            model = MultilingualClassifier(
                model_name=model_name,
                num_labels=len(label_info['labels']),
                id2label=label_info['id2label'],
                label2id=label_info['label2id'],
                dropout_rate=dropout_rate,
                head_type='adapter' if use_adapter else 'linear',
                adapter_hidden_size=adapter_hidden_size
            )
        
        # Move model to device
        model.to(device)
        
    except Exception as e:
        logger.error(f"Failed to build model {model_name}: {str(e)}")
        raise
    
    logger.info(f"Model built successfully. Trainable parameters: {count_parameters(model):,}")
    
    return model, tokenizer

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model: nn.Module) -> str:
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return f"{size_mb:.2f} MB"

# Model registry for easy access
MODEL_REGISTRY = {
    # Canonical backbones used in this project
    'distilbert': 'distilbert-base-multilingual-cased',
    'indicbert': 'ai4bharat/indic-bert',
    'xlm_roberta': 'xlm-roberta-base',
    'mdeberta': 'microsoft/mdeberta-v3-base',
    # Backwards-compatible alias for older configs/experiments
    'indicbert_v1': 'ai4bharat/indic-bert',
}

def get_model_names() -> list:
    """Get available model names."""
    return list(MODEL_REGISTRY.keys())