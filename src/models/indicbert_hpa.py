"""
IndicBERT-HPA (Hindi-Punjabi Adapter) classifier.

A novel architecture combining IndicBERT encoder with a custom clinical adapter head
for orthopedic diagnosis classification in Hindi and Punjabi languages.

Architecture:
- IndicBERT encoder (backbone)
- Clinical Adapter Head (configurable):
  - The adapter head is built dynamically based on hidden_sizes configuration
  - Example: hidden_sizes=[512] → 768 → 512 → ReLU → Dropout → num_labels
  - Example: hidden_sizes=[768, 384] → 768 → 768 → ReLU → Dropout → 384 → ReLU → Dropout → num_labels
  - Dropout is applied after each dense layer (before final classifier)
  - End-to-end training

Configuration:
The adapter architecture is controlled via YAML config under models.{lang}.hpa:
  - hidden_sizes: List of integers defining intermediate layer sizes
  - dropout: Dropout probability (applied after each dense layer)

This allows systematic hyperparameter tuning of the adapter architecture.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
import logging

logger = logging.getLogger(__name__)


class ModelOutput(NamedTuple):
    """Output from IndicBERT-HPA forward pass."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class IndicBertHPAClassifier(nn.Module):
    """
    IndicBERT-HPA (Hindi-Punjabi Adapter) – custom clinical adapter head for orthopedic diagnosis.
    
    This model uses an IndicBERT encoder with a custom adapter head specifically designed
    for clinical text classification in Hindi and Punjabi languages.
    
    The adapter head architecture is configurable via hidden_sizes parameter:
    - hidden_sizes=[512]: Single layer adapter (768 → 512 → num_labels)
    - hidden_sizes=[768, 384]: Two-layer adapter (768 → 768 → 384 → num_labels)
    - Dropout is applied after each dense layer (before final classifier)
    
    Args:
        backbone_checkpoint: HuggingFace model identifier (e.g., "ai4bharat/indic-bert")
        num_labels: Number of classification labels
        id2label: Mapping from label IDs to label names
        label2id: Mapping from label names to label IDs
        hidden_size: Hidden size of the encoder output (default: 768 for IndicBERT)
        adapter_hidden_sizes: List of hidden sizes for intermediate dense layers (default: [512])
        adapter_dropout: Dropout probability applied after each dense layer (default: 0.2)
        class_weights: Optional class weights for loss calculation (tensor or None)
    """
    
    def __init__(
        self,
        backbone_checkpoint: str,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        hidden_size: int = 768,
        adapter_hidden_sizes: List[int] = None,
        adapter_dropout: float = 0.2,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.backbone_checkpoint = backbone_checkpoint
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.hidden_size = hidden_size
        
        # Default to single layer [512] if not specified (backward compatibility)
        if adapter_hidden_sizes is None:
            adapter_hidden_sizes = [512]
        self.adapter_hidden_sizes = adapter_hidden_sizes
        self.adapter_dropout = adapter_dropout
        
        # Store class weights (will be moved to device in forward if needed)
        self.register_buffer('class_weights', class_weights)
        
        # Load IndicBERT encoder (without classification head)
        logger.info(f"Loading IndicBERT encoder from: {backbone_checkpoint}")
        self.encoder = AutoModel.from_pretrained(backbone_checkpoint)
        
        # Get actual hidden size from encoder config if available
        encoder_config = AutoConfig.from_pretrained(backbone_checkpoint)
        if hasattr(encoder_config, 'hidden_size'):
            self.hidden_size = encoder_config.hidden_size
            logger.info(f"Encoder hidden size: {self.hidden_size}")
        
        # Build Clinical Adapter Head dynamically based on hidden_sizes
        # Architecture: encoder_output → [dense layers with ReLU + Dropout] → classifier
        adapter_layers = []
        input_size = self.hidden_size
        
        # Build intermediate dense layers
        for i, hidden_size in enumerate(adapter_hidden_sizes):
            # Dense layer
            adapter_layers.append(nn.Linear(input_size, hidden_size))
            # ReLU activation
            adapter_layers.append(nn.ReLU())
            # Dropout (applied after each dense layer)
            adapter_layers.append(nn.Dropout(p=adapter_dropout))
            input_size = hidden_size
        
        # Final classification layer
        self.adapter_head = nn.Sequential(*adapter_layers)
        self.classifier = nn.Linear(input_size, num_labels)
        
        # Build architecture description for logging
        arch_desc = f"{self.hidden_size}"
        for h in adapter_hidden_sizes:
            arch_desc += f" → {h}"
        arch_desc += f" → {num_labels}"
        
        logger.info(f"Initialized IndicBERT-HPA with {num_labels} labels")
        logger.info(f"Adapter architecture: {arch_desc}")
        logger.info(f"Dropout: {adapter_dropout} (applied after each dense layer)")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward pass through IndicBERT encoder and adapter head.
        
        Args:
            input_ids: Tokenized input sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional ground truth labels [batch_size]
        
        Returns:
            ModelOutput containing:
            - logits: Classification logits [batch_size, num_labels]
            - loss: Cross-entropy loss (if labels provided, None otherwise)
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get pooled CLS representation
        # For BERT-based models, the CLS token is the first token
        # Use the last hidden state of the CLS token (index 0)
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Pass through adapter head (dynamically built layers)
        # This applies: Dense → ReLU → Dropout (repeated for each layer in hidden_sizes)
        adapter_output = self.adapter_head(pooled_output)
        
        # Final classification layer
        logits = self.classifier(adapter_output)  # [batch_size, num_labels]
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
        
        # Return output matching HuggingFace format (with .loss and .logits attributes)
        return ModelOutput(logits=logits, loss=loss)
    
    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """
        Return optimizer parameter groups for discriminative learning rates.

        - Encoder (all transformer layers) gets a smaller LR: base_lr * 0.5
        - Adapter + classifier get the full base_lr

        This implementation is backend-agnostic and works for BERT / ALBERT-style encoders.
        """
        encoder = getattr(self, "encoder", None)

        if encoder is None:
            # Fallback: no special grouping, everything at base_lr
            return [{"params": self.parameters(), "lr": base_lr}]

        # All encoder parameters in one group (lower LR)
        encoder_params = list(encoder.parameters())

        # Adapter + classifier parameters (higher LR)
        adapter_params = list(self.adapter_head.parameters())
        classifier_params = list(self.classifier.parameters())

        return [
            {"params": encoder_params, "lr": base_lr * 0.5},
            {"params": adapter_params, "lr": base_lr},
            {"params": classifier_params, "lr": base_lr},
        ]
    
    def save_pretrained(self, save_directory: str):
        """
        Save model to directory in HuggingFace format.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save encoder
        self.encoder.save_pretrained(save_directory)
        
        # Save adapter head state dict
        adapter_state = {
            'adapter_head': self.adapter_head.state_dict(),
            'classifier': self.classifier.state_dict(),
            'num_labels': self.num_labels,
            'id2label': self.id2label,
            'label2id': self.label2id,
            'hidden_size': self.hidden_size,
            'adapter_hidden_sizes': self.adapter_hidden_sizes,  # Save list of hidden sizes
            'adapter_dropout': self.adapter_dropout
        }
        
        adapter_path = os.path.join(save_directory, 'adapter_head.pt')
        torch.save(adapter_state, adapter_path)
        
        logger.info(f"Saved IndicBERT-HPA model to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_directory: str):
        """
        Load model from directory.
        
        Args:
            model_directory: Directory containing the saved model
        
        Returns:
            Loaded IndicBertHPAClassifier instance
        """
        import os
        
        # Load encoder
        encoder = AutoModel.from_pretrained(model_directory)
        
        # Load adapter head state
        adapter_path = os.path.join(model_directory, 'adapter_head.pt')
        adapter_state = torch.load(adapter_path, map_location='cpu')
        
        # Reconstruct model (use saved hidden_sizes or default to [512] for backward compatibility)
        adapter_hidden_sizes = adapter_state.get('adapter_hidden_sizes', [adapter_state.get('adapter_hidden_size', 512)])
        
        model = cls(
            backbone_checkpoint=model_directory,  # Will use local path
            num_labels=adapter_state['num_labels'],
            id2label=adapter_state['id2label'],
            label2id=adapter_state['label2id'],
            hidden_size=adapter_state['hidden_size'],
            adapter_hidden_sizes=adapter_hidden_sizes,
            adapter_dropout=adapter_state['adapter_dropout']
        )
        
        # Load encoder
        model.encoder = encoder
        
        # Load adapter head
        model.adapter_head.load_state_dict(adapter_state['adapter_head'])
        model.classifier.load_state_dict(adapter_state['classifier'])
        
        logger.info(f"Loaded IndicBERT-HPA model from {model_directory}")
        
        return model

