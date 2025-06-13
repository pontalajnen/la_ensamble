# from collections.abc import MutableMapping
from transformers import (
    AutoModelForSequenceClassification,
    # AutoTokenizer,
    # DataCollatorWithPadding,
    # BertForSequenceClassification
)
from transformers import AutoConfig
import torch.nn as nn


class HF(nn.Module):
    """
    Load models from HuggingFace. This works for all models that can be loaded via AutoModelForSequenceClassification
    model (str): the model name to be loaded
    num_labels (int): the number of labels that the classification task has
    device (str): the device that the data will be on; the model will be put to that device
    """
    def __init__(self, model: str, num_labels: int, device: str):
        super().__init__()
        config = AutoConfig.from_pretrained(model, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, config=config)
        self.model.to(device)

    def forward(self, batch):
        device = next(self.model.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids')
        labels = batch.get('labels')

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return outputs.logits
