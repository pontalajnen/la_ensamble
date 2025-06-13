# code taken and adopted from https://aleximmer.com/Laplace/huggingface_example/

# from collections.abc import MutableMapping
from transformers import (
    # AutoModelForSequenceClassification,
    # AutoTokenizer,
    # DataCollatorWithPadding,
    BertForSequenceClassification
)


class BERT(BertForSequenceClassification):
    """
    Load BERT model from HuggingFace.
    model (str): the model name to be loaded
    num_labels (int): the number of labels that the classification task has
    device (str): the device that the data will be on; the model will be put to that device
    """
    def __init__(self, device, model: str, num_labels: int):
        config = BertForSequenceClassification.from_pretrained(model).config
        config.num_labels = num_labels
        super().__init__(config)
        self.to(device)

    def forward(self, batch):
        """
        batch: a dict containing keys like 'input_ids', 'attention_mask', 'labels'
        """
        device = next(self.parameters()).device
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)
        return super().forward(
            input_ids=batch.get("input_ids").to(device),
            attention_mask=batch.get("attention_mask").to(device),
            token_type_ids=token_type_ids,
            labels=labels
        ).logits
