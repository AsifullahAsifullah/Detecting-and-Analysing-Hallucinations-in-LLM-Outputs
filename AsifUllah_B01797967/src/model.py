# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)
        return logits

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        return outputs.attentions


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = 2,
    dropout_rate: float = 0.1,
):
    from transformers import AutoTokenizer, DebertaV2Tokenizer

    if "deberta-v3" in model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    elif "deberta" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = TransformerClassifier(model_name, num_labels, dropout_rate)
    return model, tokenizer


def save_model(model: TransformerClassifier, path: str) -> None:
    import os

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": model.model_name,
        "num_labels": model.num_labels,
    }, path)
    print(f"  Model saved → {path}")


def load_model(path: str, dropout_rate: float = 0.1) -> TransformerClassifier:
    checkpoint = torch.load(path, map_location="cpu")
    model = TransformerClassifier(
        model_name=checkpoint["model_name"],
        num_labels=checkpoint["num_labels"],
        dropout_rate=dropout_rate,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model