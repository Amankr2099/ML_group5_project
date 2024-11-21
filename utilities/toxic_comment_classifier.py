import torch
import torch.nn as nn
from transformers import DistilBertModel

class ToxicCommentClassifier(nn.Module):
    def __init__(self):
        super(ToxicCommentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Unfreeze the last few layers of DistilBERT
        for name, param in self.distilbert.named_parameters():
            if "transformer.layer.4" in name or "transformer.layer.5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.fc_1 = nn.Linear(768, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 3)  # Output for 3 classes

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, input_ids, attention_mask):
        # Pass through DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token output

        out = self.fc_1(cls_output)
        out = self.layer_norm(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc_2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc_3(out)
        return out
