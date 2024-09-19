import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SimpleSPLADEPlusPlus(torch.nn.Module):
    def __init__(self, model_name="Luyu/co-condenser-marco"):
        super(SimpleSPLADEPlusPlus, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = F.relu(output.logits)  # Apply ReLU to remove negative values
        log_logits = torch.log1p(logits)  # Compute log(1 + x)
        sparse_repr = log_logits.max(dim=1).values  # Compute max along the sequence length
        return sparse_repr
