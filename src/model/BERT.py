import torch 
import torch.nn as nn
from transformers import BertModel

class BertEncoder(nn.Module):
    """
        BERT encoder for extracting CLS embedding
    """
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']        
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.bert(input_ids, attention_mask=attention_mask)        
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]    
        return cls_embedding