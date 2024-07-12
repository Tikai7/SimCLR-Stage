import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, text):
        inputs = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']        
        outputs = self.bert(input_ids, attention_mask=attention_mask)        
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]
        return cls_embedding