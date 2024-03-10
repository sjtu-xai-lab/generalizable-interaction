from torch import nn
from transformers import BertForSequenceClassification, AutoTokenizer

class BertModel(nn.Module):
    def __init__(self, requires_grad=True, device = 'cuda:0', model_type = "bert_base-uncased"):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
                model_type,
                num_labels =2,
                output_attentions = False,
                output_hidden_states = False,)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = device
        for param in self.bert.parameters():
            param.requires_grad = requires_grad

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask=batch_seq_masks,
                                 token_type_ids = batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
    




