import os
import numpy as np
import pandas as pd
from model.bert import BertModel
from torch.utils.data import Dataset, DataLoader
import torch

class DataProcessForSentence(Dataset):
    def __init__(self, bert_tokenizer, df, max_seq_len=50, return_player=False):
        super(DataProcessForSentence, self).__init__()
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_seq_len
        self.return_player = return_player

        self.input_ids, self.attention_mask, self.token_type_ids, self.labels, self.sentences, self.selected_inds = self.get_input(
            df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx], \
        self.sentences[idx]

    # convert dataframe to tensor
    def get_input(self, df):
        sentences = df['sentence'].values
        labels = df['label'].values
        tokens_seq = list(map(self.bert_tokenizer.tokenize, sentences))

        if self.return_player:
            selected_inds = []
            tokens = df['token'].values
            for i in range(len(tokens)):
                token = eval(tokens[i])
                if len(token) > 10:
                    token = token[:10]
                selected_ind = np.where(np.isin(tokens_seq[i], token))[0]
                if len(selected_ind) > 10:
                    selected_ind = selected_ind[:10]
                selected_inds.append(selected_ind + 1)
            selected_inds = np.array(selected_inds)
        else:
            selected_inds = None
        # Get fixed-length sequence and its mask
        result = list(map(self.trunate_and_pad, tokens_seq))
        input_ids = [i[0] for i in result]
        attention_mask = [i[1] for i in result]
        token_type_ids = [i[2] for i in result]
        return (
            torch.Tensor(input_ids).type(torch.long),
            torch.Tensor(attention_mask).type(torch.long),
            torch.Tensor(token_type_ids).type(torch.long),
            torch.Tensor(labels).type(torch.long),
            sentences,
            selected_inds
        )

    def trunate_and_pad(self, tokens_seq):

        # Concat '[CLS]' at the beginning
        tokens_seq = ['[CLS]'] + tokens_seq
        # Truncate sequences of which the lengths exceed the max_seq_len
        if len(tokens_seq) > self.max_seq_len:
            tokens_seq = tokens_seq[0:self.max_seq_len]
        # Generate padding
        padding = [0] * (self.max_seq_len - len(tokens_seq))
        # Convert tokens_seq to token_ids
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_seq)
        input_ids += padding
        # Create attention_mask
        attention_mask = [1] * len(tokens_seq) + padding
        # Create token_type_ids
        token_type_ids = [0] * (self.max_seq_len)

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len

        return input_ids, attention_mask, token_type_ids


def get_data(tokenizer, batch_size=64, folder_path='./data/SST-2', max_seq_len=50):
    train_df = pd.read_csv(os.path.join(folder_path, 'train.tsv'),
                           sep='\t')
    dev_df = pd.read_csv(os.path.join(folder_path, 'dev.tsv'), sep='\t')
    test_df = pd.read_csv(os.path.join(folder_path, 'test.tsv'), sep='\t', header=None, names=['sentence', 'label'])
    train_data = DataProcessForSentence(tokenizer, train_df, max_seq_len=max_seq_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    dev_data = DataProcessForSentence(tokenizer, dev_df, max_seq_len=max_seq_len)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    test_data = DataProcessForSentence(tokenizer, test_df, max_seq_len=max_seq_len)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    return train_loader, dev_loader, test_loader


def get_model_data(args, device, model_dir, finetuned_model_dir, finetuned_model_path):
    BATCH_SIZE = 10000
    MAX_SEQ_LEN = 50
    bertmodel = BertModel(requires_grad=False, device=device, model_type=model_dir)
    tokenizer = bertmodel.tokenizer
    train_loader, dev_loader, test_loader = get_data(tokenizer, batch_size=BATCH_SIZE,
                                                         max_seq_len=MAX_SEQ_LEN)
    checkpoint = torch.load(os.path.join(finetuned_model_dir, finetuned_model_path), map_location=args.device)
    print("\nLoad model from path:", os.path.join(finetuned_model_dir, finetuned_model_path))
    bertmodel.load_state_dict(checkpoint["model"], False)
    model = bertmodel.to(device)
    model.eval()
    return model, train_loader, test_loader













