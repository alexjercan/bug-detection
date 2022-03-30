import json
import torch

import pandas as pd
import torch.nn as nn

from sklearn.model_selection import train_test_split
from model import Seq2Seq
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from tqdm import tqdm
from itertools import cycle


BEAM_SIZE = 5
MAX_TARGET_LEN = 50  


def tokenize_and_align_labels(tokenizer, X, y):
    tokenized_inputs = tokenizer(X, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt")
    tokenized_outputs = tokenizer(y, padding=True, truncation=True, return_tensors="pt")

    return {
        "source_ids": tokenized_inputs["input_ids"], 
        "target_ids": tokenized_outputs["input_ids"],
        "source_mask": tokenized_inputs["attention_mask"],
        "target_mask": tokenized_outputs["attention_mask"]
    }


class CodeNetPyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.data.items()}
        return item["source_ids"], item["source_mask"], item["target_ids"], item["target_mask"]

    def __len__(self):
        return len(self.data['source_ids'])


if __name__ == "__main__":
    config = RobertaConfig.from_pretrained('microsoft/codebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=BEAM_SIZE,max_length=MAX_TARGET_LEN,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

    with open('codenetpy.json') as f:
        data = json.load(f)

    data_df = pd.DataFrame(data).iloc[:5]
    data_df = data_df[data_df['error_class'].isin(['SyntaxError', 'NameError', 'TypeError', 'ValueError', 'IndexError', 
                                                'AttributeError', 'EOFError', 'TLEError',  'ImportError', 'ZeroDivisionError'])]

    X = data_df['original_tokens'].tolist()
    y = data_df['error_class_extra'].tolist()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_data = tokenize_and_align_labels(tokenizer, X_train, y_train)
    val_data = tokenize_and_align_labels(tokenizer, X_val, y_val)

    train_dataset = CodeNetPyDataset(train_data)
    val_dataset = CodeNetPyDataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=1e-8)

    num_train_optimization_steps=len(train_dataloader) * 1000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=num_train_optimization_steps)

    bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
    train_dataloader=cycle(train_dataloader)

    model.train()

    train_loss = 0.0
    for step in bar:
        batch = next(train_dataloader)
        source_ids, source_mask, target_ids, target_mask = batch

        loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        avg_loss = train_loss / (step + 1)
        bar.set_description("Avg Train Loss: {:.4f}".format(avg_loss))