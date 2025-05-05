from transformers import AutoTokenizer
import pandas as pd
from helpers.text_preprocessing import run_text_preprocessing
import re

UNKNOWN_TOKEN = '[UNK]'

tokenizer = AutoTokenizer.from_pretrained('../model/tokenizer')

datas = pd.read_csv('../model/dataset.csv')

run_text_preprocessing(datas, '../tokens/emojis.txt')

with open('./unk_datas', 'w', encoding='utf-8') as f:
    f.write('')
with open('./unk_datas_char', 'w', encoding='utf-8') as f:
    f.write('')

for data in datas['comment']:
    tokens = tokenizer.tokenize(data)
    if UNKNOWN_TOKEN in tokens:
        current_word = ''
        unk_tokens = []
        processing_text: str = data
        for token in tokens:
            if token.startswith('##'):
                current_word += token[2:]
            elif token != UNKNOWN_TOKEN:
                if current_word != '':
                    processing_text = processing_text.replace(current_word, '', 1)
                current_word = token
        processing_text = processing_text.replace(current_word, '', 1)
        processing_text = re.sub(r'\s+', ' ', processing_text).strip()

        with open('./unk_datas', 'a+', encoding='utf-8') as f:
            f.write(f'{len(data)} - {data}\n\t{len(tokens)} - {tokens}\n')

        with open('./unk_datas_char', 'a+', encoding='utf-8') as f:
            f.write(f'{data}\n\t{processing_text}\n')