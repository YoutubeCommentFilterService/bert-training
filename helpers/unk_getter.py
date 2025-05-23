from transformers import AutoTokenizer
import pandas as pd
from text_preprocessing import TextNormalizator
import re
import os


tokenizer = AutoTokenizer.from_pretrained('../model/tokenizer')
UNKNOWN_TOKEN = tokenizer.unk_token

datas = pd.read_csv('../model/dataset.csv')
datas['comment'].str.replace('\\', ',', regex=False)
# datas = pd.read_csv('./testset.csv')

normalizator = TextNormalizator()
normalizator.run_text_preprocessing(datas)

if not os.path.exists('./unk_data'):
    os.makedirs('./unk_data')


with open('./unk_data/datas', 'w', encoding='utf-8') as f:
    f.write('')
with open('./unk_data/unk_datas', 'w', encoding='utf-8') as f:
    f.write('')
with open('./unk_data/unk_datas_char', 'w', encoding='utf-8') as f:
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

        with open('./unk_data/unk_datas', 'a+', encoding='utf-8') as f:
            f.write(f'{len(data)} - {data}\n\t{len(tokens)} - {tokens}\n')

        with open('./unk_data/unk_datas_char', 'a+', encoding='utf-8') as f:
            f.write(f'{data}\n\t{processing_text}\n')
    with open('./unk_data/datas', 'a+', encoding='utf-8') as f:
        f.write(f'{data}\n\t{tokens}\n')