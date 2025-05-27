from transformers import AutoTokenizer
import pandas as pd
from text_preprocessing import TextNormalizator
import re
import os


tokenizer = AutoTokenizer.from_pretrained('../model/tokenizer')
UNKNOWN_TOKEN = tokenizer.unk_token

datas = pd.read_csv('../model/dataset.csv')
datas['comment'] = datas['comment'].str.replace('\\', ',', regex=False)
origin_datas = datas.copy(True)
# datas = pd.read_csv('./testset.csv')

normalizator = TextNormalizator()
normalizator.run_text_preprocessing(datas)

if not os.path.exists('./unk_data'):
    os.makedirs('./unk_data')


with open('./unk_data/datas', 'w', encoding='utf-8') as f:
    f.write('')
with open('./unk_data/unk_datas_char', 'w', encoding='utf-8') as f:
    f.write('')

for idx, data in enumerate(datas['comment']):
    tokens = tokenizer.tokenize(data)
    write_text = data if origin_datas["comment"].iloc[idx] == data else f'{origin_datas["comment"].iloc[idx]}\n{data}'
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

        with open('./unk_data/unk_datas_char', 'a+', encoding='utf-8') as f:
            f.write(f'{write_text}\n\t{processing_text}\n\n')

    with open('./unk_data/datas', 'a+', encoding='utf-8') as f:
        f.write(f'{write_text}\n\t{tokens}\n')