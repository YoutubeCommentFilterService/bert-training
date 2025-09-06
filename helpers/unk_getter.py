from transformers import AutoTokenizer
import pandas as pd
from text_preprocessing import TextNormalizator
import re
import os
import time

tokenizer = AutoTokenizer.from_pretrained('../model/tokenizer')
UNKNOWN_TOKEN = tokenizer.unk_token

datas = pd.read_csv('../model/dataset.csv')
datas['comment'] = (
    datas['comment']
        .str.replace('\\', ',', regex=False)
        .str.strip()
)
origin_datas = datas.copy(True)
# datas = pd.read_csv('./testset.csv')

normalizator = TextNormalizator()
print('preprocessing started')
start = time.time()
normalizator.run_text_preprocessing(datas)
print('preprocessing ended - ', time.time() - start)

os.makedirs('./unk_data', exist_ok=True)

UNK_COMMENT_FILE = './unk_data/datas.txt'
UNK_COMMENT_CHAR_FILE = './unk_data/unk_datas_char.txt'
UNK_NICK_FILE = './unk_data/datas_nick.txt'

datas['comment'] = datas['comment'].str.strip()

print('comment unk getter started')
with open(UNK_COMMENT_FILE, 'w', encoding='utf-8') as f:
    f.write('')
with open(UNK_COMMENT_CHAR_FILE, 'w', encoding='utf-8') as f:
    f.write('')
start = time.time()
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

        with open(UNK_COMMENT_CHAR_FILE, 'a+', encoding='utf-8') as f:
            f.write(f'{write_text}\n\t{processing_text}\n\n')

    with open(UNK_COMMENT_FILE, 'a+', encoding='utf-8') as f:
        f.write(f'{write_text}\n\t{tokens}\n')
print('comment unk getter ended - ', time.time() - start)

print('nickname unk getter started')
with open(UNK_NICK_FILE, 'w', encoding='utf-8') as f:
    f.write('')
origin_nickname = origin_datas['nickname'].dropna().reset_index(drop=True)
start = time.time()
for idx, data in enumerate(datas['nickname'].dropna().reset_index(drop=True)):
    try:
        tokens = tokenizer.tokenize(data)
        write_text = data if origin_nickname.iloc[idx] == data else f'{origin_nickname.iloc[idx]}\n{data}'
        with open(UNK_NICK_FILE, 'a+', encoding='utf-8') as f:
            f.write(f'{write_text}\n\t{tokens}\n')
    except:
        print(idx, data)
print('nickname unk getter ended - ', time.time() - start)