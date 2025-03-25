import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import unicodedata
from typing import Union
import numpy as np
import os

def get_eval_model(model_path: str, device, unique_labels):
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(unique_labels)).to(device)
    model_fp16 = AutoModelForSequenceClassification.from_pretrained(model_path+"_fp16", num_labels=len(unique_labels)).to(device).half()

    model_fp32.eval()
    model_fp16.eval()

    return model_fp32, model_fp16

def normalize_unicode_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    unicode_single_hangul_dict = {'ᄀ': 'ㄱ', 'ᄂ': 'ㄴ', 'ᄃ': 'ㄷ', 'ᄅ': 'ㄹ', 'ᄆ': 'ㅁ', 'ᄇ': 'ㅂ', 'ᄉ': 'ㅅ', 'ᄋ': 'ㅇ', 'ᄌ': 'ㅈ', 'ᄎ': 'ㅊ', 'ᄏ': 'ㅋ', 'ᄐ': 'ㅌ', 'ᄑ': 'ㅍ', 'ᄒ': 'ㅎ', 'ᄍ': 'ㅉ', 'ᄄ': 'ㄸ', 'ᄁ': 'ㄲ', 'ᄊ': 'ㅆ', 'ᅡ': 'ㅏ', 'ᅣ': 'ㅑ', 'ᅥ': 'ㅓ', 'ᅧ': 'ㅕ', 'ᅩ': 'ㅗ', 'ᅭ': 'ㅛ', 'ᅮ': 'ㅜ', 'ᅲ': 'ㅠ', 'ᅳ': 'ㅡ', 'ᅵ': 'ㅣ', 'ᅢ': 'ㅐ', 'ᅦ': 'ㅔ', 'ᅴ': 'ㅢ', 'ᆪ': 'ㄱㅅ', 'ᆬ': 'ㄴㅈ', 'ᆭ': 'ㄴㅎ', 'ᆲ': 'ㄹㅂ', 'ᆰ': 'ㄹㄱ', 'ᆳ': 'ㄹㅅ', 'ᆱ': 'ㄹㅁ', 'ᄚ': 'ㄹㅎ', 'ᆴ': 'ㄹㅌ', 'ᆵ': 'ㄹㅍ', 'ᄡ': 'ㅂㅅ', 'ᄈ': 'ㅂㅂ'}
    normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
    return ''.join(unicode_single_hangul_dict[ch] if ch in unicode_single_hangul_dict else ch for ch in normalized)

def normalize_tlettak_font(text: str, 
                           space_pattern: Union[str, re.Pattern] = r'[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]+[\s!?@.,❤]*', 
                           search_pattern: Union[str, re.Pattern] = r'(\b\w\b)([\s!?@.,❤]+)(\b\w\b)',
                           ) -> str:
    if isinstance(space_pattern, str):
        space_pattern = re.compile(space_pattern)
    if isinstance(search_pattern, str):
        search_pattern = re.compile(search_pattern)

    result = []
    sub = []
    pos = 0
    
    while pos < len(text):
        space_matched = space_pattern.match(text, pos)
        search_matched = search_pattern.match(text, pos)

        if search_matched:
            sub.extend([search_matched.group(1), search_matched.group(3)])
            pos = search_matched.end() - 1
        elif space_matched:
            s_end = space_matched.end()
            result.append(''.join(sub[::2]) + text[pos:s_end].strip())
            pos = s_end
            sub.clear()
        else:   # 둘 다 매칭 실패인 경우 뒷문장 전부를 붙여씀
            result.append(text[pos:])
            break
    return ' ' .join(result)

def replace_nickname_data(df: pd.DataFrame):
    space_pattern = r'[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9:]+[\s!?@.,❤]*'
    pattern = r"(\w)([\s!?@.,❤]+)(\b\w\b)"

    # prefix, subfix 제거
    df['nickname'] = df['nickname']\
        .str.strip()\
        .str.replace('@', '')\
        .str.replace(r'-[a-zA-Z0-9]+(?=\s|$)', '', regex=True)
    # 특수 기호 제거
    df['nickname'] = df['nickname']\
        .str.replace(r'[-._]', '', regex=True)
    # 영어, 한글, 숫자가 아닌 경우 기본 닉네임 처리
    df['nickname'] = df['nickname']\
        .str.replace(r'[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ0-9]+', '[DEFAULT_NICK]', regex=True)
    
    with open('./tokens/emojis.txt', 'r', encoding='utf-8') as f:
        emojis = [line.strip() for line in f.readlines()]

    emoji_pattern = '|'.join(map(re.escape, emojis))
    df['comment'].str.replace(emoji_pattern, '[TEXT_EMOJI]', regex=True)
    
    # 유니코드 문장부호 수정
    df['comment'] = df['comment']\
        .str.replace(r'[ㆍ·・•]', '.', regex=True)\
        .str.replace(r'[ᆢ…]+', '..', regex=True)\
        .str.replace(r'[‘’]+', "'", regex=True)\
        .str.replace(r'[“”]+', '"', regex=True)\
        .str.replace(r'[\u0020\u200b\u2002\u2003\u2007\u2008\u200c\u200d]+', ' ', regex=True)\
        .str.replace(r'[\U0001F3FB-\U0001F3FF\uFE0F]', '', regex=True)
    # 유니코드 꾸밈 문자(결합 문자) 제거
    df['comment'] = df['comment'].str.replace(r'\*+', '', regex=True)
    df['comment'] = df['comment'].apply(lambda x: normalize_unicode_text(x) if isinstance(x, str) else x)
    # special token 파싱
    df['comment'] = df['comment']\
        .str.replace(r'https?:\/\/(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ-]+\.[a-zA-Z]{2,}(?:\/[^?\s]*)?(?:\?[^\s]*)?', '[URL]', regex=True)\
        .str.replace(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', regex=True)\
    # 한글자 + 부호 + 한글자 패턴 처리
    df['comment'] = df['comment'].apply(lambda x: normalize_tlettak_font(x, space_pattern, pattern) if isinstance(x, str) else x)
    # special token 파싱
    df['comment'] = df['comment']\
        .str.replace(r'@{1,2}[A-Za-z0-9가-힣\_\-\.]+', '[TAG]', regex=True)\
        .str.replace(r'#[A-Za-z0-9ㄱ-ㅎㅏ-ㅣ가-힣\_\-\.]+', '[HASH_TAG]', regex=True)\
        .str.replace('¡', '!').str.replace('¿', '?')\
        .str.replace(r'([👇✋👍])', '[THUMB]', regex=True)\
        .str.replace(r'([➡⬇↗↘↖↙→←↑↓⇒]|[\-\=]+>|<[\-\=]+)', '[ARROW]', regex=True)\
        .str.replace(r'[💚💛🩷🩶💗💖❤🩵🖤💘♡♥🧡🔥💕️🤍💜🤎💙]', '[HEART]', regex=True)\
        .str.replace(r'🎉', '[CONGRAT]', regex=True)
    # 쓸데없이 많은 문장부호 제거
    df['comment'] = df['comment']\
        .str.replace(r'([^\s])[.,](?=\S)', r'\1', regex=True)\
        .str.replace(r'([.,?!^]+)', r' \1 ', regex=True)\
        .str.replace(r'\s+([.,?!^]+)', r'\1', regex=True)\
        .str.replace(r'\s{2,}', ' ', regex=True)
    # timestamp 처리
    to_replace = '[TIMESTAMP]'
    df['comment'] = df['comment']\
        .str.replace(r'\d+:(?:\d+:?)?\d+', to_replace, regex=True)
    # 밈 처리
    # df['comment'] = df['comment']\
    #     .str.replace(r'(?i)chill', '칠', regex=True)
    # 한글, 영어가 아닌 경우 처리
    df['comment'] = df['comment']\
        .str.replace(r'[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ♡♥\!\?\@\#\$\%\^\&\*\(\)\-\_\=\+\\\~\,\.\/\<\>\[\]\{\}\;\:\'\"\s]', '', regex=True)
    # 2개 이상 연속된 문자 처리
    df['comment'] = df['comment']\
        .str.replace(r'(.)\1{2,}', r'\1\1', regex=True)
    # 빈 문자열의 경우 empty 처리
    df['comment'] = df['comment'].str.strip()
    df['comment'] = df['comment'].fillna('[EMPTY]')

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts.tolist(), 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)

def prepare_data(datas, labels, tokenizer, max_length, batch_size=32):
    """
    데이터 준비 및 데이터로더 생성
    """
    # 데이터 분할
    dataset = CustomDataset(datas, labels, tokenizer, max_length)
    
    # 데이터로더 생성
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

def test_fp_precision_eval(model_type, dataloader_fp32, dataloader_fp16, device, unique_labels):
    import time
    """
    FP32와 FP16 모델의 정확도를 평가하는 함수
    """
    model_fp32, model_fp16 = get_eval_model(os.path.join(os.path.expanduser('~'), 'youtube-comment-colab', 'model', f'{model_type}_model'), device, unique_labels)
    
    fp32_correct = 0
    fp32_total = 0
    fp32_total_loss = 0.0
    fp32_time = 0
    
    fp16_correct = 0
    fp16_total = 0
    fp16_total_loss = 0.0
    fp16_time = 0

    start = 0
    
    fp32_start = time.perf_counter_ns()
    with torch.no_grad():
        for batch in dataloader_fp32:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            # FP32 모델 평가
            with torch.no_grad():
                # inputs_fp32 = {k: v.float() for k, v in inputs.items()}
                start = time.perf_counter_ns()
                outputs_fp32 = model_fp32(**inputs)
                # loss_fp32 = outputs_fp32.loss()
            
                predictions_fp32 = torch.argmax(outputs_fp32.logits, dim=-1)
                fp32_time += time.perf_counter_ns() - start
            fp32_total += labels.size(0)
            fp32_correct += (predictions_fp32 == labels).sum().item()
            # fp32_total_loss += loss_fp32.item()
    print('\t\tdataloader fp32 - ', (time.perf_counter_ns() - fp32_start) / 1e9)

    fp16_start = time.perf_counter_ns()
    with torch.no_grad():
        for batch in dataloader_fp16:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            with torch.no_grad():
                # FP16 모델 평가
                # inputs_fp16 = {k: v.half() for k, v in inputs.items()}
                start = time.perf_counter_ns()
                outputs_fp16 = model_fp16(**inputs)
                # loss_fp16 = outputs_fp16.loss()
            
                predictions_fp16 = torch.argmax(outputs_fp16.logits, dim=-1)
                fp16_time += time.perf_counter_ns() - start
            fp16_total += labels.size(0)
            fp16_correct += (predictions_fp16 == labels).sum().item()
            # fp16_total_loss += loss_fp16.item()
    print('\t\tdataloader fp16 - ', (time.perf_counter_ns() - fp16_start) / 1e9)
    
    # 정확도 및 손실 계산
    fp32_accuracy = fp32_correct / fp32_total
    fp16_accuracy = fp16_correct / fp16_total
    
    # fp32_avg_loss = fp32_total_loss / len(test_dataloader)
    # fp16_avg_loss = fp16_total_loss / len(test_dataloader)
    
    return {
        'fp32_accuracy': fp32_accuracy,
        'fp16_accuracy': fp16_accuracy,
        # 'fp32_avg_loss': fp32_avg_loss,
        # 'fp16_avg_loss': fp16_avg_loss,
        'fp32_correct': fp32_correct,
        'fp16_correct': fp16_correct,
        'fp32_total': fp32_total,
        'fp16_total': fp16_total,
        'fp32_time': fp32_time,
        'fp16_time': fp16_time,
    }

def print_fp_precision_evaluation(eval_results, model_type: str):
    """
    FP32와 FP16 모델 평가 결과 출력
    """
    print(f"{model_type} 모델 평가 결과:")
    
    print(f"FP32 실행 시간: {eval_results['fp32_time'] / 1e9}")
    print(f"FP16 실행 시간: {eval_results['fp16_time'] / 1e9}")

    print(f"\nFP32 정확도: {eval_results['fp32_accuracy'] * 100:.2f}%")
    print(f"FP16 정확도: {eval_results['fp16_accuracy'] * 100:.2f}%")
    print(f"정확도 차이: {abs(eval_results['fp32_accuracy'] - eval_results['fp16_accuracy']) * 100:.2f}%")
    
    # print(f"\nFP32 평균 손실: {eval_results['fp32_avg_loss']:.4f}")
    # print(f"FP16 평균 손실: {eval_results['fp16_avg_loss']:.4f}")
    
    print(f"\nFP32 정확한 예측: {eval_results['fp32_correct']} / {eval_results['fp32_total']}")
    print(f"FP16 정확한 예측: {eval_results['fp16_correct']} / {eval_results['fp16_total']}")

def eval(df: pd.DataFrame, model_type: str, tokenizer):
    import time
    print(f'==================== {model_type} 모델 평가 로그 ====================')
    start = time.perf_counter_ns()
    columns = df[[f'{model_type}', f'{model_type}_class']].dropna().reset_index(drop=True)
    label_map = {label: idx for idx, label in enumerate(columns[f'{model_type}_class'].unique())}
    labels = columns[f'{model_type}_class'].map(label_map)

    max_token_length = 40 if model_type == 'nickname' else 256

    datas = columns[model_type]

    batch_size = 16

    pd_start = time.perf_counter_ns()
    test_loader_fp32 = prepare_data(
        datas,
        labels,
        tokenizer,
        max_token_length,
        batch_size
    )
    print('\tprepare fp32 data time: ', (time.perf_counter_ns() - pd_start) / 1e9)

    pd_start = time.perf_counter_ns()
    test_loader_fp16 = prepare_data(
        datas,
        labels,
        tokenizer,
        max_token_length,
        batch_size * 2
    )
    print('\tprepare fp16 data time: ', (time.perf_counter_ns() - pd_start) / 1e9)

    eval_start = time.perf_counter_ns()
    eval_results = test_fp_precision_eval(
        model_type=model_type,
        dataloader_fp32=test_loader_fp32, 
        dataloader_fp16=test_loader_fp16, 
        device=device,
        unique_labels=columns[f'{model_type}_class'].unique()
    )
    print('\ttotal eval time: ', (time.perf_counter_ns() - eval_start) / 1e9)

    print_fp_precision_evaluation(eval_results, model_type)
    print('전체 추론 시간: ', (time.perf_counter_ns() - start) / 1e9)

from s3_helper import S3Helper
helper = S3Helper('/home/sh/youtube-comment-colab', 'youtube-comment-predict')
print('download dataset.csv...')
helper.download(['dataset.csv'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("./model/tokenizer")

df = pd.read_csv(("./model/dataset.csv"), usecols=["nickname", "comment", "nickname_class", "comment_class"])
df['comment'] = df['comment'].str.replace(r'\\', ',', regex=True) 
replace_nickname_data(df)

eval(df, 'nickname', tokenizer)
eval(df, 'comment', tokenizer)