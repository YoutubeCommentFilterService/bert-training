import os
from pathlib import Path
from typing import Any, List, Optional, Set, Union
from tqdm import tqdm
from dotenv import load_dotenv
import argparse
from collections import Counter

import pandas as pd

import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, AutoConfig

from helpers.text_preprocessing import run_text_preprocessing

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, label_index_map):
        self.encodings = encodings
        self.labels = [label_index_map[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() if isinstance(val[idx], torch.Tensor) else val[idx] 
            for key, val in self.encodings.items()
        }
        item['labels'] = self.labels[idx]
        return item
    
class TokenizeManager():
    def __init__(self, train_model_name:str="klue/bert-base", tokenizer_path:str="./model/tokenizer"):
        self.train_model_name = train_model_name
        self.tokenizer_path = tokenizer_path

    def is_valid_tokenizer_dir(self, path: str) -> bool:
        return os.path.isdir(path) and any(os.scandir(path))

    def update_tokenizer(self):
        def load_tokens_set(path: str) -> Set[str]:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return set([line.strip() for line in f if line.strip()])
            except FileNotFoundError:
                return set()
            
        def get_unique_token(path: str, existing_tokens_list: Optional[List[str]]) -> List[str]:
            more_tokens_set = load_tokens_set(path)
            existing_tokens_set = set(existing_tokens_list or [])
            return list(more_tokens_set - existing_tokens_set)
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path if self.is_valid_tokenizer_dir(self.tokenizer_path) else self.train_model_name
        )

        unique_special_tokens = get_unique_token('./tokens/special_tokens.txt', tokenizer.additional_special_tokens)
        unique_common_tokens = get_unique_token('./tokens/common_tokens.txt', tokenizer.get_vocab().keys())

        if unique_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': unique_special_tokens})
        if unique_common_tokens:
            tokenizer.add_tokens(unique_common_tokens)
        self._tokenizer = tokenizer

    def save_tokenizer(self):
        self._tokenizer.save_pretrained(self.tokenizer_path)

class TrainModel():
    def __init__(self, data: pd.DataFrame, model_type: str, save_path:str, train_model_name:str = "klue/bert-base", batch_size:int = 16, epoches: int = 10):
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)

        self.model_type = model_type
        self.max_token_length = 256 if model_type == 'comment' else 40
        self.epoches = epoches
        self.batch_size = batch_size
        self.model_path = f"{save_path}/{model_type}_model"
        self._tokenizer = AutoTokenizer.from_pretrained(f"{save_path}/tokenizer")

        self._assign_pandas_data(data)
        self._load_model(self.model_path, train_model_name)

    def _assign_pandas_data(self, data: pd.DataFrame):
        columns = data[[f'{self.model_type}_class', f'{self.model_type}']]
        columns = columns.dropna(how='any')
        data_pd = columns[f'{self.model_type}']

        label_pd = columns[f'{self.model_type}_class']
        unique_label_pd = label_pd.unique()

        num_rows = data_pd.shape[0]
        if self.model_type == 'nickname':
            if num_rows < 4000:
                test_size = 0.2
            elif num_rows < 12000:
                test_size = 0.15
            else:
                test_size = 0.1
        else:
            if num_rows < 5000:
                test_size = 0.25
            elif num_rows < 15000:
                test_size = 0.2
            else:
                test_size = 0.15

        self._data_pd, self._label_pd, self._test_size = data_pd, label_pd, test_size
        self._label_index_map = {label: idx for idx, label in enumerate(unique_label_pd)}

        print(f'model: {self.model_type}\n\ttest_size: {test_size}\n\tdata_count: {Counter(label_pd)}')

    def _generate_train_test_data(self):
        self._train_datas, self._eval_datas, \
            self._train_labels, self._eval_labels = train_test_split(
                self._data_pd,
                self._label_pd,
                test_size=self._test_size,
                shuffle=True,
                stratify=self._label_pd
            )
        
    def _generate_scheduler(self):
        training_steps = len(self._train_loader) * self.epoches
        self.scheduler = get_scheduler(
            'linear',
            optimizer=self._optimizer,
            num_warmup_steps=0,
            num_training_steps=training_steps
        )

    def _load_model(self, model_path:str, train_model_name:str):
        current_labels_len = len(self._label_pd.unique())
        
        if self.is_valid_model_dir(model_path):
            config = AutoConfig.from_pretrained(model_path)
            original_num_labels = config.num_labels

            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=current_labels_len,
                ignore_mismatched_sizes=True
            )
            if current_labels_len != original_num_labels:
                parameter_lr, classifier_lr = 2e-5, 1e-4
            else:
                parameter_lr, classifier_lr = 2e-5, 1e-3
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                train_model_name, 
                num_labels=current_labels_len
            )
            parameter_lr, classifier_lr = 3e-5, 1e-3

        self._optimizer = torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": parameter_lr},         # 사전학습된 인코더: 낮은 학습률
            {"params": model.classifier.parameters(), "lr": classifier_lr},   # 새로운 분류기: 높은 학습률
        ])

        model.resize_token_embeddings(len(self._tokenizer), mean_resizing=True)
        self._model = model

    def _generate_loader(self):
        def get_encoding(datas: List[Any]) -> Any:
            return self._tokenizer(
                list(datas),
                truncation=True,
                padding=True,
                max_length=self.max_token_length,
                add_special_tokens=True,
                return_tensors='pt'
            )
        
        def get_loader(datas:list, labels:list, shuffle:bool) -> torch.utils.data.DataLoader:
            encoding = get_encoding(datas)
            datasets = CustomDataset(encoding, labels, self._label_index_map)
            return torch.utils.data.DataLoader(datasets, batch_size=self.batch_size, shuffle=shuffle)
        
        self._generate_train_test_data()
        self._train_loader = get_loader(self._train_datas.tolist(), self._train_labels.tolist(), True)
        self._eval_loader = get_loader(self._eval_datas.tolist(), self._eval_labels.tolist(), False)

    def is_valid_model_dir(self, path: str) -> bool:
        return os.path.isdir(path) and any(os.scandir(path))

    # 학습
    def train(self):
        self._model.to(self.device)
        self._generate_loader()
        self._generate_scheduler()

        scaler = torch.amp.GradScaler(self.device)
        for epoch in range(self.epoches):
            self._model.train()
            loop = tqdm(self._train_loader, desc=f'  {self.model_type} Epoch {epoch}', leave=True)
            for batch in loop:
                batch = {key: val.to(self.device) for key, val in batch.items()}

                self._optimizer.zero_grad()

                with torch.autocast(self.device_type):
                    outputs = self._model(**batch)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(self._optimizer)
                scaler.update()
                self.scheduler.step()

                loop.set_postfix(loss=loss.item())

                del outputs, loss, batch
            torch.cuda.empty_cache()

            if epoch % 2 == 1:
                self.evaluate()

    # 검증
    def evaluate(self) -> Union[float, str]:
        self._model.to(self.device)
        self._model.eval()
        val_loss = 0
        correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self._eval_loader:
                batch = {key: val.to(self.device).clone().detach() for key, val in batch.items()}

                outputs = self._model(**batch)
                val_loss += outputs.loss.item() * self.batch_size

                predictions = torch.argmax(outputs.logits, dim=-1)
                labels = batch['labels']

                correct += (predictions == batch['labels']).sum().item()

                # cpu로 이동
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                del outputs, predictions, batch
            torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(self._eval_loader)

        # binary: 이진분류
        # macro: 클래스 별 평균
        # weighted: 클래스 수를 가중치로
        report = classification_report(
            all_labels, all_preds, digits=6, output_dict=False
        )

        print(f"  Validation Loss: {avg_val_loss}")
        print(report)

        return avg_val_loss, report

    def save(self):
        self._model.save_pretrained(self.model_path)

        fp16 = self._model.half()
        fp16.save_pretrained(self.model_path+"_fp16")
        print(f"{self.model_type} model and tokenizer saved")

    def predict(self, text) -> str:
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_token_length)
        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        with torch.no_grad():
            outputs = self._model(**tokens)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        return self._label_pd[predicted_class]

def delete_model(model):
    try:
        del model
    except:
        print("model 없음")

from helpers.s3_helper import S3Helper

def train_and_eval(model:TrainModel, epoch:int) -> None:
    print(f' {model.model_type}')
    model.train()
    if epoch % 2 == 1:
        model.evaluate()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='훈련을 진행합니다.')
    parser.add_argument('-u', '--upload', action='store_true', help='생성한 모델들을 업로드합니다.')
    parser.add_argument('-s', '--save', action='store_true', help='모델을 불러와 그대로 저장합니다. fp16 잘 되는지 테스트용')
    parser.add_argument('-r', '--reset', action='store_true', help='모델을 삭제합니다. 처음부터 만들기 위한 목적')

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

    save_root_path = Path(os.path.join(os.path.expanduser('~'), 'youtube-comment-colab', 'model'))
    if not save_root_path.exists():
        save_root_path.mkdir()

    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(project_root_dir, "env", ".env"))

    project_root_path = os.path.dirname(__file__)

    helper = S3Helper(project_root_path, 'youtube-comment-predict')

    if args.reset or not os.path.exists('./model/nickname_model'):
        main_train_loops = 3
        train_epoches = 3
    else:
        main_train_loops = 1
        train_epoches = 5

    if args.reset:
        import shutil
        shutil.rmtree('./model')

    if args.train:
        tokenizer = TokenizeManager()
        tokenizer.update_tokenizer()
        tokenizer.save_tokenizer()

        helper.download(['dataset.csv'])
        df = pd.read_csv(os.path.join(save_root_path, "dataset.csv"), usecols=["nickname", "comment", "nickname_class", "comment_class"])
        # csv로 내보낼때 변경한 값을 처리
        df['comment'] = df['comment'].str.replace(r'\\', ',', regex=True)   # spread sheet에서 export할 때 , 를 \ 로 바꿔놨음. 안그러면 csv가 지랄하더라...

        df = run_text_preprocessing(df, './tokens/emojis.txt')

        batch_size = 16

        nickname_model = TrainModel(df, "nickname", save_path=save_root_path, epoches=train_epoches, batch_size=batch_size)
        comment_model = TrainModel(df, "comment", save_path=save_root_path, epoches=train_epoches, batch_size=batch_size)

        for i in range(main_train_loops):
            print(f'train epoch: {i + 1}')
            train_and_eval(nickname_model, train_epoches)
            train_and_eval(comment_model, train_epoches)

        nickname_model.save()
        comment_model.save()
        del nickname_model, comment_model

    if args.upload:
        helper.upload(from_local=True)

    if args.save:
        df = pd.read_csv(os.path.join(save_root_path, "dataset.csv"), usecols=["nickname", "comment", "nickname_class", "comment_class"])
        df['comment'] = df['comment'].str.replace(r'\\', ',', regex=True)

        batch_size = 16
        
        nickname_model = TrainModel(df, "nickname", save_path=save_root_path, epoches=5, batch_size=batch_size)
        nickname_model.save()
        del nickname_model

        comment_model = TrainModel(df, "comment", save_path=save_root_path, epoches=5, batch_size=batch_size)
        comment_model.save()
        del comment_model