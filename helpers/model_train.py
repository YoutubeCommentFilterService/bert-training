import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, AutoConfig

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from typing import Any, List, Optional, Set, Union
from collections import Counter
from tqdm import tqdm
import pandas as pd

import os

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

        self.scaler = torch.amp.GradScaler(self.device)

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
        
    def _generate_scheduler(self, loop_size: int = 3):
        training_steps = len(self._train_loader) * self.epoches * loop_size
        self.scheduler = get_scheduler(
            'linear',
            optimizer=self._optimizer,
            num_warmup_steps=int(training_steps * 0.1),
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
    
    def set_scheduler(self, loop_size: int):
        self._generate_loader()
        self._generate_scheduler(loop_size)

    # 학습
    def train(self, loop: int):
        self._model.to(self.device)
        self._generate_loader()

        for epoch in range(self.epoches):
            self._model.train()
            loop = tqdm(self._train_loader, desc=f'  {self.model_type} Epoch {epoch+1}', leave=True)
            for batch in loop:
                batch = {key: val.to(self.device) for key, val in batch.items()}

                self._optimizer.zero_grad()

                with torch.autocast(self.device_type):
                    outputs = self._model(**batch)
                    loss = outputs.loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self._optimizer)
                self.scaler.update()
                self.scheduler.step()

                loop.set_postfix(loss=loss.item())

                del outputs, loss, batch
            torch.cuda.empty_cache()

            if epoch % 2 == 1:
                self.evaluate()
        if (epoch % 2 == 0):
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

        # dummy_input_ids = torch.randint(low=1, high=self._tokenizer.vocab_size, size=(self.batch_size, self.max_token_length), dtype=torch.long)
        # dummy_attention_mask = torch.ones((self.batch_size, self.max_token_length), dtype=torch.long)
        # dummy_token_type_ids = torch.zeros((self.batch_size, self.max_token_length), dtype=torch.long)

        # if not os.path.exists(f'model/{self.model_type}_onnx'):
        #     os.makedirs(f'model/{self.model_type}_onnx', exist_ok=True)
        # torch.onnx.export(
        #     self._model,
        #     (dummy_input_ids, dummy_token_type_ids, dummy_attention_mask),
        #     f'model/{self.model_type}_onnx/model.onnx',
        #     input_names=['input_ids', 'token_type_ids', 'attention_mask'],
        #     output_names=['output'],
        #     dynamic_axes={
        #         'input_ids': {0: 'batch_size', 1: 'max_token_length'},
        #         'token_type_ids': {0: 'batch_size', 1: 'max_token_length'},
        #         'attention_mask': {0: 'batch_size', 1: 'max_token_length'},
        #         'output': {0: 'batch_size'}
        #     },
        #     opset_version=17
        # )

        print(f"{self.model_type} model saved")

    def predict(self, text) -> str:
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_token_length)
        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        with torch.no_grad():
            outputs = self._model(**tokens)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        return self._label_pd[predicted_class]