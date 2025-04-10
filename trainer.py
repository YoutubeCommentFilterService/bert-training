import os
from pathlib import Path
from typing import Any, List, Set
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

import pandas as pd
import numpy as np

import torch

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from optimum.onnxruntime import ORTModelForSequenceClassification
from onnxruntime import SessionOptions, GraphOptimizationLevel

from torch.utils.data import DataLoader
from torch.optim import AdamW
from yaml import add_path_resolver

from helpers.text_preprocessing import run_text_preprocessing

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, label_index_map):
        self.encodings = encodings
        self.labels = [label_index_map[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        item = {key: val[idx].clone().detach() if isinstance(val[idx], torch.Tensor) else val[idx] 
                for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

class TrainModel():
    def __init__(self, data: pd.core.frame.DataFrame, model_type: str, save_path:str, test_size:float = 0.1, train_model_name:str = "klue/bert-base", batch_size:int = 16, epoches: int = 10, lr: float = 1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_type = model_type
        self.max_token_length = 256 if model_type == 'comment' else 40
        self.epoches = epoches
        self.lr_bound = lr
        self.batch_size = batch_size
        self.train_model_name = train_model_name
        self.test_size = test_size

        self.model_path = f"{save_path}/{self.model_type}_model"
        self.tokenizer_path = f"{save_path}/tokenizer"

        self.__assign_pandas_data(data)
        self.__load_tokenizer()
        self.__load_model()
        self.__get_loader()

        training_steps = len(self.__train_loader) * epoches
        self.scheduler = get_scheduler('linear',
                                       optimizer=self.__optimizer,
                                       num_warmup_steps=0,
                                       num_training_steps=training_steps)


    def __assign_pandas_data(self, data: pd.core.frame.DataFrame):
        columns = data[[f'{self.model_type}_class', f'{self.model_type}']]
        columns = columns.dropna(how='any')
        data_pd = columns[f'{self.model_type}']

        self.__label_pd = columns[f'{self.model_type}_class']
        self.__unique_label_pd = self.__label_pd.unique()

        self.__label_index_map = {label: idx for idx, label in enumerate(self.__unique_label_pd)}

        self.__train_datas, self.__eval_datas, \
            self.__train_labels, self.__eval_labels = train_test_split(data_pd,
                                                                       self.__label_pd,
                                                                       test_size=self.test_size,
                                                                       shuffle=True)

    def __load_tokenizer(self) -> None:
        def load_tokens(path: str) -> Set[str]:
            with open(path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f if line.strip()])
        if os.path.isdir(self.tokenizer_path) and any(os.scandir(self.tokenizer_path)):
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        else:  
            tokenizer = AutoTokenizer.from_pretrained(self.train_model_name)

        special_tokens = load_tokens('./tokens/special_tokens.txt')
        common_tokens = load_tokens('./tokens/common_tokens.txt')

        existing_special_tokens = set(tokenizer.additional_special_tokens)
        existing_common_tokens = set(tokenizer.get_vocab().keys())

        unique_special_tokens = list(special_tokens - existing_special_tokens)
        unique_common_tokens = list(common_tokens - existing_common_tokens)

        tokenizer.add_special_tokens({'additional_special_tokens': unique_special_tokens})
        tokenizer.add_tokens(unique_common_tokens)
        self._tokenizer = tokenizer

    def __load_model(self):
        if os.path.isdir(self.model_path) and any(os.scandir(self.model_path)):
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.train_model_name, 
                                                                       num_labels=len(self.__unique_label_pd.tolist()))

        model.resize_token_embeddings(len(self._tokenizer))
        self.__model = model
        self.__optimizer = AdamW(model.parameters(), lr=self.lr_bound)


    def __get_loader(self):
        def get_encoding(datas: List[Any]) -> Any:
            return self._tokenizer(list(datas),
                                truncation=True,
                                padding=True,
                                max_length=self.max_token_length,
                                add_special_tokens=True,
                                return_tensors='pt')
        train_encoding = get_encoding(list(self.__train_datas))
        eval_encoding = get_encoding(list(self.__eval_datas))

        train_datasets = CustomDataset(train_encoding, self.__train_labels.tolist(), self.__label_index_map)
        eval_datasets = CustomDataset(eval_encoding, self.__eval_labels.tolist(), self.__label_index_map)

        self.__train_loader = DataLoader(train_datasets, batch_size=self.batch_size, shuffle=True)
        self.__eval_loader = DataLoader(eval_datasets, batch_size=self.batch_size, shuffle=False)

    # 학습
    def train(self):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__model.to(self.device)
        scaler = torch.amp.GradScaler(device_type)
        for epoch in range(self.epoches):
            self.__model.train()
            loop = tqdm(self.__train_loader, leave=True)
            for batch in loop:
                batch = {key: val.to(self.device) for key, val in batch.items()}

                self.__optimizer.zero_grad()

                with torch.autocast(device_type):
                    outputs = self.__model(**batch)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(self.__optimizer)
                scaler.update()
                self.scheduler.step()

                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

                del outputs, loss, batch
                torch.cuda.empty_cache()
    # 검증
    def evaluate(self):
        self.__model.to(self.device)
        self.__model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for batch in self.__eval_loader:
                batch = {key: val.to(self.device).clone().detach() for key, val in batch.items()}

                outputs = self.__model(**batch)
                val_loss += outputs.loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()

                del outputs, predictions, batch
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(self.__eval_loader)
        accuracy = correct / len(self.__eval_loader.dataset)
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Accuracy: {accuracy}")

        return avg_val_loss, accuracy

    def save(self):
        self.__model.save_pretrained(self.model_path)
        self._tokenizer.save_pretrained(self.tokenizer_path)

        fp16 = self.__model.half()
        fp16.save_pretrained(self.model_path+"_fp16")
        print(f"{self.model_type} model and tokenizer saved")

        # 당분간 onnxruntime은 사용하지 않기로 한다.
        # torch로 하는것도 버그 터져 죽겠는데 뭔 onnx여...
        # sess_options = SessionOptions()
        # sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # ort_model = ORTModelForSequenceClassification.from_pretrained(self.__model_path, export=True, use_io_binding=True)
        # ort_model.save_pretrained(self.__onnx_save_path, session_options=sess_options)
        # print(f"{self.__type} onnx saved")

    def predict(self, text):
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_token_length)
        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        with torch.no_grad():
            outputs = self.__model(**tokens)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        return self.__label_pd[predicted_class]

def delete_model(model):
    try:
        del model
    except:
        print("model 없음")

# from google_drive_helper import GoogleDriveHelper
from helpers.s3_helper import S3Helper

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

    space_pattern = r'[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9:]+[\s!?@.,❤]*'
    pattern = r"(\w)([\s!?@.,❤]+)(\b\w\b)"

    # save_root_path = '/content/drive/MyDrive/comment-filtering'
    # data = pd.read_csv("./dataset.csv", usecols=["nickname", "comment", "nickname_class", "comment_class"])

    save_root_path = Path(os.path.join(os.path.expanduser('~'), 'youtube-comment-colab', 'model'))
    if not save_root_path.exists():
        save_root_path.mkdir()

    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(project_root_dir, "env", ".env"))

    google_drive_owner_email = os.getenv("GOOGLE_DRIVE_OWNER_EMAIL")
    do_not_download_list = ['dataset-backup']
    do_not_upload_list = ['dataset.csv']
    google_client_key_path = os.path.join(project_root_dir, 'env', 'ml-server-key.json')

    # helper = GoogleDriveHelper(project_root_dir=project_root_dir,
    #                         google_client_key_path=google_client_key_path,
    #                         google_drive_owner_email=google_drive_owner_email,
    #                         do_not_download_list=do_not_download_list,
    #                         do_not_upload_list=do_not_upload_list,
    #                         local_target_root_dir_name='model',
    #                         drive_root_folder_name='comment-filtering')
    
    # helper.print_directory_metadata()
    # helper.download_all_files('dataset.csv')
    root_path = os.path.dirname(__file__)

    helper = S3Helper(root_path, 'youtube-comment-predict')

    if args.reset:
        import shutil
        shutil.rmtree('./model')

    if args.train:
        helper.download(['dataset.csv'])
        df = pd.read_csv(os.path.join(save_root_path, "dataset.csv"), usecols=["nickname", "comment", "nickname_class", "comment_class"])
        # csv로 내보낼때 변경한 값을 처리
        df['comment'] = df['comment'].str.replace(r'\\', ',', regex=True)   # spread sheet에서 export할 때 , 를 \ 로 바꿔놨음. 안그러면 csv가 지랄하더라...

        df = run_text_preprocessing(df, './tokens/emojis.txt')

        print(df['comment'])

        batch_size = 16

        nickname_test_size = 0.1
        comment_test_size = 0.2

        torch.cuda.empty_cache()
        nickname_model = TrainModel(df, "nickname", save_path=save_root_path, test_size=nickname_test_size, epoches=5, batch_size=batch_size)
        nickname_model.train()
        nickname_model.evaluate()
        nickname_model.save()
        torch.cuda.empty_cache()

        del nickname_model

        torch.cuda.empty_cache()
        comment_model = TrainModel(df, "comment", save_path=save_root_path, test_size=comment_test_size, epoches=5, batch_size=batch_size)
        comment_model.train()
        comment_model.evaluate()
        comment_model.save()
        torch.cuda.empty_cache()

        del comment_model

    if args.upload:
        # helper.upload_all_files()
        helper.upload(from_local=True)

    # for folder_name, inner_data in helper.directory_struct.items():
    #     if folder_name == 'comment-filtering':
    #         continue
        
    #     for file_name, metadata in inner_data.items():
    #         if file_name in ['id', 'parent_id']:
    #             continue
            
    #         helper.delete_file(metadata.get('id'))
        
    #     helper.delete_file(inner_data.get('id'))

    if args.save:
        df = pd.read_csv(os.path.join(save_root_path, "dataset.csv"), usecols=["nickname", "comment", "nickname_class", "comment_class"])
        df['comment'] = df['comment'].str.replace(r'\\', ',', regex=True)

        batch_size = 16

        nickname_test_size = 0.1
        comment_test_size = 0.2
        
        nickname_model = TrainModel(df, "nickname", save_path=save_root_path, test_size=nickname_test_size, epoches=5, batch_size=batch_size)
        nickname_model.save()
        del nickname_model

        comment_model = TrainModel(df, "comment", save_path=save_root_path, test_size=comment_test_size, epoches=5, batch_size=batch_size)
        comment_model.save()
        del comment_model