import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

import pandas as pd

import torch

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from optimum.onnxruntime import ORTModelForSequenceClassification
from onnxruntime import SessionOptions, GraphOptimizationLevel

from torch.utils.data import DataLoader
from torch.optim import AdamW

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, label_index_map):
        self.encodings = encodings
        self.labels = [label_index_map[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class TrainModel():
    def __init__(self, data: pd.core.frame.DataFrame, model_type: str, save_path:str, batch_size:int = 16, epoches: int = 10, lr: float = 1e-5):
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.__type = model_type
        self.__max_token_length = 256 if model_type == 'comment' else 40
        self.__epoches = epoches
        self.__lr = lr
        self.__batch_size = batch_size

        self.__model_path = f"{save_path}/{self.__type}_model"
        self.__tokenizer_path = f"{save_path}/{self.__type}_tokenizer"
        self.__onnx_save_path = f"{save_path}/{self.__type}_onnx"

        self.__assign_pandas_data(data)
        self.__load_model()
        self.__get_loader()

        self.__scheduler = get_scheduler(
            'linear',
            optimizer=self.__optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.__train_loader)*self.__epoches
        )

    def __assign_pandas_data(self, data: pd.core.frame.DataFrame):
        data_pd = data[f'{self.__type}'].dropna()

        self.__label_pd = data[f'{self.__type}_class'].dropna()
        self.__unique_label_pd = self.__label_pd.unique()

        print(len(data_pd), len(self.__unique_label_pd))

        self.__label_index_map = {label: idx for idx, label in enumerate(self.__unique_label_pd)}

        self.__train_datas, self.__eval_datas, self.__train_labels, self.__eval_labels = train_test_split(data_pd, self.__label_pd, test_size=0.1, shuffle=True)

    def __load_model(self):
        # if os.path.exists(self.__model_path) and os.path.exists(self.__tokenizer_path):
        #     self.__model = AutoModelForSequenceClassification.from_pretrained(self.__model_path)
        #     self.__tokenizer = AutoTokenizer.from_pretrained(self.__tokenizer_path)
        # else:
        trainModelName = "klue/bert-base"
        with open('special_tokens.txt', 'r', encoding='utf-8') as f:
            special_tokens = f.read().splitlines()
        self.__tokenizer = AutoTokenizer.from_pretrained(trainModelName)
        self.__tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

        self.__model = AutoModelForSequenceClassification.from_pretrained(trainModelName, num_labels=len(self.__unique_label_pd.tolist()))
        self.__model.resize_token_embeddings(len(self.__tokenizer))

        self.__optimizer = AdamW(self.__model.parameters(), lr=self.__lr)

    def __get_loader(self):
        train_encoding = self.__tokenizer(
            list(self.__train_datas), truncation=True, padding=True, max_length=self.__max_token_length, add_special_tokens=True, return_tensors='pt'
        )
        eval_encoding = self.__tokenizer(
            list(self.__eval_datas), truncation=True, padding=True, max_length=self.__max_token_length, add_special_tokens=True, return_tensors='pt'
        )

        train_datasets = CustomDataset(train_encoding, self.__train_labels.tolist(), self.__label_index_map)
        eval_datasets = CustomDataset(eval_encoding, self.__eval_labels.tolist(), self.__label_index_map)

        self.__train_loader = DataLoader(train_datasets, batch_size=self.__batch_size, shuffle=True)
        self.__eval_loader = DataLoader(eval_datasets, batch_size=self.__batch_size, shuffle=False)

    # 학습
    def train(self):
        self.__model.to(self.__device)
        for epoch in range(self.__epoches):
            self.__model.train()
            loop = tqdm(self.__train_loader, leave=True)
            for batch in loop:
                batch = {key: val.to(self.__device).clone().detach() for key, val in batch.items()}

                outputs = self.__model(**batch)
                loss = outputs.loss

                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()
                self.__scheduler.step()

                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

    # 검증
    def evaluate(self):
        self.__model.to(self.__device)
        self.__model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for batch in self.__eval_loader:
                batch = {key: val.to(self.__device).clone().detach() for key, val in batch.items()}

                outputs = self.__model(**batch)
                val_loss += outputs.loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()

        avg_val_loss = val_loss / len(self.__eval_loader)
        accuracy = correct / len(self.__eval_loader.dataset)
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Accuracy: {accuracy}")

        return avg_val_loss, accuracy

    def save(self):
        self.__model.save_pretrained(self.__model_path)
        self.__tokenizer.save_pretrained(self.__tokenizer_path)
        print(f"{self.__type} model and tokenizer saved")

        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        ort_model = ORTModelForSequenceClassification.from_pretrained(self.__model_path, export=True, use_io_binding=True)
        ort_model.save_pretrained(self.__onnx_save_path, session_options=sess_options)
        print(f"{self.__type} onnx saved")

    def predict(self, text):
        tokens = self.__tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.__max_token_length)
        tokens = {key: val.to(self.__device) for key, val in tokens.items()}

        with torch.no_grad():
            outputs = self.__model(**tokens)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        return self.__label_pd[predicted_class]

def delete_model(model):
    try:
        del model
    except:
        print("model 없음")

from google_drive_helper import GoogleDriveHelper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='훈련을 진행합니다.')
    parser.add_argument('-u', '--upload', action='store_true', help='생성한 모델들을 업로드합니다.')

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

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

    helper = GoogleDriveHelper(project_root_dir=project_root_dir,
                            google_client_key_path=google_client_key_path,
                            google_drive_owner_email=google_drive_owner_email,
                            do_not_download_list=do_not_download_list,
                            do_not_upload_list=do_not_upload_list,
                            local_target_root_dir_name='model',
                            drive_root_folder_name='comment-filtering')
    
    # helper.print_directory_metadata()
    helper.download_all_files('dataset.csv')

    if args.train:
        df = pd.read_csv(os.path.join(save_root_path, "dataset.csv"), usecols=["nickname", "comment", "nickname_class", "comment_class"])

        df['nickname'] = df['nickname'].str.replace('@', '')
        df['nickname'] = df['nickname'].str.replace(r'[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ0-9-_.]+', '[DEFAULT_NICK]', regex=True)
        df['nickname'] = df['nickname'].str.replace(r"-[a-zA-Z0-9]{3}$|-[a-zA-Z0-9]{5}$", "", regex=True)
        df['nickname'] = df['nickname'].str.replace(r'[-._]', ' ', regex=True)
        df['nickname'] = df['nickname'].str.strip()


        df['comment'] = df['comment'].str.replace('|', ',', regex=True)   # spread sheet에서 export할 때 , 를 | 로 바꿔놨음. 안그러면 csv가 지랄하더라...
        df['comment'] = df['comment'].str.replace(r'https?:\/\/[^\s]+|www\.[^\s]+', '[URL]', regex=True)
        df['comment'] = df['comment'].str.replace(r'#(\w+)', '[HASH_TAG]', regex=True)
        df['comment'] = df['comment'].str.replace(r'[’‘]+', "'", regex=True)
        df['comment'] = df['comment'].str.replace(r'[”“]+', '"', regex=True)
        df['comment'] = df['comment'].str.replace(r'[\*\^]', '', regex=True)
        df['comment'] = df['comment'].str.replace(r'\d+:(?:\d+:?)?\d+', '[TIME]', regex=True)
        df['comment'] = df['comment'].str.replace(r'chill', '칠', regex=True)
        df['comment'] = df['comment'].str.strip()
        df['comment'] = df['comment'].fillna('[EMPTY]')


        torch.cuda.empty_cache()
        nickname_model = TrainModel(df, "nickname", save_path=save_root_path, epoches=5)
        nickname_model.train()
        nickname_model.evaluate()
        nickname_model.save()
        torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        comment_model = TrainModel(df, "comment", save_path=save_root_path, epoches=5)
        comment_model.train()
        comment_model.evaluate()
        comment_model.save()
        torch.cuda.empty_cache()

    if args.upload:
        helper.upload_all_files()

    # for folder_name, inner_data in helper.directory_struct.items():
    #     if folder_name == 'comment-filtering':
    #         continue
        
    #     for file_name, metadata in inner_data.items():
    #         if file_name in ['id', 'parent_id']:
    #             continue
            
    #         helper.delete_file(metadata.get('id'))
        
    #     helper.delete_file(inner_data.get('id'))