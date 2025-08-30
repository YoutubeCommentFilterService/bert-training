import os
from pathlib import Path

from dotenv import load_dotenv
import argparse

from helpers.tokenize_manager import TokenizeManager
import shutil
import re

import pandas as pd

from helpers.text_preprocessing import TextNormalizator
from helpers.model_train import TrainModel

import torch

from helpers.s3_helper import S3Helper

project_root_path = Path(__file__).parent.resolve()

def train_and_eval(model:TrainModel, loop:int) -> None:
    print(f' {model.model_type}')
    model.train(loop)
    if loop % 2 == 1:
        model.evaluate()
    torch.cuda.empty_cache()

def remove_model(model_type: str):
    p = Path('model')
    if not p.exists():
        return
    
    for subdir in p.iterdir():
        if subdir.is_dir() and re.match(rf'^{model_type}', subdir.name):
            shutil.rmtree(subdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Comment Classification Trainer")

    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('-ta', '--train-all', action='store_true', help='모든 훈련을 진행합니다.')
    train_group.add_argument('-tt', '--train-tokenizer', action='store_true', help='토크나이저 훈련을 진행합니다.')
    train_group.add_argument('-tc', '--train-comment', action='store_true', help='댓글 모델 훈련을 진행합니다.')
    train_group.add_argument('-tn', '--train-nickname', action='store_true', help='닉네임 모델 훈련을 진행합니다.')

    utility_group = parser.add_argument_group('Utility Options')
    utility_group.add_argument('-ra', '--reset-all', action='store_true', help='모든 모델을 삭제합니다.')
    utility_group.add_argument('-rt', '--reset-tokenizer', action='store_true', help='토크나이저를 삭제합니다.')
    utility_group.add_argument('-rc', '--reset-comment', action='store_true', help='댓글 모델을 삭제합니다.')
    utility_group.add_argument('-rn', '--reset-nickname', action='store_true', help='닉네임 모델을 삭제합니다.')

    utility_group.add_argument('-ua', '--upload-all', action='store_true', help='생성한 모델들을 업로드합니다.')
    utility_group.add_argument('-ut', '--upload-tokenizer', action='store_true', help='생성한 토크나이저를 업로드합니다.')
    utility_group.add_argument('-uc', '--upload-comment', action='store_true', help='생성한 댓글 모델을 업로드합니다.')
    utility_group.add_argument('-un', '--upload-nickname', action='store_true', help='생성한 닉네임 모델을 업로드합니다.')
    
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

    dummy_df = pd.DataFrame({
        "nickname": [],
        "nickname_class": [],
        "comment": [],
        "comment_class": []
    })

    model_root_path = project_root_path/'model'
    model_root_path.mkdir(exist_ok=True)

    load_dotenv(project_root_path/'env'/'.env')

    if args.reset_all:
        args.reset_tokenizer = True
        args.reset_comment = True
        args.reset_nickname = True

    if args.train_all:
        args.train_tokenizer = True
        args.train_comment = True
        args.train_nickname = True

    if args.upload_all:
        args.upload_tokenizer = True
        args.upload_comment = True
        args.upload_nickname = True

    if args.upload_tokenizer or args.upload_comment or args.upload_nickname or args.train_comment or args.train_nickname:
        helper = S3Helper(str(project_root_path), 'youtube-comment-predict')

    nickname_train_loops, nickname_train_epoches = 1, 3
    comment_train_loops, comment_train_epoches = 1, 3

    if args.reset_tokenizer:
        remove_model('tokenizer')

    if args.reset_nickname or not (model_root_path/'nickname_model').exists():
        remove_model('nickname')
        nickname_train_loops, nickname_train_epoches = 3, 3

    if args.reset_comment or not (model_root_path/'comment_model').exists():
        remove_model('comment')
        comment_train_loops, comment_train_epoches = 3, 3

    if args.train_tokenizer:
        # tokenizer는 토큰을 업데이트해서 저장하는 용도이기때문에 데이터셋이 필요 없다.
        tokenizer = TokenizeManager(root_project_path=".", is_clear=args.reset_all)
        tokenizer.update()
        tokenizer.save()

        if not args.train_nickname and (model_root_path/'nickname').exists():
            TrainModel(dummy_df, 'nickname', save_path=model_root_path).save()
        if not args.train_comment and (model_root_path/'comment').exists():
            TrainModel(dummy_df, 'comment', save_path=model_root_path).save()

    if args.train_comment or args.train_nickname:
        batch_size = 16
        
        helper.download(['dataset.csv'])

        text_normalizator = TextNormalizator(normalize_file_path='./tokens/text_preprocessing.json', emoji_path='./tokens/emojis.txt', tokenizer_path='./model/tokenizer')
        df = pd.read_csv(model_root_path/"dataset.csv", usecols=["nickname", "comment", "nickname_class", "comment_class"])

        # csv로 내보낼때 변경한 값을 처리
        df['comment'] = df['comment'].str.replace(r'\\', ',', regex=True)   # spread sheet에서 export할 때 , 를 \ 로 바꿔놨음. 안그러면 csv가 지랄하더라...
        text_normalizator.run_text_preprocessing(df)

    if args.train_nickname:
        nickname_model = TrainModel(df, "nickname", save_path=model_root_path, epoches=nickname_train_epoches, batch_size=batch_size)
        nickname_model.set_scheduler(nickname_train_loops)

        for i in range(nickname_train_loops):
            print(f'train epoch: {i + 1}')
            train_and_eval(nickname_model, i)
        
        nickname_model.save()
        del nickname_model

    if args.train_comment:
        comment_model = TrainModel(df, "comment", save_path=model_root_path, epoches=comment_train_epoches, batch_size=batch_size)
        comment_model.set_scheduler(comment_train_loops)

        for i in range(comment_train_loops):
            print(f'train epoch: {i + 1}')
            train_and_eval(comment_model, i)
        
        comment_model.save()
        del comment_model


    if args.upload_tokenizer or args.upload_nickname or args.upload_comment:
        target_upload_path = []
        base_model_path = os.path.join(project_root_path, 'model')
        if args.upload_tokenizer:
            target_upload_path.append(os.path.join(base_model_path, 'tokenizer'))
        if args.upload_nickname:
            target_upload_path.append(os.path.join(base_model_path, 'nickname_model_fp16'))
        if args.upload_comment:
            target_upload_path.append(os.path.join(base_model_path, 'comment_model_fp16'))
        helper.upload(local_fpaths=target_upload_path)
        print('upload finished')