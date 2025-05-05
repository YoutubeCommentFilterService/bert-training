import os
from transformers import AutoTokenizer
from typing import Optional, List, Set
import json

class TokenizeManager():
    def __init__(self, train_model_name:str="klue/bert-base", root_project_path:str="..", is_clear:bool=False):
        self.train_model_name = train_model_name
        self.tokenizer_path = f"{root_project_path}/model/tokenizer"
        self.root_project_path = root_project_path
        self.is_clear = is_clear
        self.train_model_name = self.get_tokenizer_type()
        print(self.train_model_name)

    def is_valid_tokenizer_dir(self, path: str) -> bool:
        return os.path.isdir(path) and any(os.scandir(path))

    def get_tokenizer_type(self):
        if self.is_clear:
            return self.train_model_name
        if not os.path.exists(f'{self.root_project_path}/model/tokenizer'):
            return self.train_model_name
        if not os.path.exists(f'{self.root_project_path}/model/tokenizer/added_tokens.json'):
            return self.train_model_name
        try:
            with open(f'{self.root_project_path}/model/tokenizer/added_tokens.json', 'r', encoding='utf-8') as f:
                tokens = json.load(f)
            if 'ㅏㅡㅑ' not in tokens:
                return self.train_model_name
        except Exception as e:
            return self.train_model_name
        return self.tokenizer_path

    def update_tokenizer(self):
        def get_unique_token(path: str, existing_tokens_list: Optional[List[str]]) -> List[str]:
            def load_tokens_list(path: str) -> List[str]:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return list(dict.fromkeys([line.strip() for line in f if line.strip()]))
                except FileNotFoundError:
                    return []
            more_tokens_list = load_tokens_list(path)
            exsiting_tokens_set = set(existing_tokens_list)
            return [ x for x in more_tokens_list if x not in exsiting_tokens_set ]
        
        tokenizer = AutoTokenizer.from_pretrained(self.train_model_name)

        unique_special_tokens, unique_common_tokens = [], []

        unique_special_tokens.extend(get_unique_token(f'{self.root_project_path}/tokens/special_tokens.txt', tokenizer.additional_special_tokens))
        unique_common_tokens.extend(get_unique_token(f'{self.root_project_path}/tokens/common_punct_tokens.txt', tokenizer.get_vocab().keys()))
        unique_common_tokens.extend(get_unique_token(f'{self.root_project_path}/tokens/common_shortcut_tokens.txt', tokenizer.get_vocab().keys()))
        unique_common_tokens.extend(get_unique_token(f'{self.root_project_path}/tokens/common_word_tokens.txt', tokenizer.get_vocab().keys()))

        if unique_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': unique_special_tokens})
        if unique_common_tokens:
            tokenizer.add_tokens(unique_common_tokens)
        self._tokenizer = tokenizer

    def save_tokenizer(self):
        self._tokenizer.save_pretrained(self.tokenizer_path)