import pandas as pd
import unicodedata
from typing import Literal, Union, Dict, List, Tuple, TypedDict
import re
import hangul_jamo
import json
from transformers import AutoTokenizer

class IncorrectType(TypedDict):
    char: Dict[str, str]
    sentence: List[Tuple[str, str, bool]]

class StructedType(TypedDict):
    base: Dict[str, List[str]]
    extern: Dict[str, List[str]]

class SingleCharType(TypedDict):
    ko: Dict[str, str]
    en: Dict[str, str]

class EmojiType(TypedDict):
    unicode: Dict[str, str]
    text: Dict[str, str]

class TextPreprocessingType(TypedDict):
    incorrect: IncorrectType
    structed: StructedType
    single: SingleCharType
    punct: Dict[str, str]
    emoji: EmojiType
    special_token: List[List[str]]

class PathType(TypedDict):
    normalized: str
    text_face_emoji: str
    tokenizer: str

class TextNormalizator:
    def __init__(self, normalize_file_path: str = '../tokens/text_preprocessing.json', emoji_path: str = '../tokens/emojis.txt', tokenizer_path: str = '../model/tokenizer'):
        self.path: PathType = {
            'normalize': normalize_file_path,
            'text_face_emoji': emoji_path,
            'tokenizer': tokenizer_path
        }

        self.reload()

    def run_text_preprocessing(self, df: pd.DataFrame):
        def trace_error(df: pd.DataFrame, cnt: int):
            print(cnt)
            return df
        
        # 일단 전처리
        df = self._normalize_unicode(df)
        df = self._clean_nickname(df)

        # 스팸 형식 제거
        df = self._normalize_incorrect_grammar(df)

        # 실행
        df = (
            self._replace_special_tokens(df)
                .pipe(self._replace_misc_patterns)
                .pipe(self._sort_punct)
                .pipe(self._replace_structed_patterns)
                .pipe(self._cleanup_formatting)
                .pipe(self._clean_duplicated_token)
        )
        df = (
            self._remove_isolated_english(df)
                .pipe(self._set_default_nickname)
        )
        return df

    def reload(self):
        with open(self.path['normalize'], 'r', encoding='utf-8') as f:
            self.normalize_type: TextPreprocessingType = json.load(f)

        with open(self.path['text_face_emoji'], 'r', encoding='utf-8') as f:
            self.text_face_emoji = [ line.strip() for line in f.readlines() if line ]
        
        try:
            self.special_tokens = AutoTokenizer.from_pretrained(self.path['tokenizer']).additional_special_tokens
        except Exception as e:
            with open(self.path['tokenizer'], 'r', encoding='utf-8') as f:
                self.special_tokens = [ line.strip() for line in f.readlines() ]

    def change_path(self, type: Literal['normalize', 'face', 'tokenizer'], path: str):
        if type == 'normalize':
            self.path['normalize'] = path
        elif type == 'face':
            self.path['text_face_emoji'] = path
        elif type =='tokenizer':
            self.path['tokenizer'] = path

    def _normalize_unicode_text(self, text: str) -> str:
        unicode_single_hangul_dict = self.normalize_type['single']['ko']
        unicode_alphabet_dict = self.normalize_type['single']['en']

        subbed_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\s+', r'\1', text)
        normalized = hangul_jamo.compose(subbed_text)
        normalized = unicodedata.normalize("NFKC", normalized)
        normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = ''.join(unicode_single_hangul_dict.get(ch, ch) for ch in normalized)
        for char, pattern in unicode_alphabet_dict.items():
            normalized = re.sub(pattern, char, normalized)
        normalized = re.sub(r'[cㄷ][o0ㅇ]m', 'com', normalized, flags=re.IGNORECASE)
        normalized = ''.join(ch for ch in normalized if not ('\u4E00' <= ch <= '\u9FFF'))

        return normalized
    
    def _normalize_korify(self, text: str):
        CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
        JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
        JUNG_MAPPING = {
            'H': 'ㅐ', 'I': 'ㅣ', 'l': 'ㅣ'
        }
        def combine_jamos(cho, jung):
            cho_idx = CHO.find(cho)
            jung_char = JUNG_MAPPING.get(jung)
            if cho_idx == -1 or not jung_char or jung_char not in JUNG:
                return cho + jung  # 조합 불가한 건 그대로
            jung_idx = JUNG.find(jung_char)
            return chr(0xAC00 + cho_idx * 588 + jung_idx * 28)
        
        return re.sub(r'([ㄱ-ㅎ])[ ,\\]*([A-Za-z])', lambda m: combine_jamos(m.group(1), m.group(2)), text)
    
    def _normalize_tlettak_font(
            self,
            text: str, 
            space_pattern: Union[str, re.Pattern] = r'\s*[\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\(\)\[\]\{\}\-+=~,.\/<>;:\'"]+[\s!?@.,❤]*', 
            search_pattern: Union[str, re.Pattern] = r'\b([\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\&\^\-+=~,.\/<>;:\'"]{1}\b)([\s!?\^@.,ㅣ~❤]+)(\b[\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\-+=~,.\/<>;:\'"]{1}\b)'
        ) -> str:
    
        space_pattern = re.compile(space_pattern) if isinstance(space_pattern, str) else space_pattern
        search_pattern = re.compile(search_pattern) if isinstance(search_pattern, str) else search_pattern

        text = re.sub(r'([\(\[\{])', r' \1 ', text)  # 여는 괄호
        text = re.sub(r'([\)\]\}])', r' \1 ', text)  # 닫는 괄호

        result = []
        substr = []
        pos = 0
        length = len(text)
        
        while pos < length:
            if (search_matched := search_pattern.match(text, pos)):
                substr.extend([search_matched.group(1), search_matched.group(3)])
                pos = search_matched.end() - 1
            elif (space_matched := space_pattern.match(text, pos)):
                s_end = space_matched.end()
                result.append(''.join(substr[::2]) + text[pos:s_end].strip())
                pos = s_end
                substr.clear()
            else:   # 둘 다 매칭 실패인 경우 뒷문장 전부를 붙여씀
                result.append(text[pos:])
                break

        text = ' ' .join(result)
        text = re.sub(r'([\(\[\{]) | ([\)\]\}])', lambda m: m.group(1) or m.group(2), text)
        return text
    
    def _normalize_unicode(self, df: pd.DataFrame):
        df['comment'] = (
            df['comment'] # \u2640\u2642\u2695\u2696\u2708\u2764
                .str.replace(r'[\u0020\u200b\u2002\u2003\u2007\u2008\u200c\u200d]+', ' ', regex=True)
                .str.replace(r'[\U0001F3FB-\U0001F3FF\uFE0F]', '', regex=True)
                .str.replace(r'\*+', '', regex=True)
                .str.replace('9글', '구글')
                .apply(lambda x: self._replace_unicode_punctuation(x) if isinstance(x, str) else x)
                .apply(lambda x: self._normalize_unicode_text(x) if isinstance(x, str) else x)
                .apply(lambda x: self._normalize_korify(x) if isinstance(x, str) else x)
        )
        df['comment'] = df['comment'].str.replace(r'(?<!\d)(.)\1{2,}', r'\1\1', regex=True)
        return df
    
    def _replace_unicode_punctuation(self, text: str) -> str:
        unicode_punctuation_map = self.normalize_type['punct']
        
        for key, pattern in unicode_punctuation_map.items():
            text = re.sub(pattern, key, text)
        return text
    
    def _replace_special_tokens(self, df: pd.DataFrame):
        def _replace_emoji(text: str):
            unicode_emojis = self.normalize_type['emoji']['unicode']
            text_base_emojis = self.normalize_type['emoji']['text']

            for tag, emoji_str in unicode_emojis.items():
                pattern = r'(?:[' + ''.join(re.escape(c) for c in emoji_str) + r'])+'
                text = re.sub(pattern, tag, text)

            for tag, emoji_arr in text_base_emojis.items():
                pattern =r'(?i)' +  r'(?:' + '|'.join(emoji_arr) + r')+'
                text = re.sub(pattern, tag, text)

            return text
        # TODO: 텍스트 조합 기반 이모지도 다 바꿔야 한다
        with open(self.path['text_face_emoji'], 'r', encoding='utf-8') as f:
            text_face_emojis = [line.strip() for line in f.readlines()]
        emoji_pattern = '|'.join(map(re.escape, text_face_emojis))
        special_tokens = self.normalize_type['special_token']
        for token in special_tokens:
            df['comment'] = df['comment'].str.replace(token[0], token[1], regex=True)
        df['comment'] = df['comment'].str.replace(emoji_pattern, '[TEXT_EMOJI]', regex=True)
        df['comment'] = df['comment'].apply(lambda x: _replace_emoji(x) if isinstance(x, str) else x)
        return df
    
    def _cleanup_formatting(self, df: pd.DataFrame):
        patterns = [
            (r'(?<=\d)\.(?=\d)', '[POINT_DOT]'),
            (r'(?<=\d),(?=\d)', '[POINT_COM]'),
            (r'\s*([.,?!^]+)\s*', r'\1 '),
            (r'\s{2,}', ' '),
            (r'\[POINT_DOT\]', '.'),
            (r'\[POINT_COM\]', ','),
        ]
        for pattern in patterns:
            df['comment'] = df['comment'].str.replace(pattern[0], pattern[1], regex=True)
        return df
    
    def _replace_structed_patterns(self, df: pd.DataFrame):
        normalize_patterns = self.normalize_type['structed']['extern']
        patterns = self.normalize_type['structed']['base']

        for key, regexs in patterns.items():
            for regex in regexs:
                df['comment'] = df['comment'].str.replace(regex, key, regex=True)

        for key, regexs in normalize_patterns.items():
            for regex in regexs:
                df['comment'] = df['comment'].str.replace(regex, key, regex=True)
        return df
    
    def _replace_misc_patterns(self, df: pd.DataFrame):
        def _fix_spam_likely_text(text: str):
            pattern = r'(?:([가-힣]+)ㅣ([가-힣]+))+'
            while len(re.findall(pattern, text)):
                text = re.sub(pattern, r'\1\2', text)
            return text
        df['comment'] = (
            df['comment']
                .str.replace(r'\[+', '[', regex=True)
                .str.replace(r'\]+', ']', regex=True)
                .str.replace(r'[^\w가-힣ㄱ-ㅎㅏ-ㅣ!?%&\^\(\)\[\]{}\-+=~,./<>;:\'"\s]', '', regex=True)
                .str.replace(r'(?<!\d)([a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\(\)\[\]\{\}\-_+=~,.\/<>;:\'"\s])\1{3,}', r'\1\1', regex=True)
                .str.strip()
                .fillna('[EMPTY]')
                # 한글자 + 부호 + 한글자 패턴 처리
                .apply(lambda x: self._normalize_tlettak_font(x) if isinstance(x, str) else x)
                .apply(lambda x: _fix_spam_likely_text(x) if isinstance(x, str) else x)
        )
        return df
    
    def _clean_duplicated_token(self, df: pd.DataFrame):
        for special_token in self.special_tokens:
            pattern = r'(?:' + re.escape(special_token) + r'\s*)+'
            df['comment'] = df['comment'].apply(lambda x: re.sub(pattern, special_token, x) if isinstance(x, str) else x)
        return df
    
    def _clean_nickname(self, df: pd.DataFrame):
        # user로 시작하는 경우 체크
        def change_if_nickname_is_default(text: str):
            # 태그로 붙는 맨 뒤의 - 의 경우, 뒤에 대문자가 "전혀" 오지 않는다.
            return re.sub(
                r'^user-([a-z0-9]+)$',
                lambda m: '[DEFAULT_NICK]' if len(m.group(1)) % 2 == 1 or len(m.group(1)) > 7 else m.group(0),
                text
            )
        
        def change_if_starts_with_user(text: str) -> str:
            return re.sub(
                r'^user[-_.]([a-zA-Z0-9가-힣-._]+)$',
                r'\1',
                text
            )
        
        def remove_hyphen_or_underscore_format(text: str) -> str:
            pattern = r'[\-_][a-zA-Z0-9]{2,7}$'

            while matched := re.search(pattern, text):
                text = text[:matched.span(0)[0]]

            return text
        
        def normalize_nickname(text: str):
            return text if text == '[DEFAULT_NICK]' else re.sub(r'[^a-zA-Z가-힣0-9]+', '', text)
        
        df['nickname'] = (
            df['nickname']
                .str.strip()
                .str.replace(r'^@', '', regex=True)
                .apply(lambda x: change_if_nickname_is_default(x) if isinstance(x, str) else x)
                .apply(lambda x: change_if_starts_with_user(x) if isinstance(x, str) else x)
                .apply(lambda x: remove_hyphen_or_underscore_format(x) if isinstance(x, str) else x)
                .apply(lambda x: normalize_nickname(x) if isinstance(x, str) else x)
        )

        return df
    
    def _normalize_incorrect_grammar(self, df: pd.DataFrame):
        sentence_patterns = self.normalize_type['incorrect']['sentence']
        char_patterns = self.normalize_type['incorrect']['char']
        pattern = r'(' + '|'.join(map(re.escape, char_patterns.keys())) + r')'
        for column in df.columns:
            print(column)
            df[column] = df[column].str.replace(pattern, lambda match: char_patterns.get(match.group(0), match.group(0)), regex=True)

        for column in df.columns:
            for pattern, to_sub, regex_flag in sentence_patterns:
                df[column] = df[column].str.replace(pattern, to_sub, regex=regex_flag)

        return df
    
    def _set_default_nickname(self, df: pd.DataFrame):
        def _change_nickname(nickname: str):
            if re.search(r'[가-힣]', nickname):
                if len(nickname) < 3:
                    return '[DEFAULT_NICK]'
            else:
                if len(nickname) < 5:
                    return '[DEFAULT_NICK]'
            return nickname

        df['nickname'] = df['nickname'].apply(lambda x: _change_nickname(x) if isinstance(x, str) else x)
        return df
    
    def _remove_isolated_english(self, df: pd.DataFrame):
        df['nickname'] = df['nickname'].str.replace(r'(?<=[가-힣])([a-zA-Z])(?=[가-힣])(?!양)', '', regex=True)
        return df
    
    def _sort_punct(sekf, df: pd.DataFrame):
        def _sort_process(match: re.Match):
            text = ''.join(sorted(match.group(0), key=lambda c: punct_order.index(c)))

            text = re.sub(r',{2,}', '..', text)
            text = re.sub(r'\.{2,}', '..', text)
            text = re.sub(r'\?{2,}', '?', text)
            text = re.sub(r'!{2,}', '!', text)
            text = re.sub(r'\~{2,}', '~', text)

            return text
        punct_order = '~.,?!'
        punct_pattern = r'[' + re.escape(punct_order) + r']{2,}'

        df['comment'] = df['comment'].apply(lambda x: re.sub(punct_pattern, lambda x: _sort_process(x), x))
        df['comment'] = (
            df['comment']
                .replace(r'"{2,}', '\'', regex=False)
                .replace(r'\'{2,}', '\'', regex=False)
        )
        return df

if __name__ == "__main__":
    preprocess_logic = TextNormalizator()
    # with open('../tokens/emojis.txt', 'r', encoding='utf-8') as f:
    #     lines = [ line.strip() for line in f.readlines() ]
    # df = pd.DataFrame(lines, columns=['comment'])

    # df = _normalize_unicode(df)

    df = pd.read_csv('./testset.csv', encoding='utf-8')
    df['nickname'] = df['nickname'].str.strip()
    df['comment'] = df['comment'].apply(lambda x: x.replace('\\', ',') if isinstance(x, str) else x)
    df['comment'] = df['comment'].str.strip()

    updated_logic_df = df.copy()

    preprocess_logic.run_text_preprocessing(updated_logic_df)
    print(updated_logic_df)


    # df['comment'] = df['comment'].map(lambda x: x.replace(',', '\\') if isinstance(x, str) else x)
    # updated_logic_df['comment'] = updated_logic_df['comment'].map(lambda x: x.replace(',', '\\') if isinstance(x, str) else x)

    # print(updated_logic_df.iloc[341])

    # comparison_updated_logic = df['comment'].compare(updated_logic_df['comment'])

    # special_tokens = [
    #     'DAYS,' 
    #     'DATE,' 
    #     'TIME',
    #     'FLOAT',
    #     'NUMBER',
    #     'DURATION',
    #     'RANGE',
    #     'COST',
    #     'RANK',
    #     'ANNIV',
    #     'MEASURE',
    #     'UNIT'
    # ]
    # pattern = r'(?:' + '|'.join(special_tokens) + r')'
    # mask = comparison_updated_logic.astype(str).apply(lambda x: x.str.contains(pattern)).any(axis=1)
    # filtered = comparison_updated_logic[mask]
    
    # with pd.ExcelWriter('comparition_results.xlsx') as writer:
    #     filtered.to_excel(writer, sheet_name="updated_logic")

    # from openpyxl import load_workbook
    # wb = load_workbook('comparition_results.xlsx')

    # ws_updated_logic = wb['updated_logic']

    # base_width = 30

    # for idx, column in enumerate(['B', 'C']):
    #     ws_updated_logic.column_dimensions[column].width = base_width * 6

    # # 수정된 Excel 파일 저장
    # wb.save('comparition_results_with_custom_width.xlsx')