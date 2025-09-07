import pandas as pd
import unicodedata
from typing import Literal, Union, Dict, List, Tuple, TypedDict, Pattern, Callable
import re
import hangul_jamo
import json
from transformers import AutoTokenizer
import os
from functools import lru_cache

class IncorrectType(TypedDict):
    char: Dict[str, str]
    word: Dict[str, str]
    sentence: List[Tuple[str, str]]
    monde: List[Tuple[str, str]]
    sentence_eval: List[Tuple[str, Callable]] # eval(['sentencce_eval'][N][1]) 로 사용하면 됨

class StructedType(TypedDict):
    base: Dict[str, Union[List[str], List[Pattern]]]
    extern: Dict[str, Union[List[str], List[Pattern]]]

class SingleCharType(TypedDict):
    ko: Dict[str, str]
    en: Dict[str, Pattern]

class EmojiType(TypedDict):
    unicode: Dict[str, Union[str, Pattern]]
    text: Dict[str, Union[List[Pattern], Pattern]]

class TextPreprocessingType(TypedDict):
    incorrect: IncorrectType
    structed: StructedType
    single: SingleCharType
    punct: Dict[str, Pattern]
    emoji: EmojiType
    special_token: List[Tuple[str, str]]

class PathType(TypedDict):
    normalized: str
    text_face_emoji: str
    tokenizer: str

class ParenPattern(TypedDict):
    open: Tuple[Pattern, str]
    close: Tuple[Pattern, str]

class StaticPattern(TypedDict):
    paren: ParenPattern
    cleanup: List[Tuple[Pattern, str]]

class TextNormalizator:
    def __init__(self, normalize_file_path: str = '../tokens/text_preprocessing.json', emoji_path: str = '../tokens/emojis.txt', tokenizer_path: str = '../model/tokenizer'):
        self.path: PathType = {
            'normalize': normalize_file_path,
            'text_face_emoji': emoji_path,
            'tokenizer': tokenizer_path
        }

        self.reload()

        self.cho_list = [
            'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
            'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
        ]
        self.jung_list = [
            'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ',
            'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ',
            'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
            'ㅡ', 'ㅢ', 'ㅣ'
        ]
        self.jong_list = [
            None, 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ',
            'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
            'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
            'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
        ]

        self.static_patterns: StaticPattern = {
            'paren': {
                'open': (re.compile(r'([\(\[\{])'), r' \1'),
                'close': (re.compile(r'([\)\]\}])'), r' \1')
            },
            'cleanup': [
                (re.compile(r'(?<=\d)\.(?=\d)'), '[POINT_DOT]'),
                (re.compile(r'(?<=\d),(?=\d)'), '[POINT_COM]'),
                (re.compile(r'\s*([.,?!^]+)\s*'), r'\1 '),
                (re.compile(r'\s{2,}'), ' '),
                (re.compile(r'\[POINT_DOT\]'), '.'),
                (re.compile(r'\[POINT_COM\]'), ','),
            ]
        }

    def run_text_preprocessing(self, df: pd.DataFrame):
        def trace_error(df: pd.DataFrame, cnt: int):
            print(cnt)
            return df
        # 문법 수정 전 우선 수행되어야 하는 전처리들
        # 닉네임 전처리
        for column in df.columns:
            df[column] = df[column].str.lower()
        df = self._clean_nickname(df)
        df = self._normalize_unicode(df)

        # 스팸 형식 제거
        df = self._normalize_incorrect_grammar(df)
        ## self._process_dan_mo_ja

        df = self._remove_spaces(df)
        df['comment'] = df['comment'].str.lower()

        # 실행
        df = (
            self._sort_punct(df)
                .pipe(self._process_moum)
                .pipe(self._process_yeocho_font)
                .pipe(self._replace_special_tokens)
                .pipe(self._replace_misc_patterns)
                .pipe(self._replace_structed_patterns)
                .pipe(self._cleanup_formatting)
                .pipe(self._clean_duplicated_token)
        )
        df = (
            self._remove_num_end_with(df)
                .pipe(self._remove_isolated_english)
                .pipe(self._set_default_nickname)
        )
        
        df['comment'] = df['comment'].str.replace(r'z{2,}', 'ㅋㅋ', regex=True)
        # special token 복원용
        for column in df.columns:
            df[column] = df[column].str.replace(r'(\[[a-zA-Z_]+\])', lambda m: m.group(1).upper(), regex=True)
        df['comment'] = df['comment'].apply(lambda x: hangul_jamo.compose(x) if isinstance(x, str) else x)

        # df['nickname'] = df['nickname'].mask(
        #     ~df['nickname'].str.startswith('[', na=False),
        #     df['nickname'].str.lower()
        # )

        # 마지막으로 한번 더 정리
        # df = self._normalize_incorrect_grammar(df)
        return df
    
    def _remove_spaces(self, df: pd.DataFrame):
        def _rm_sp(pattern: Pattern, text: str) -> str:
            prev = None
            while prev != text:
                prev = text
                text = pattern.sub(r'\1\2', text)
            return text
        def _rm_bc(pattern: Pattern, text: str) -> str:
            while True:
                matched = pattern.search(text)
                if not matched:
                    break
                cho, jung, jong = self.decompose_hangul(matched.group(1))
                if jong == self.jong_list.index(matched.group(2)[0]):
                    text = text[:matched.start(1)] + self.compose_hangul(cho, jung) + ' ' + matched.group(2) + ' ' + text[matched.end(2):]
                else:
                    text = text[:matched.start(2)] + ' ' + matched.group(2) + ' ' + text[matched.end(2):]
            return text
        batchim_pattern = re.compile(r'([가-힣])([ㅋㅎㅌㅊ]+)')
        pre_process_pattern = re.compile(r'(?:[ㅜㅣ]\s+)+[ㅜㅣ]')
        pattern1 = re.compile(r'([ㄱ-ㅎㅏ-ㅣ])[\s0-9\?\!]+([ㄱ-ㅎㅏ-ㅣ])') # 이거때문에 ㅜ ㅜ ㅜ ㅜ 가 처리되지 않는 문제가 발생, 전처리로 "(?:ㅜ\\s+)+ㅜ 를 다른 것으로 처리"
        pattern2 = re.compile(r'([가-힣])1+([가-힣])') # <-- [^] 형식으로 안맞는건 제외

        df['comment'] = df['comment'].str.replace(pre_process_pattern, 'TEMP_U', regex=True)
        for pattern in [pattern1, pattern2]:
            df['comment'] = df['comment'].apply(lambda x: _rm_sp(pattern, x) if isinstance(x, str) else x)
        df['comment'] = df['comment'].str.replace('TEMP_U', 'ㅜ ㅜ ㅜ')
        
        df['comment'] = df['comment'].apply(lambda x: _rm_bc(batchim_pattern, x) if isinstance(x, str) else x)
        df['comment'] = df['comment'].str.replace(r'([ㄱ-ㅎㅏ-ㅣ]+)', r' \1 ', regex=True)
        df['comment'] = df['comment'].apply(lambda x: hangul_jamo.decompose(x) if isinstance(x, str) else x)
        df['comment'] = df['comment'].apply(lambda x: hangul_jamo.compose(x) if isinstance(x, str) else x)
        df['comment'] = df['comment'].str.replace(r'[ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅝㅞㅟㅢㅣ]+', '', regex=True) # ㅠㅜㅡ 제외하고 모든 단독 모음 제거
        return df

    def reload(self):
        with open(self.path['normalize'], 'r', encoding='utf-8') as f:
            self.normalize_type: TextPreprocessingType = json.load(f)

        char_patterns = self.normalize_type['incorrect']['char']
        word_patterns = self.normalize_type['incorrect']['word']

        self.char_compiled = re.compile(r'(' + '|'.join(re.escape(k) for k in char_patterns.keys()) + r')')
        self.word_compiled = re.compile(r'(' + '|'.join(re.escape(k) for k in word_patterns.keys()) + r')')

        for idx, item in enumerate(self.normalize_type['incorrect']['sentence_eval']):
            self.normalize_type['incorrect']['sentence_eval'][idx] = [re.compile(item[0]), eval(item[1])]

        for idx, item in enumerate(self.normalize_type['incorrect']['sentence']):
            self.normalize_type['incorrect']['sentence'][idx] = [re.compile(item[0]), item[1]]

        for idx, item in enumerate(self.normalize_type['incorrect']['monde']):
            self.normalize_type['incorrect']['monde'][idx] = [re.compile(item[0]), item[1]]

        for key, val in self.normalize_type['single']['en'].items():
            self.normalize_type['single']['en'][key] = re.compile(val)

        for key, val in self.normalize_type['punct'].items():
            self.normalize_type['punct'][key] = re.compile(val)

        for key, emoji_str in self.normalize_type['emoji']['unicode'].items():
            val = r'(?:[' + ''.join(re.escape(c) for c in emoji_str) + r'])+'
            self.normalize_type['emoji']['unicode'][key] = re.compile(val)

        for key, emoji_arr in self.normalize_type['emoji']['text'].items():
            val = r'(?i)' +  r'(?:' + '|'.join(emoji_arr) + r')+'
            self.normalize_type['emoji']['text'][key] = re.compile(val)

        for super_key, item_dict in self.normalize_type['structed'].items():
            for key, regex_list in item_dict.items():
                self.normalize_type['structed'][super_key][key] = [ re.compile(regex) for regex in regex_list ]
        
        with open(self.path['text_face_emoji'], 'r', encoding='utf-8') as f:
            self.text_face_emoji = [ line.strip() for line in f.readlines() if line ]
        
        try:
            self.special_tokens = AutoTokenizer.from_pretrained(self.path['tokenizer']).additional_special_tokens
        except Exception as e:
            print('exception occurred!')
            if os.path.exists(self.path['tokenizer'], 'special_tokens_map.json'):
                with open(os.path.join(self.path['tokenizer'], 'special_tokens_map.json'), 'r', encoding='utf-8') as f:
                    self.special_tokens = json.load(f)['additional_special_tokens']
            else:
                raise Exception("sperical tokens are not found")

    def change_path(self, type: Literal['normalize', 'face', 'tokenizer'], path: str):
        if type == 'normalize':
            self.path['normalize'] = path
        elif type == 'face':
            self.path['text_face_emoji'] = path
        elif type =='tokenizer':
            self.path['tokenizer'] = path

    def _normalize_unicode_text(self, text: str) -> str:
        unicode_single_hangul_dict = self.normalize_type['single']['ko']

        normalized = hangul_jamo.compose(text)
        normalized = unicodedata.normalize("NFKC", normalized)
        normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = ''.join(unicode_single_hangul_dict.get(ch, ch) for ch in normalized)
        return normalized
    
    ## TODO: space_pattern, search_pattern 손 볼 필요가 있음
    ##    전처리 결과 보니까 다 붇혀버리는데...?
    def _normalize_tlettak_font(
            self,
            text: str, 
            space_pattern: Union[str, re.Pattern] = re.compile(r'\s*[\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\(\)\[\]\{\}\-+=~,.\/<>;:\'"]+[\s!?@.,❤]*'), 
            search_pattern: Union[str, re.Pattern] = re.compile(r'\b([\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\&\^\-+=~,.\/<>;:\'"]{1}\b)([\s!?\^@.,ㅣ~❤]+)(\b[\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\-+=~,.\/<>;:\'"]{1}\b)')
        ) -> str:
    
        space_pattern = re.compile(space_pattern) if isinstance(space_pattern, str) else space_pattern
        search_pattern = re.compile(search_pattern) if isinstance(search_pattern, str) else search_pattern

        open_paren, close_paren = self.static_patterns['paren']['open'], self.static_patterns['paren']['close']

        text = re.sub(open_paren[0], open_paren[1], text)  # 여는 괄호
        text = re.sub(close_paren[0], close_paren[1], text)  # 닫는 괄호

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

        text = ' '.join(result)
        text = re.sub(r'([\(\[\{]) | ([\)\]\}])', lambda m: m.group(1) or m.group(2), text)
        return text
    
    def _normalize_unicode(self, df: pd.DataFrame):
        df['comment'] = (
            df['comment'] # \u2640\u2642\u2695\u2696\u2708\u2764 - ♀, ♂, ⚕, ⚖, ✈, ❤ 기호
                .str.replace(r'[\u2002\u2003\u2007\u2008]+', ' ', regex=True)
                .str.replace(r'[\U0001F3FB-\U0001F3FF\uFE0F\u0674\u1160\u200B\u200C\u200D\uFEFF\u2060]+', '', regex=True)
                .str.replace(r'\*+', '', regex=True)
        )
        for key, compiled_pattern in self.normalize_type['punct'].items():
            df['comment'] = df['comment'].str.replace(compiled_pattern, key, regex=True)
        
        ## 유니코드 문자 정규화 과정 <- 여기에서 축약이 발생한다
        df['comment'] = (
            df['comment']
                .str.replace(r'([ㄱ-ㅎㅏ-ㅣ])\s+', r'\1 ', regex=True)
                .apply(lambda x: self._normalize_unicode_text(x) if isinstance(x, str) else x)
        )

        unicode_alphabet_dict = self.normalize_type['single']['en']
        for char, compiled_pattern in unicode_alphabet_dict.items():
            df['comment'] = df['comment'].str.replace(compiled_pattern, char, regex=True)

        df['comment'] = df['comment'].str.replace(r'(?i)[cㄷ][o0ㅇ]m', 'com', regex=True)
        df['comment'] = df['comment'].apply(lambda x: ''.join(ch for ch in x if not ('\u4E00' <= ch <= '\u9FFF')))
            
        ## 특정 포맷으로 적은 한글 문자열을 변환
        # JUNG_MAPPING = {
        #     'H': 'ㅐ', 'I': 'ㅣ', 'l': 'ㅣ'
        # }
        # def combine_jamos(match: re.Match):
        #     cho, jung = match.groups()
        #     cho_idx = self.cho_list.index(cho) if cho in self.cho_list else -1
        #     jung_char = JUNG_MAPPING.get(jung)
        #     if cho_idx == -1 or jung_char not in self.jung_list:
        #         return cho + jung  # 조합 불가한 건 그대로
        #     jung_idx = self.jung_list.index(jung_char)
        #     return chr(0xAC00 + cho_idx * 588 + jung_idx * 28)
        # df['comment'] = df['comment'].str.replace(r'([ㄱ-ㅎ])[ ,\\]*([A-Za-z])', combine_jamos, regex=True)
        df['comment'] = df['comment'].str.replace(r'(?<!\d)(.)\1{2,}', r'\1\1', regex=True)
        return df
    
    def _replace_special_tokens(self, df: pd.DataFrame):
        # TODO: 텍스트 조합 기반 이모지도 다 바꿔야 한다
        with open(self.path['text_face_emoji'], 'r', encoding='utf-8') as f:
            text_face_emojis = [line.strip() for line in f.readlines()]
        emoji_pattern = '|'.join(map(re.escape, text_face_emojis))
        special_tokens = self.normalize_type['special_token']
        for token in special_tokens:
            df['comment'] = df['comment'].str.replace(re.compile(token[0]), token[1], regex=True)
        df['comment'] = df['comment'].str.replace(emoji_pattern, '[TEXT_EMOJI]', regex=True)
        for tag, compiled_pattern in self.normalize_type['emoji']['unicode'].items():
            df['comment'] = df['comment'].str.replace(compiled_pattern, tag, regex=True)
        for tag, compiled_pattern in self.normalize_type['emoji']['text'].items():
            df['comment'] = df['comment'].str.replace(compiled_pattern, tag, regex=True)
        return df
    
    def _cleanup_formatting(self, df: pd.DataFrame):
        for pattern in self.static_patterns['cleanup']:
            df['comment'] = df['comment'].str.replace(pattern[0], pattern[1], regex=True)
        return df
    
    def _replace_structed_patterns(self, df: pd.DataFrame):
        for key, regexs in self.normalize_type['structed']['base'].items():
            for regex in regexs:
                df['comment'] = df['comment'].str.replace(regex, key, regex=True)

        for key, regexs in self.normalize_type['structed']['extern'].items():
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
        df['nickname'] = (
            df['nickname']
                .str.strip()
                .str.replace(r'^@', '', regex=True)
                .str.replace(r'[^a-zA-Z가-힣0-9\-._]+', '', regex=True)
                .str.replace(r'^user-([a-z0-9]+)$', lambda m: '' if len(m.group(1)) % 2 == 1 or len(m.group(1)) > 7 else m.group(0), regex=True)
                .str.replace(r'^user-([a-zA-Z0-9가-힣\-._]+)$', r'\1', regex=True)
                .str.replace(r'[\-_][a-zA-Z0-9]{3,}$', '', regex=True)
                .str.replace(r'(.)\1+', r'\1\1', regex=True)
        )
        df['nickname'] = df['nickname'].str.replace(r'[^a-zA-Z가-힣0-9]+', '', regex=True)

        return df
    
    def _process_moum(self, df: pd.DataFrame):
        def process_shrink(match: re.Match, shrink_jung: str):
            base, trail = match.groups(1)
            if len(trail) == 1:
                return base + trail
            _, jung, jong = self.decompose_hangul(base)
            if jong == 0 and self.jung_list[jung] in shrink_jung:
                return base
            return base + sorted(trail, reverse=True)[0]
        
        def create_base(pattern: str):
            result = ''
            for char in pattern:
                if ord(char) > ord('ㅣ'):
                    continue
                result += ''.join(list(map(lambda x: self.compose_hangul(x[0], self.jung_list.index(char)), enumerate(self.cho_list))))
            return result

        patterns = {
            'ㅘㅏㅑ': 'ㅏ아',
            'ㅝㅓㅕ': 'ㅓ어',
            'ㅙㅞㅐㅒㅔㅖ': 'ㅐㅒㅔㅖ애얘에예',
            'ㅛ': 'ㅛㅗ요오',
            'ㅣ': 'ㅣ이',
            'ㅡ': 'ㅡ으'
        }

        for base, sub in patterns.items():
            df['comment'] = df['comment'].str.replace(rf"([{create_base(base)}])([{sub}]+)", lambda m: process_shrink(m, base), regex=True)

        return df
    
    def _process_yeocho_font(self, df: pd.DataFrame):
        def decompose_text(match: re.Match):
            base = match.group(1)
            cho, _, _  = self.decompose_hangul(base)
            cho = self.cho_list[cho]
            if cho == 'ㅍ':
                return 'ㅠㅠ'
            return cho + 'ㅠㅠ'
        
        yu_f_pattern = r"([후휴푸퓨쿠큐])[쿠큐푸퓨ㅜㅠ]+"
        
        df['comment'] = (
            df['comment']
                .str.replace(yu_f_pattern, lambda m: decompose_text(m), regex=True)
        )
        return df
    
    def _normalize_incorrect_grammar(self, df: pd.DataFrame):
        char_patterns = self.normalize_type['incorrect']['char']
        word_patterns = self.normalize_type['incorrect']['word']
        sentence_patterns = self.normalize_type['incorrect']['sentence']
        sentence_eval_patterns = self.normalize_type['incorrect']['sentence_eval']
        monde_patterns = self.normalize_type['incorrect']['monde']

        for column in df.columns:
            df[column] = (
                df[column]
                    .str.replace(self.word_compiled, lambda m: word_patterns[m.group(1)], regex=True)
                    .str.replace(self.char_compiled, lambda m: char_patterns[m.group(1)], regex=True)
            )

            for pattern, to_sub in monde_patterns:
                df[column] = df[column].str.replace(pattern, to_sub, regex=True)
            for pattern, to_sub in sentence_patterns:
                df[column] = df[column].str.replace(pattern, to_sub, regex=True)
            for pattern, to_sub_eval in sentence_eval_patterns:
                df[column] = df[column].str.replace(pattern, to_sub_eval, regex=True)
        return df
    
    def _set_default_nickname(self, df: pd.DataFrame):
        def _change_nickname(nickname: str):
            if len(nickname) == 0:
                return '[DEFAULT_NICK]'
            elif re.search(r'^[a-zA-Z0-9\-_.]+$', nickname):
                return '[DEFAULT_NICK]'
            elif re.search(r'[가-힣]', nickname) and len(nickname) < 3:
                return '[DEFAULT_NICK]'
            return nickname

        df['nickname'] = df['nickname'].apply(lambda x: _change_nickname(x) if isinstance(x, str) else x)
        return df
    
    def _remove_isolated_english(self, df: pd.DataFrame):
        df['nickname'] = df['nickname'].str.replace(r'(?<=[가-힣])([a-zA-Z])(?=[가-힣])(?!양|컵)', '', regex=True)
        return df
    
    def _remove_num_end_with(self, df: pd.DataFrame):
        df['nickname'] = df['nickname'].str.replace(r'\d+$', '', regex=True)
        return df
    
    def _sort_punct(self, df: pd.DataFrame):
        def process_sort(match: re.Match):
            text = ''.join(sorted(match.group(0), key=lambda c: punct_order.index(c)))

            text = re.sub(r'.+,+', '..', text)
            text = re.sub(r',{2,}', '..', text)
            text = re.sub(r'\.{2,}', '..', text)
            text = re.sub(r'\?{2,}', '?', text)
            text = re.sub(r'!{2,}', '!', text)
            text = re.sub(r'\~{2,}', '~', text)

            return text
        punct_order = '~.,?!'
        punct_pattern = r'[' + re.escape(punct_order) + r']{2,}'

        df['comment'] = df['comment'].apply(lambda x: re.sub(punct_pattern, lambda p: process_sort(p), x))
        df['comment'] = (
            df['comment']
                .replace(r'"{2,}', "'", regex=False)
                .replace(r'\'{2,}', "'", regex=False)
        )
        return df

    def decompose_hangul(self, c: str):
        code = ord(c) - 0xAC00
        cho = code // 588
        jung = (code % 588) // 28
        jong = code % 28
        return cho, jung, jong
    
    def compose_hangul(self, cho: int = 0, jung: int = 0, jong: int = 0):
        return chr(0xAC00 + cho * 588 + jung * 28 + jong)



if __name__ == "__main__":
    preprocess_logic = TextNormalizator()
    # with open('../tokens/emojis.txt', 'r', encoding='utf-8') as f:
    #     lines = [ line.strip() for line in f.readlines() ]
    # df = pd.DataFrame(lines, columns=['comment'])

    # df = _normalize_unicode(df)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)

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