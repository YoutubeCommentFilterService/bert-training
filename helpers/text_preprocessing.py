import pandas as pd
import unicodedata
from typing import Literal, Union, Dict, List, Tuple, TypedDict, Pattern, Callable
import re
import hangul_jamo
import json
from transformers import AutoTokenizer
import os

class IncorrectType(TypedDict):
    char: Dict[str, str]
    word: Dict[str, str]
    sentence: List[Tuple[str, str]]
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
    special_token: List[List[str]]

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
        
        # 일단 전처리
        df = self._normalize_unicode(df)
        df = self._clean_nickname(df)

        # 스팸 형식 제거
        df = self._normalize_incorrect_grammar(df)
        ## self._process_dan_mo_ja

        # 실행
        df = (
            self._sort_punct(df)
                .pipe(self._replace_special_tokens)
                .pipe(self._replace_misc_patterns)
                .pipe(self._replace_structed_patterns)
                .pipe(self._cleanup_formatting)
                .pipe(self._clean_duplicated_token)
                .pipe(self._sort_hangul)
        )
        df = (
            self._remove_isolated_english(df)
                .pipe(self._set_default_nickname)
        )
        
        # 마지막으로 한번 더 정리
        df = self._normalize_incorrect_grammar(df)

        # special token 복원용
        for column in df.columns:
            df[column] = df[column].str.replace(r'(\[[a-zA-Z_]+\])', lambda m: m.group(1).upper(), regex=True)
        return df

    def reload(self):
        with open(self.path['normalize'], 'r', encoding='utf-8') as f:
            self.normalize_type: TextPreprocessingType = json.load(f)

        char_patterns = self.normalize_type['incorrect']['char']
        word_patterns = self.normalize_type['incorrect']['word']

        self.char_compiled = re.compile(r'(' + '|'.join(re.escape(k) for k in char_patterns.keys()) + r')')
        self.word_compiled = re.compile(r'(' + '|'.join(re.escape(k) for k in word_patterns.keys()) + r')')

        for idx, item in enumerate(self.normalize_type['incorrect']['sentence']):
            self.normalize_type['incorrect']['sentence'][idx] = [re.compile(item[0]), item[1]]

        for idx, item in enumerate(self.normalize_type['incorrect']['sentence_eval']):
            self.normalize_type['incorrect']['sentence_eval'][idx] = [re.compile(item[0]), eval(item[1])]

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
        normalized = hangul_jamo.compose(normalized)
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

        text = ' ' .join(result)
        text = re.sub(r'([\(\[\{]) | ([\)\]\}])', lambda m: m.group(1) or m.group(2), text)
        return text
    
    def _normalize_unicode(self, df: pd.DataFrame):
        df['comment'] = (
            df['comment'] # \u2640\u2642\u2695\u2696\u2708\u2764 - ♀, ♂, ⚕, ⚖, ✈, ❤ 기호
                .str.replace(r'[\u2002\u2003\u2007\u2008]+', ' ', regex=True)
                .str.replace(r'[\U0001F3FB-\U0001F3FF\uFE0F\u0674\u1160\u200B\u200C\u200D\uFEFF\u2060\u1160]+', '', regex=True)
                .str.replace(r'\*+', '', regex=True)
                .str.replace('9글', '구글')
        )
        for key, compiled_pattern in self.normalize_type['punct'].items():
            df['comment'] = df['comment'].str.replace(compiled_pattern, key, regex=True)
        
        ## 유니코드 문자 정규화 과정
        df['comment'] = (
            df['comment']
                .str.replace(r'([ㄱ-ㅎㅏ-ㅣ])\s+', r'\1', regex=True)
                .apply(lambda x: self._normalize_unicode_text(x) if isinstance(x, str) else x)
        )
        unicode_alphabet_dict = self.normalize_type['single']['en']
        for char, compiled_pattern in unicode_alphabet_dict.items():
            df['comment'] = df['comment'].str.replace(compiled_pattern, char, regex=True)
        df['comment'] = df['comment'].str.replace(r'(?i)[cㄷ][o0ㅇ]m', 'com', regex=True)
        df['comment'] = df['comment'].apply(lambda x: ''.join(ch for ch in x if not ('\u4E00' <= ch <= '\u9FFF')))
            
        ## 특정 포맷으로 적은 한글 문자열을 변환
        JUNG_MAPPING = {
            'H': 'ㅐ', 'I': 'ㅣ', 'l': 'ㅣ'
        }
        def combine_jamos(match: re.Match):
            cho, jung = match.groups()
            cho_idx = self.cho_list.index(cho) if cho in self.cho_list else -1
            jung_char = JUNG_MAPPING.get(jung)
            if cho_idx == -1 or jung_char not in self.jung_list:
                return cho + jung  # 조합 불가한 건 그대로
            jung_idx = self.jung_list.index(jung_char)
            return chr(0xAC00 + cho_idx * 588 + jung_idx * 28)
        
        df['comment'] = df['comment'].str.replace(r'([ㄱ-ㅎ])[ ,\\]*([A-Za-z])', combine_jamos, regex=True)
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
        def remove_hyphen_or_underscore_format(text: str) -> str:
            pattern = r'[\-_][a-zA-Z0-9]{2,7}$'

            while matched := re.search(pattern, text):
                text = text[:matched.span(0)[0]]

            return text
        
        df['nickname'] = (
            df['nickname']
                .str.strip()
                .str.replace(r'^@', '', regex=True)
                .str.replace(r'[^a-zA-Z가-힣0-9\-._]+', '', regex=True)
                .str.replace(r'^user-([a-z0-9]+)$', lambda m: '' if len(m.group(1)) % 2 == 1 or len(m.group(1)) > 7 else m.group(0), regex=True)
                .str.replace(r'^user-([a-zA-Z0-9가-힣\-._]+)$', r'\1', regex=True)
                .apply(lambda x: remove_hyphen_or_underscore_format(x) if isinstance(x, str) else x)
        )
        df['nickname'] = df['nickname'].str.replace(r'[^a-zA-Z가-힣0-9]+', '', regex=True)

        return df
    
    def _normalize_incorrect_grammar(self, df: pd.DataFrame):
        char_patterns = self.normalize_type['incorrect']['char']
        word_patterns = self.normalize_type['incorrect']['word']
        sentence_patterns = self.normalize_type['incorrect']['sentence']
        sentence_eval_patterns = self.normalize_type['incorrect']['sentence_eval']

        for column in df.columns:
            df[column] = (
                df[column]
                    .str.replace(self.word_compiled, lambda m: word_patterns[m.group(1)], regex=True)
                    .str.replace(self.char_compiled, lambda m: char_patterns[m.group(1)], regex=True)
            )

        error_flag = None
        for column in df.columns:
            try:
                for pattern, to_sub in sentence_patterns:
                    try:
                        df[column] = df[column].str.replace(pattern, to_sub, regex=True)
                    except Exception as e:
                        error_flag = True
                        print(pattern, to_sub)
                for pattern, to_sub_eval in sentence_eval_patterns:
                    try:
                        df[column] = df[column].str.replace(pattern, to_sub_eval, regex=True)
                    except Exception as e:
                        error_flag = True
                        print(pattern, to_sub)
            except Exception as e:
                error_flag = True
                print(pattern, to_sub)
        if error_flag:
            exit(0)
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
    
    def _sort_hangul(self, df: pd.DataFrame):
        def process_jongsung(match: re.Match):
            base, trail = match.groups()
            cho, jung, jong = self.decompose_hangul(base)
            # 종성은 None이 될 수도 있다
            if self.jong_list[jong] != None and self.jong_list[jong] in chosung_order:
                new_base = self.compose_hangul(cho, jung, 0)
                new_trail = self.jong_list[jong] + trail
                return new_base + new_trail
            return ''.join(match.groups())
        
        def process_jungsung(match: re.Match):
            base, trail = match.groups()
            cho, jung, _ = self.decompose_hangul(base)
            if self.jung_list[jung] in jungsung_order:
                new_base = self.cho_list[cho]
                new_trail = self.jung_list[jung] + trail
                return new_base + new_trail
            return ''.join(match.groups())

        def process_sort(match: re.Match):
            sort_order = chosung_order + jungsung_order
            text = ''.join(sorted(match.group(0), key=lambda x: sort_order.index(x)))
            text = re.sub(r'ㅋ[ㄲㅍ]+', 'ㅋ', text)
            return text
        
        chosung_order = 'ㄱㅋㄲㅍㅎ'
        jungsung_order = 'ㅠㅜ'
        jongsung_pattern = r'([가-힣])([' + chosung_order + ']+)'
        jungsung_pattern = r'([가-힣])([' + jungsung_order + ']+)'
        sort_pattern = r'[' + chosung_order + 'ㅠㅜ' + r']{2,}'

        df['comment'] = (
            df['comment']
                .str.replace(r'[퓨푸][ㅠㅜ]+', 'ㅠㅠ', regex=True)
                .apply(lambda x: re.sub(jungsung_pattern, lambda m: process_jungsung(m), x))
                .apply(lambda x: re.sub(jongsung_pattern, lambda m: process_jongsung(m), x))
                .apply(lambda x: re.sub(sort_pattern, lambda m: process_sort(m), x))
        )
        df['comment'] = df['comment'].str.replace(r'(.)\1{2,}', r'\1\1', regex=True)
        return df
    
    def _process_dan_mo_ja(self, df: pd.DataFrame):
        def process_iya(match: re.Match):
            trail = ''.join(sorted(match.group(2), key=lambda t: iya_order.find(t)))
            trail = re.sub(r'(.)\1{2,}', r'\1', trail)
            trail = re.sub('아ㅏ', '아', trail)
            trail = re.sub('야ㅑ', '야', trail)

            base_cho, base_jung, base_jong = self.decompose_hangul(match.group(1))
            if base_jong != 0:
                return match.group(1) + trail
            if self.jung_list[base_jung] == 'ㅣ' and re.match(r'[야아ㅑㅏ]', trail):
                return self.compose_hangul(base_cho, self.jung_list.index('ㅑ'))
            if self.jung_list[base_jung] in 'ㅏㅑㅣㅡ':
                return match.group(1)
            return match.group(1) + trail
        
        def process_wawu(match: re.Match):
            trail = ''.join(sorted(match.group(2), key=lambda t: wawu_order.find(t)))
            
            trail = re.sub(r'(.)\1{2,}', r'\1', trail)
            for order in range(0, len(wawu_order), 2):
                o = wawu_order[order: order+2]
                trail = re.sub(o, o[0], trail)
            
            cho, jung, jong = self.decompose_hangul(match.group(1))
            if jong > 0:
                return match.group(1) + trail
            if self.jung_list[jung] in 'ㅝㅓ' and re.search(r'[워ㅝ어ㅓ]', match.group(2)):
                return match.group(1)
            if self.jung_list[jung] == 'ㅘ' and re.search(r'[아ㅏ]', match.group(2)):
                return match.group(1)
            if self.jung_list[jung] == 'ㅞ' and re.search(r'[에ㅔ애ㅐ어ㅓ]', match.group(2)):
                return match.group(1)
            return match.group(1) + trail

        def process_uncommon_jaum(match: re.Match):
            dt = {
                '7': 'ㄱ',
                '^': 'ㅅ',
                '[': 'ㄷ',
                '77': 'ㄲ',
                '^^': 'ㅆ',
                '[[': 'ㄸ',
            }
            base = re.sub(r'(.)\1{2,}', r'\1\1', match.group(1))
            return self.compose_hangul(self.cho_list.index(dt[base]), self.jung_list.index(match.group(2)))
        
        iya_order = '이ㅣ아ㅏ야ㅑ'
        iya_pattern = rf'([가-힣])([{iya_order}]+)'
        wawu_order = '와ㅘ왜ㅙ외ㅚ워ㅝ웨ㅞ위ㅟ아ㅏ야ㅑ어ㅓ여ㅕ오ㅗ요ㅛ우ㅜ유ㅠ으ㅡ이ㅣ아ㅗ'
        wawu_pattern = rf'([가-힣])([{wawu_order}]+)'
        uncommon_jaum_pattern = r'([7\^\[]+)([ㅏ-ㅣ])'

        df['comment'] = (
            df['comment']
                .apply(lambda x: re.sub(uncommon_jaum_pattern, lambda m: process_uncommon_jaum(m), x))
                # .apply(lambda x: re.sub(iya_pattern, lambda m: process_iya(m), x))
                .apply(lambda x: re.sub(wawu_pattern, lambda m: process_wawu(m), x))
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