import pandas as pd
import unicodedata
from typing import Union
import re

def normalize_unicode_text(text: str) -> str:
    unicode_single_hangul_dict = {
        'ᄀ': 'ㄱ', 'ᄂ': 'ㄴ', 'ᄃ': 'ㄷ', 'ᄅ': 'ㄹ', 'ᄆ': 'ㅁ', 'ᄇ': 'ㅂ', 
        'ᄉ': 'ㅅ', 'ᄋ': 'ㅇ', 'ᄌ': 'ㅈ', 'ᄎ': 'ㅊ', 'ᄏ': 'ㅋ', 'ᄐ': 'ㅌ', 
        'ᄑ': 'ㅍ', 'ᄒ': 'ㅎ', 'ᄍ': 'ㅉ', 'ᄄ': 'ㄸ', 'ᄁ': 'ㄲ', 'ᄊ': 'ㅆ', 
        'ᅡ': 'ㅏ', 'ᅣ': 'ㅑ', 'ᅥ': 'ㅓ', 'ᅧ': 'ㅕ', 'ᅩ': 'ㅗ', 'ᅭ': 'ㅛ', 
        'ᅮ': 'ㅜ', 'ᅲ': 'ㅠ', 'ᅳ': 'ㅡ', 'ᅵ': 'ㅣ', 'ᅢ': 'ㅐ', 'ᅦ': 'ㅔ', 
        'ᅴ': 'ㅢ', 'ᆪ': 'ㄱㅅ', 'ᆬ': 'ㄴㅈ', 'ᆭ': 'ㄴㅎ', 'ᆲ': 'ㄹㅂ', 
        'ᆰ': 'ㄹㄱ', 'ᆳ': 'ㄹㅅ', 'ᆱ': 'ㄹㅁ', 'ᄚ': 'ㄹㅎ', 'ᆴ': 'ㄹㅌ', 
        'ᆵ': 'ㄹㅍ', 'ᄡ': 'ㅂㅅ', 'ᄈ': 'ㅂㅂ',
        '𐨛': 'ㅋ', 'ヲ': 'ㅋ'
    }
    visual_map = {
        'a': '[ᴀ𝗮𝘢𝙖𝓪αаＡａ𝖺𝓐]',
        'b': '[ᵇ𝒷𝗯𝙗𝓫𝖇ʙＢｂ𝑏]',
        'c': '[ⅽᴄϲⲥϲⲤ¢匚𐰽ᏟⅭℂⅽＣ∁ｃ𝖼𝑐𝒸𝓬ꞓ]',
        'd': '[𝖽𝑑ⅾⅆｄ𝐝𝗱𝙙𝒅𝒟𝔡𝕕]',
        'e': '[ｅ𝐞𝗲𝙚𝑒𝒆𝓮𝖾℮𝔢𝕖𝕰еε]',
        'f': '[𝒇𝒻𝓯𝖿𝕗𝐟𝗳𝙛ｆ]',
        'g': '[𝗀𝓰𝙜𝐠𝑔𝒈𝓰𝖌ｇ]',
        'h': '[𝐡𝗵𝙝𝑯𝒉𝓱𝖍ｈ]',
        'i': '[𝐢𝗶𝙞𝑖𝒊𝓲𝖎ｉ]',
        'j': '[𝐣𝗷𝙟𝑗𝒋𝓳𝖏ｊ]',
        'k': '[𝐤𝗸𝙠𝑘𝒌𝓴𝖐ｋ]',
        'l': '[𝐥𝗹𝙡𝑙𝒍𝓵𝖑ⅼｌ]',
        'm': '[ⅿ𝗺𝙢𝑚𝒎𝓶𝖒𝕞ｍⲘΜм]',
        'n': '[𝗻𝙣𝑛𝒏𝓷𝖓ｎ𝗇𝐧]',
        'o': '[οОＯ〇ｏ𝑜𝗈𝗼𝙤𝓞𝓸𝖔ⲟⓞⵔꝋ]',
        'p': '[𝐩𝗽𝙥𝑝𝒑𝓹𝖕ｐ𝕡ρр]',
        'q': '[𝐪𝗾𝙦𝑞𝒒𝓺𝖖ｑ]',
        'r': '[𝐫𝗿𝙧𝑟𝒓𝓻𝖗ｒ𝕣ꞃ]',
        's': '[𝐬𝗿𝙨𝑠𝒔𝓼𝖘ｓ𝕤ꜱ]',
        't': '[𝐭𝗍𝙩𝑡𝒕𝓽𝖙ｔ𝕥ꞇ]',
        'u': '[𝐮𝗎𝙪𝑢𝒖𝓾𝖚ｕ]',
        'v': '[𝐯𝗏𝙫𝑣𝒗𝓿𝖛ｖ]',
        'w': '[𝐰𝗐𝙬𝑤𝒘𝔀𝖜ｗ]',
        'x': '[𝐱𝗑𝙭𝑥𝒙𝓍𝖝ｘ𝕩х]',
        'y': '[𝐲𝗒𝙮𝑦𝒚𝓎𝖞ｙ𝕪у]',
        'z': '[𝐳𝗓𝙯𝑧𝒛𝔃𝖟ｚ𝕫ᴢ]',
    }
    
    normalized = unicodedata.normalize("NFKC", text)
    normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
    for char, pattern in visual_map.items():
        normalized = re.sub(pattern, char, normalized)
    normalized = re.sub(r'[cC][o0O][mM]', 'com', normalized)
    normalized = ''.join(ch for ch in normalized if not ('\u4E00' <= ch <= '\u9FFF'))

    return ''.join(unicode_single_hangul_dict.get(ch, ch) for ch in normalized)

def normalize_korify(text: str):
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

def normalize_tlettak_font(text: str, 
                           space_pattern: Union[str, re.Pattern] = r'\s*[\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\(\)\[\]\{\}\-+=~,.\/<>;:\'"]+[\s!?@.,❤]*', 
                           search_pattern: Union[str, re.Pattern] = r'\b([\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\-+=~,.\/<>;:\'"]{1}\b)([\s!?\^@.,ㅣ~❤]+)(\b[\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\-+=~,.\/<>;:\'"]{1}\b)'
                           ) -> str:
    
    space_pattern = re.compile(space_pattern) if isinstance(space_pattern, str) else space_pattern
    search_pattern = re.compile(search_pattern) if isinstance(search_pattern, str) else search_pattern

    text = re.sub(r'([\(\[\{])', r'\1 ', text)  # 여는 괄호
    text = re.sub(r'([\)\]\}])', r' \1', text)  # 닫는 괄호

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

# 유니코드 정규화
def _normalize_unicode(df: pd.DataFrame):
    df['comment'] = (
        df['comment']
            .str.replace(r'[\u0020\u200b\u2002\u2003\u2007\u2008\u200c\u200d]+', ' ', regex=True)
            .str.replace(r'[\U0001F3FB-\U0001F3FF\uFE0F]', '', regex=True)
            .str.replace(r'\*+', '', regex=True)
            .apply(lambda x: replace_unicode_punctuation(x) if isinstance(x, str) else x)
            .apply(lambda x: normalize_unicode_text(x) if isinstance(x, str) else x)
            .apply(lambda x: normalize_korify(x) if isinstance(x, str) else x)
    )
    df['comment'] = df['comment'].str.replace(r'(?<!\d)(.)\1{2,}', r'\1\1', regex=True)
    return df

# 유니코드 문장부호 변환
def replace_unicode_punctuation(text: str) -> str:
    unicode_punctuation_map = {
        '¡': '!', '！': '!',
        '¿': '?', '？': '?',
        '‘': "'", '’': "'", '＇': "'",
        '“': '"', '”': '"', '＂': '"',
        'ㆍ': '.', '·': '.', '・': '.', '•': '.', '．': '.',
        '，': ',',
        'ᆢ': '..', '…': '...',
        '：': ':', '；': ';',
        '（': '(', '）': ')',
        '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-', '―': '-',
    }
    return ''.join(unicode_punctuation_map.get(ch, ch) for ch in text)

def _replace_special_tokens(df: pd.DataFrame, emoji_path: str):
    with open(emoji_path, 'r', encoding='utf-8') as f:
        emojis = [line.strip() for line in f.readlines()]
    emoji_pattern = '|'.join(map(re.escape, emojis))
    df['comment'] = (
        df['comment']
            # url 전처리
            .str.replace(r'https?:\/\/(?:[a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\-]+\.)+(?:[a-zA-Z0-9가-힣]{2,})(?::\d+)?(?:\/[^\s]*)?', '[URL]', regex=True)
            # email 전처리
            .str.replace(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', regex=True)
            # tag 전처리
            .str.replace(r'@+[\w가-힣\.\-]*', '[TAG]', regex=True)
            # 해시태그 전처리
            .str.replace(r'#[\w가-힣.ㄱ-ㅎㅏ-ㅣ-]+', '[HASH_TAG]', regex=True)
            # 타임스탬프 전처리
            .str.replace(r'\d+:[0-5]\d:[0-5]\d\s*초?', '[TIMESTAMP]', regex=True)
            .str.replace(r'[0-5]?\d+:[0-5]\d\s*초?', '[TIMESTAMP]', regex=True)
            # 비율 전처리. 화면 비율이든 과실 비율이든
            .str.replace(r'\d+:\d+', '[RATIO]', regex=True)
            # 텍스트 기반 이모지 전처리
            .str.replace(emoji_pattern, '[TEXT_EMOJI]', regex=True)
            # 개추, 추천 요청 전처리
            .str.replace(r'([👆👈👇✋👍])', '[THUMB]', regex=True)
            # 화살표 전처리
            .str.replace(r'([➡⬅⬇⍐↗↘↖↙→←↑↓⇒⏫🔙]|[\-=]+>+|<+[\-=]+)', '[ARROW]', regex=True)
            # 하트 이모지 전처리
            .str.replace(r'[💚💛🩷🩶💗💖❤🩵🖤💘♡♥🧡🔥💕️🤍💜🤎💙]', '[HEART]', regex=True)
            # 축하 이모지 전처리
            .str.replace(r'🎉', '[CONGRAT]', regex=True)
            # yes, no 이모지 전처리
            .str.replace(r'❌', '[NO]', regex=True)
            .str.replace(r'[⭕️✅]', '[YES]', regex=True)
    )
    return df

def _cleanup_formatting(df: pd.DataFrame):
    df['comment'] = (
        df['comment']
            # 소숫점, 1000단위 변환
            .str.replace(r'(?<=\d)\.(?=\d)', '[POINT_DOT]', regex=True)
            .str.replace(r'(?<=\d),(?=\d)', '[POINT_COM]', regex=True)
            # 문장부호 앞의 공백 제거 및 뒤에 공백 추가
            .str.replace(r'\s*([.,?!^]+)\s*', r'\1 ', regex=True)
            # 쓸데없이 많은 공백 제거
            .str.replace(r'\s{2,}', ' ', regex=True)
            # 소숫점, 1000단위 복원
            .str.replace(r'\[POINT_DOT\]', '.', regex=True)
            .str.replace(r'\[POINT_COM\]', ',', regex=True)
    )
    return df

def _replace_structed_patterns(df: pd.DataFrame):
    date_patterns = [
        r'\d{1,4}\s*년(?:\s*\d{1,2}\s*월)?(?:\s*\d{1,2}\s*일)?',
        r'\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?',
        r'\d{1,2}\s*일'
    ]
    time_patterns = [
        r'\d{1,2}\s*시간?(?:\s*\d{1,2}\s*분)?(?:\s*\d{1,2}\s*초)?', # 시 + 분 + 초
        r'\d{1,2}\s*분(?:\s*\d{1,2}\s*초)?', # 분 + 초
        r'\d{1,2}\s*초' # 초
    ]
    float_patterns = [r'\d+\.\d+']
    number_patterns = [r'\d{1,3}(?:,\d{3})+', r'\d+']
    duration_patterns = [
        r'\[TIME(?:STAMP)?\][\s]*[~-][\s]*\[TIME(?:STAMP)?\]',
    ]

    patterns = [
        [date_patterns, '[DATE]'],
        [time_patterns, '[TIME]'],
        [float_patterns, '[FLOAT]'],
        [number_patterns, '[NUMBER]'],
        [duration_patterns, '[DURATION]']
    ]

    for [pattern, token] in patterns:
        for p in pattern:
            df['comment'] = df['comment'].str.replace(p, token, regex=True)
    return df

def _replace_misc_patterns(df: pd.DataFrame):
    df['comment'] = (
        df['comment']
            .str.replace(r'\[+', '[', regex=True)
            .str.replace(r'\]+', ']', regex=True)
            .str.replace(r'[^\w가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\(\)\[\]\{\}\-+=~,.\/<>;:\'"\s]', '', regex=True)
            .str.replace(r'(?<!\d)([a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ!?%\^\(\)\[\]\{\}\-_+=~,.\/<>;:\'"\s])\1{3,}', r'\1\1', regex=True)
            .str.strip()
            .fillna('[EMPTY]')
            # 한글자 + 부호 + 한글자 패턴 처리
            .apply(lambda x: normalize_tlettak_font(x) if isinstance(x, str) else x)
    )
    return df

def _clean_duplicated_token(df: pd.DataFrame):
    tags = ['TIMESTAMP', 'URL', 'EMAIL', 'TAG', 'HASH_TAG', 'THUMB', 'ARROW', 'TEXT_EMOJI', 'HEART', 'CONGRAT', 'DATE', 'TIME', 'FLOAT', 'NUMBER']
    for tag in tags:
        pattern = r'(?:\[' + re.escape(tag) + r'\]\s*)+'
        df['comment'] = df['comment'].apply(lambda x: re.sub(pattern, f'[{tag}]', x) if isinstance(x, str) else x)
    return df

# 닉네임 정제
def _clean_nickname(df: pd.DataFrame):
    # user로 시작하는 경우 체크
    def _change_if_nickname_is_default(text: str):
        # 태그로 붙는 맨 뒤의 - 의 경우, 뒤에 대문자가 "전혀" 오지 않는다.
        return re.sub(
            r'^user-([a-z0-9]+)$',
            lambda m: '[DEFAULT_NICK]' if len(m.group(1)) % 2 == 1 or len(m.group(1)) > 7 else m.group(0),
            text
        )
    
    def _change_if_starts_with_user(text: str):
        return re.sub(
            r'^user[-_.]([a-zA-Z0-9가-힣-._]+)$',
            r'\1',
            text
        )
    
    def _remove_if_hyphen_and_odd_word(text: str):
        # group(1): 캡처된 그룹
        # group(0): pattern 그 자체
        pattern = r'-([a-z0-9]+)$'
        before_text = text
        while re.search(pattern, text):
            text = re.sub(
                pattern, 
                lambda m: '' if len(m.group(1)) % 2 == 1 or len(m.group(1)) > 7 else m.group(0), 
                text
            )
            if text == before_text:
                break
            before_text = text

        return text
    
    def _normalize_nickname(text: str):
        if text == '[DEFAULT_NICK]':
            return text
        text = re.sub(r'[-._]', '', text)
        text = re.sub(r'[^a-zA-Z가-힣0-9]+', '', text)
        return text
    
    df['nickname'] = (
        df['nickname']
            .str.strip()
            .str.replace(r'^@', '', regex=True)
            .apply(lambda x: _change_if_nickname_is_default(x) if isinstance(x, str) else x)
            .apply(lambda x: _change_if_starts_with_user(x) if isinstance(x, str) else x)
            .apply(lambda x: _remove_if_hyphen_and_odd_word(x) if isinstance(x, str) else x)
            .apply(lambda x: _normalize_nickname(x) if isinstance(x, str) else x)
    )
    return df

# 닉네임 정규화
def _normalize_spam_nickname(df: pd.DataFrame):
    df['nickname'] = (
        df['nickname']
            .str.replace(r'[1Iil]9', '19', regex=True)
            .str.replace(r'[1Iil]9(?:x|X|금)', '19금', regex=True)
            .str.replace(r'[ㅇoO0]F([가-힣])', r'야\1', regex=True)
            .str.replace(r'(?:야|얏|얃)\w*(?:동|둉|덩|뎡|둥|듕)', '야동', regex=True)
            .str.replace(r'얃\w*(?:옹|용|엉|영|웅|융)', '야동', regex=True)
            .str.replace(r'(?:채|체|챼|쳬)(?:널|녈|놀|뇰|눌|뉼)', '채널', regex=True)
            .str.replace(r'(?:챈|첸|첀|쳰)(?:얼|열|올|욜|울|율)', '채널', regex=True)
            .str.replace(r'(?:프|푸)(?:사|샤)', '프사', regex=True)
            .str.replace(r'카g|카G', '카지', regex=True)
            .str.replace(r'갸입', '가입', regex=True)
            .str.replace(r'V[1l]P', 'VIP', regex=True)
            .str.replace(r'(?:온|On|on|ON)(?:팬|Fan|fan|FAN)', '온팬', regex=True)
            .str.replace(r'(?:뮨|문|무|뮤)(?:의|늬|희)', '문의', regex=True)
            .str.replace(r'(?:눌|뉼)(?:러|려)', '눌러', regex=True)
            .str.replace(r'클맄', '클릭', regex=True)
            .str.replace(r'(?:쿨|끌)(?:릭|맄)', '클릭', regex=True)
            .str.replace(r'(?:꾸|뀨)(?:욱|육)|뀩', '꾹', regex=True)
            .str.replace(r'(?:샤|ㅅF)고', '사고', regex=True)
            .str.replace(r'후뱡', '후방', regex=True)
            .str.replace(r'쥬소', '주소', regex=True)
            .str.replace(r'꾤', '꼴', regex=True)
            .str.replace(r'계졍', '계정', regex=True)
    )
    return df

def _remove_isolated_english(df: pd.DataFrame):
    df['nickname'] = (
        df['nickname']
            .str.replace(r'(?<=[가-힣])([a-zA-Z])(?=[가-힣])(?!양)', '', regex=True)
    )
    return df

def _set_default_nickname(df: pd.DataFrame):
    def _change(nickname: str):
        if re.search(r'[가-힣]', nickname):
            if len(nickname) < 3:
                return '[DEFAULT_NICK]'
        else:
            if len(nickname) < 5:
                return '[DEFAULT_NICK]'
        return nickname

    df['nickname'] = df['nickname'].apply(lambda x: _change(x) if isinstance(x, str) else x)
    return df

def run_text_preprocessing(df: pd.DataFrame, emoji_path: str):
    df = (
        _normalize_unicode(df)
            .pipe(_replace_special_tokens, emoji_path)
            .pipe(_replace_structed_patterns)
            .pipe(_cleanup_formatting)
            .pipe(_replace_misc_patterns)
            .pipe(_clean_duplicated_token)
    )
    df = (
        _clean_nickname(df)
            .pipe(_normalize_spam_nickname)
            .pipe(_remove_isolated_english)
            .pipe(_set_default_nickname)
    )
    return df

def replace_regex_predict_data(df: pd.DataFrame):
    pattern_spacer = '=!?@'
    space_pattern = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9:]+[\s!?@.,❤]*')
    pattern = re.compile(rf"[{pattern_spacer}]*(\w)([{pattern_spacer}\s.,❤]+)(\w)")
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
    
    with open('../tokens/emojis.txt', 'r', encoding='utf-8') as f:
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
        .str.replace(r'([➡⬇↗↘↖↙⏫🔙→←↑↓⇒]|[\-\=]+>|<[\-\=]+)', '[ARROW]', regex=True)\
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

    return df