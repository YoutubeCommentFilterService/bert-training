from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('../model/tokenizer')
tokens = []

with open('../tokens/common_shortcut_tokens.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
tokens.extend(list(dict.fromkeys([ token.strip() for token in lines if token ])))

with open('../tokens/common_word_tokens.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
tokens.extend(list(dict.fromkeys([ token.strip() for token in lines if token ])))

with open('../tokens/is_common_token_in.txt', 'w', encoding='utf-8') as f:
    for token in tokens:
        tokenized = tokenizer.tokenize(token)
        if len(tokenized) == 1 and tokenized[0] != '[UNK]':
            continue
        f.write(f"{token} {tokenized}\n")