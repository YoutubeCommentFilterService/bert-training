def sort_file(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        ori_lines = [ line.strip() for line in f if line.strip() ]
        lines = list(sorted(set(ori_lines)))
    
    print(len(ori_lines), len(lines))

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

sort_file('./tokens/common_word_tokens.txt')
sort_file('./tokens/common_shortcut_tokens.txt')
sort_file('./tokens/common_en_tokens.txt')
sort_file('./tokens/common_monde_tokens.txt')
