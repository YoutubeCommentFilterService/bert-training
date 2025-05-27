def save_pretty_json(dict_data: dict, filename: str, max_per_line: int = 70):
    dict_data = dict(sorted(dict_data.items(), key=lambda x: (-len(x[0]), x[0][0])))
    print(dict_data)
    filename = filename if filename.endswith('.json') else filename + '.json'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('{\n')
        iloc = 0
        for idx, (key, value) in enumerate(dict_data.items()):
            if iloc == 0:
                f.write('\t\t\t')
            write_data = f'"{key}": "{value}"'
            iloc += len(write_data)
            if idx != len(dict_data) - 1:
                iloc += len(", ")
                write_data += ", "
            f.write(write_data)
            if iloc >= max_per_line:
                f.write('\n')
                iloc = 0
        if iloc != 0:
            f.write('\n')
        f.write('\t\t}')

char = {}
word = {}


save_pretty_json(char, 'test_1.json')
save_pretty_json(word, 'test_2.json')