def save_pretty_json(dict_data: dict):
    output_str = ''
    for key, value in dict_data.items():
        dict_data = dict(sorted(value.items(), key=lambda x: (-len(x[0]), x[0][0])))
        output_str += f'\t\t"{key}": {{\n'
        iloc = 0
        for idx, (key, value) in enumerate(dict_data.items()):
            if iloc == 0:
                output_str += '\t\t\t'
            write_data = f'"{key}": "{value}"{", " if idx != len(dict_data) - 1 else ""}'
            iloc += len(write_data)
            output_str += write_data
            if iloc >= 65:
                output_str += '\n'
                iloc = 0
        if iloc != 0:
            output_str += '\n'
        output_str += '\t\t},\n'

    with open('pretty_formated_output.json', 'w', encoding='utf-8') as f:
        f.write(output_str.rstrip())

import json
with open('../tokens/text_preprocessing.json', 'r', encoding='utf-8') as f:
    obj = json.load(f)

incorrect = obj['incorrect']
data = {
    'char': incorrect['char'],
    'word': incorrect['word'],
}
save_pretty_json(data)