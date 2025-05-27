import json
import re
import os

example_list_file = 'example_list.txt'
if not os.path.exists(example_list_file):
    example_list_json = list(filter(lambda x: re.search(r'.json$', x), os.listdir('.')))
    try:
        example_list_json.remove('mungchi.py')
    except Exception as e:
        pass
    try:
        example_list_json.remove(example_list_file)
    except Exception as e:
        pass

    datas = []

    sensetypes = set()

    for example in example_list_json:
        with open(example, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = data['channel']['item']

            for item in items:
                senseinfo = item.get('senseinfo', {})
                sensetypes.add(senseinfo.get('type', ''))
                if senseinfo.get('type', '') != '일반어':
                    continue
                example_info = senseinfo.get('example_info', [])
                for example in example_info:
                    text = example['example']
                    text = re.sub(r'[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]', '', text)
                    text = re.sub(r'\(\)', '', text)
                    text = re.sub(r'[‘’]', "'", text)
                    text = re.sub(r'[“”]', '"', text)
                    text = re.sub('…', '..', text)
                    text = re.sub(r',.+', '..', text)
                    text = re.sub(r'[\{\}]', '', text)
                    text = re.sub(r'</?[A-Za-z]+>', '', text)
                    text = re.sub(r'[＜≪≫＞]', "'", text)
                    text = re.sub(r'\<([가-힣\s\d,.]+)\>', r"'\1'", text)
                    text = re.sub(r'<', '', text)
                    text = re.sub(r'(.)\1{2,}', r"\1\1", text)
                    text = re.sub(r'"(.+?)"', r'[Q]\1[Q]', text)
                    text = re.sub('"', '', text)
                    text = re.sub(r'\[Q\]', '"', text)
                    text = re.sub(r'[가-힣]+\(([\w\s.]+)\)', r"\1", text)
                    text = re.sub(r'\([\d%~?\.]+\)', r'', text)
                    text = re.sub(r'\.+', ' ', text)
                    text = re.sub("'", '', text)
                    datas.append(text.strip())

    datas = dict.fromkeys(datas)

    with open(example_list_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(datas))

else:
    with open(example_list_file, 'r', encoding='utf-8') as f:
        datas = [ line.strip() for line in f.readlines() ]

pattern = r'앋'
print(list(filter(lambda x: re.search(pattern, x), datas)))
trail_patterns = set()
base_patterns = set()
one_patterns = set()
trail_pattern = re.compile(pattern + r'([가-힣])')
base_pattern = re.compile(r'([가-힣])' + pattern)
one_pattern = re.compile(r'\b' + pattern + r'\b')
for data in datas:
    for token in trail_pattern.findall(data):
        trail_patterns.add(token)
    for token in base_pattern.findall(data):
        base_patterns.add(token)
    for token in one_pattern.findall(data):
        one_patterns.add(token)

print(trail_patterns)
print(base_patterns)
print(one_patterns)
