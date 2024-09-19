import json
import csv
from owlready2 import *

def process_json_to_csv(json_file_path, csv_file_path, fma_ontology):
    # 读取JSON文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    entities = []

    # 递归函数来处理嵌套的字典
    def extract_entities(obj):
        if isinstance(obj, dict):
            if 'info' in obj:
                info = obj['info']
                if 'name' in info:
                    entities.append(info['name'])
                if 'relations' in info:
                    for relation_type in info['relations']:
                        for entity in info['relations'][relation_type]:
                            if isinstance(entity, str):
                                entities.append(entity)
            if 'children' in obj:
                extract_entities(obj['children'])
            for key, value in obj.items():
                if key not in ['info', 'children']:
                    extract_entities(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_entities(item)

    # 处理JSON数据
    extract_entities(data)

    # 去重
    entities = list(set(entities))

    # 获取IRI并写入CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Entity', 'IRI'])  # 写入表头


        for entity in entities:
            try:
                iri = fma_ontology.search(preferred_name=entity)[0]
                writer.writerow([entity, iri])
            except Exception as e:
                try:
                    entity = fma_ontology.search_one(iri = f"*{entity}")
                    name = entity.label.first() if entity.label else entity.name
                    writer.writerow([name, iri])
                except Exception as e:
                    print(f"处理实体 '{entity}' 时出错: {str(e)}")


# 使用示例
input_json = 'fma_head_hierarchy_maxdepth_5_0821.json'  # 请替换为您的JSON文件路径
output_csv = 'fma_head_hierarchy_0821.csv'  # 输出的CSV文件路径
# left_words_csv = 'fma_head_hierarchy_onlyleft.csv'  # 包含'left'且不存在对应非'left'版本的词的CSV文件路径
fma_ontology = get_ontology("http://purl.org/sig/ont/fma.owl").load()
process_json_to_csv(input_json, output_csv,fma_ontology)


