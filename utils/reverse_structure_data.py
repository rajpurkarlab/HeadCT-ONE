import os
import glob
import json 
import re
import random
import argparse
import pandas as pd 
from tqdm import tqdm

def preprocess_sentences(input_json_file,save_json_file):
    with open(input_json_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    processed_data = {}
    for doc in tqdm(data):
        doc_key = doc['doc_key'].split('_')[0]
        sentences = ' '.join(doc['sentences'][0])
        
        if doc_key not in processed_data:
            processed_data[doc_key] = {'doc_key':doc_key, 'sentences': [sentences], 'ner': {}}
        else:
            processed_data[doc_key]['sentences'].append(sentences)
        
        sentences = ' '.join(doc['sentences'][0])
        processed_data[doc_key]['ner'][sentences] = {}
        predicted_entities = {}
        for entity_info in doc['predicted_ner'][0]:
            start, end, entity_type  = entity_info
            entity_text = ' '.join(doc['sentences'][0][start:end + 1])
            predicted_entities[entity_text.replace(' . ','.')] = entity_type
        processed_data[doc_key]['ner'][sentences]['entities'] = predicted_entities

        predicted_relations = []
        for relation_info in doc['predicted_relations'][0]:
            if relation_info: 
                start1, end1, start2, end2, relation_type = relation_info
                entity1_text = ' '.join(doc['sentences'][0][start1:end1 + 1])
                entity2_text = ' '.join(doc['sentences'][0][start2:end2 + 1])
                predicted_relations.append({'source_entity': entity1_text.replace(' . ','.'), 'target_entity': entity2_text.replace(' . ','.'), 'type': relation_type})

        processed_data[doc_key]['ner'][sentences]['relations'] = predicted_relations
    
    with open(save_json_file, 'w', encoding='utf-8') as output_file:
        json.dump(processed_data, output_file, ensure_ascii=False, indent=4)
