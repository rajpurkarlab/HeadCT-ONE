import os
import glob
import json 
import re
import random
import argparse
import pandas as pd 
from tqdm import tqdm

def get_sentence_list(text):
    """
    Splits a given text into sentences based on the rule that each sentence ends with a period and the
    next sentence starts with an uppercase letter.
    """
    sentences = []
    start = 0
    for i in range(len(text)):
        # Check if the current character is a period and the next character is an uppercase letter, considering multiple spaces
        if text[i] == ".":
            j = i + 1
            # Skip any additional spaces to check for the next uppercase letter
            while j < len(text) and text[j] == " ":
                j += 1
            if j < len(text) and text[j].isupper():
                sentences.append(text[start:i + 1].strip())
                start = i + 1
    # Add the last sentence if there's any text left after the last period found
    if start < len(text):
        sentences.append(text[start:].strip())
    return sentences

def find_word_indices(sen, target_word):
    target_words = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ',target_word).lower().split()
    start_index = -1
    end_index = -1
    for i, word in enumerate(sen):
        if word == target_words[0] and (start_index == -1 or end_index == -1):
            if sen[i:i+len(target_words)] == target_words:
                start_index = i
                end_index = i + len(target_words) - 1
    return start_index, end_index

def get_ner_list(sen,res_dict_id):
    return_ner_list = []
    return_ner_dict = {}
    for entity, entity_type in res_dict_id.items():
        start_index, end_index = find_word_indices(sen, entity)
        return_ner_list.append([start_index, end_index, entity_type])
        return_ner_dict[entity] = [start_index, end_index]
    return return_ner_list,return_ner_dict
    
def clean_findings_section(findings):
    # if findings is a dict, convert to string
    if isinstance(findings, dict):
        findings = json.dumps(findings)
    if 'FINDINGS:' in findings:
        findings = findings.split('FINDINGS:')[1]
    elif 'Findings:' in findings:
        findings = findings.split('Findings:')[1]
    elif 'FINDINGS\n' in findings:
        findings = findings.split('FINDINGS\n')[1]
    elif 'RESULT:' in findings:
        findings = findings.split('RESULT:')[1]
    elif 'FINDINGS/RESULTS' in findings:
        findings = findings.split('FINDINGS/RESULTS')[1]
        
    if 'CT CERVICAL SPINE FINDINGS' in findings:
        findings = findings.split('CT CERVICAL SPINE FINDINGS')[0]
    elif 'CT CERVICAL FINDINGS' in findings:
        findings = findings.split('CT CERVICAL FINDINGS')[0]
    elif 'CT THORACIC SPINE FINDINGS' in findings:
        findings = findings.split('CT THORACIC SPINE FINDINGS')[0]
    elif 'CT CERVIC the AL SPINE FINDINGS' in findings:
        findings = findings.split('CT CERVIC the AL SPINE FINDINGS')[0]
    
    if 'CT HEAD FINDINGS' in findings:
        findings = findings.split('CT HEAD FINDINGS')[1]
    elif 'FINDINGS' in findings:
        findings = findings.split('FINDINGS')[1]
    
    findings = findings.replace('\n',' ').strip()
    return findings
        
def preprocess_sentences_test_withquery(input_jsonl_file,save_json_file,id_key,query_key = 'original_report'):
    json_data = {}
    with open(input_jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            row_id = data[id_key]
            findings = clean_findings_section(data[query_key])
            json_data[row_id] = findings
    
    index = 0
    save_rows = []
    node_id_list = list(json_data.keys())
    final_list = []
    for select_id in tqdm(node_id_list):
        report = json_data[select_id]
        sentence_list = get_sentence_list(report)
        for sen_idx in range(len(sentence_list)):
            sentence = sentence_list[sen_idx]
            sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ',sentence).lower().split()
            temp_dict = {}
            temp_dict["doc_key"] = str(select_id) + '_' + str(sen_idx)
            temp_dict["sentences"] = [sen]
            temp_dict["ner"] = [[]]
            temp_dict["relations"] = [[]]
            final_list.append(temp_dict)
            save_rows.append([index,str(select_id) + '_' + str(sen_idx),str(sentence)])
            index += 1
            
    with open(save_json_file,'w') as outfile:
        for item in final_list:
            json.dump(item, outfile)
            outfile.write("\n")


