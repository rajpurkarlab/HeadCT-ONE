'''
Author: xiaoman-zhang
Date: 2024-03-14 14:56:13
LastEditors: xiaoman-zhang
LastEditTime: 2024-03-14 14:56:15
FilePath: /2024_RadEval/src/ner/pure_ner/shared/const.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
task_ner_labels = {
    'headct':['anatomy','observation_present','observation_notpresent','procedure','descriptor'],
    'headct_version2':['anatomy','observation_present','observation_notpresent','device_present','device_notpresent','procedure','descriptor'],
    'mimic01': ['size','anatomy','disorder_present','disorder_notpresent','concept','procedures','devices_present','devices_notpresent'],
    'mimic02': ['anatomy','disorder_present','disorder_notpresent','concept','procedures','devices_present','devices_notpresent'],
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}

task_rel_labels = {
    'headct': ['located_at','suggestive_of','associate_with','modify'],
    'mimic01': ['located_at','suggestive_of','modify'],
    'mimic02': ['located_at','suggestive_of','modify'],
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
