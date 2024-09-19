import torch
from transformers import AutoTokenizer, AutoModel
import os
import re
import json
import pickle
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from scipy.spatial.distance import cdist
from owlready2 import *

# File I/O functions
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def split_compound_phrase(phrase):
    def is_compound(compound_word):
        prefix_to_lobe = {
            'fronto': 'frontal',
            'parieto': 'parietal',
            'temporo': 'temporal',
            'occipito': 'occipital'
        }
        extracted_lobes = []
        original_word = compound_word
        for prefix, lobe in prefix_to_lobe.items():
            if prefix in compound_word:
                extracted_lobes.append(lobe)
                compound_word = compound_word.replace(prefix, '', 1)
            elif lobe in compound_word:
                extracted_lobes.append(lobe)
                compound_word = compound_word.replace(lobe, '', 1)
        extracted_lobes.sort()
        return extracted_lobes if extracted_lobes else [original_word]

    words = phrase.split()
    result_phrases = []
    has_compound = False

    for i, word in enumerate(words):
        compound_result = is_compound(word.lower())
        if len(compound_result) > 1:  # It's a compound word
            has_compound = True
            for lobe in compound_result:
                new_phrase = words.copy()
                new_phrase[i] = lobe
                result_phrases.append(' '.join(new_phrase))

    # If no compound words were found, return the original phrase
    if not has_compound:
        result_phrases = [phrase]

    return result_phrases

def extract_lobes(compound_word):
    # List of known lobe prefixes and full names
    # lobe_names = ['frontal', 'parietal', 'temporal', 'occipital']
    prefix_to_lobe = {
        'fronto': 'frontal',
        'parieto': 'parietal',
        'temporo': 'temporal',
        'occipito': 'occipital'
    }
    # Extract lobes using prefixes first
    extracted_lobes = []
    for prefix, lobe in prefix_to_lobe.items():
        if prefix in compound_word:
            extracted_lobes.append(lobe)
            compound_word = compound_word.replace(prefix, '', 1)
        elif lobe in compound_word:
            extracted_lobes.append(lobe)
            compound_word = compound_word.replace(lobe, '', 1)
    # Sort lobes for consistency in different orderings
    extracted_lobes.sort()
    return extracted_lobes

def split_compound_phrase(phrase):
    def is_compound(compound_word):
        prefix_to_lobe = {
            'fronto': 'frontal',
            'parieto': 'parietal',
            'temporo': 'temporal',
            'occipito': 'occipital'
        }
        extracted_lobes = []
        original_word = compound_word
        for prefix, lobe in prefix_to_lobe.items():
            if prefix in compound_word:
                extracted_lobes.append(lobe)
                compound_word = compound_word.replace(prefix, '', 1)
            elif lobe in compound_word:
                extracted_lobes.append(lobe)
                compound_word = compound_word.replace(lobe, '', 1)
        extracted_lobes.sort()
        return extracted_lobes if extracted_lobes else [original_word]

    words = phrase.split()
    result_phrases = []
    has_compound = False

    for i, word in enumerate(words):
        compound_result = is_compound(word.lower())
        if len(compound_result) > 1:  # It's a compound word
            has_compound = True
            for lobe in compound_result:
                new_phrase = words.copy()
                new_phrase[i] = lobe
                result_phrases.append(' '.join(new_phrase))

    # If no compound words were found, return the original phrase
    if not has_compound:
        result_phrases = [phrase]

    return result_phrases


def preprocessed_json(input_file_path):
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    for doc_key, doc_data in data.items():
        for sentence, sentence_data in doc_data['ner'].items():
            new_entities = {}
            new_relations = []

            # Process entities
            for entity, entity_type in sentence_data['entities'].items():
                split_entities = split_compound_phrase(entity)
                if len(split_entities) > 1:
                    for split_entity in split_entities:
                        new_entities[split_entity] = entity_type
                else:
                    new_entities[entity] = entity_type

            # Process relations
            for relation in sentence_data['relations']:
                source_splits = split_compound_phrase(relation['source_entity'])
                target_splits = split_compound_phrase(relation['target_entity'])

                for source in source_splits:
                    for target in target_splits:
                        new_relation = deepcopy(relation)
                        new_relation['source_entity'] = source
                        new_relation['target_entity'] = target
                        new_relations.append(new_relation)

            # Update the sentence data with new entities and relations
            sentence_data['entities'] = new_entities
            sentence_data['relations'] = new_relations
    return data

def create_descriptor_dict(descriptor_df):
    descriptor_dict = {}
    map_descriptor_dict = {}
    for _, row in descriptor_df.iterrows():
        descriptor_dict[row['Entity']] = row['Entity']
        map_descriptor_dict[row['Entity']] = [row['First Level'], row['Second Level'], row['Third Level']]
    return descriptor_dict,map_descriptor_dict

def create_findings_dicts(minimalist_findings_df):
    minimalist_findings_dict = {}
    map_minimalist_findings_dict = {}
    for _, row in minimalist_findings_df.iterrows():
        minimalist_findings_dict[row['finding']] = row['finding']
        map_minimalist_findings_dict[row['finding']] = row['finding']
        if row['synonyms'] is not np.nan:
            for synonym in row['synonyms'].split(','):
                synonym = synonym.strip()
                if synonym:
                    minimalist_findings_dict[synonym] = synonym
                    map_minimalist_findings_dict[synonym] = row['finding']
    return minimalist_findings_dict, map_minimalist_findings_dict

def create_devices_dicts(devices_df):
    devices_dict = {}
    map_devices_dict = {}
    for _, row in devices_df.iterrows():
        devices_dict[row['finding']] = row['finding']
        map_devices_dict[row['finding']] = row['finding']
        if row['synonyms'] is not np.nan:
            for synonym in row['synonyms'].split(','):
                synonym = synonym.strip()
                if synonym:
                    devices_dict[synonym] = synonym
                    map_devices_dict[synonym] = row['finding']
    return devices_dict, map_devices_dict


# Embedding generation functions
def generate_embeddings(texts, model, tokenizer, device, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def generate_and_save_embeddings(entity_dict, model, tokenizer, output_file, device):
    entities = list(entity_dict.items())
    ids, texts = zip(*entities)
    
    embeddings = generate_embeddings(texts, model, tokenizer, device)
    
    embedding_dict = {id: embedding for id, embedding in zip(ids, embeddings)}
    save_pickle(embedding_dict, output_file)
    
    print(f"Embeddings saved to {output_file}")
    return embedding_dict



def load_or_generate_embeddings(file_path, entity_dict, model, tokenizer, device):
    if os.path.exists(file_path):
        return load_pickle(file_path)
    else:
        return generate_and_save_embeddings(entity_dict, model, tokenizer, file_path, device)

# Similarity and mapping functions
def find_most_similar_entity(query_embedding, embedding_dict):
    embeddings = np.array(list(embedding_dict.values()))
    ids = list(embedding_dict.keys())
    
    similarities = 1 - cdist(query_embedding.reshape(1, -1), embeddings, metric='cosine')[0]
    most_similar_index = np.argmax(similarities)
    
    return ids[most_similar_index], np.max(similarities)

def remove_findings(entity, minimalist_findings):
    words = entity.split()
    filtered_words = [word for word in words if not any(finding in word.lower() for finding in minimalist_findings)]
    return ' '.join(filtered_words)


def deduplicate_relations(relations):
    # Convert each relation dict to a tuple of its items
    relation_tuples = [tuple(sorted(r.items())) for r in relations]
    
    # Use a set to remove duplicates
    unique_relation_tuples = set(relation_tuples)
    
    # Convert back to list of dicts
    unique_relations = [dict(t) for t in unique_relation_tuples]
    
    return unique_relations

def map_entities(sentence_entities_dict,anatomy_entity_dict, fma_entity_dict, map_minimalist_findings_dict, map_devices_dict, map_descriptor_dict,
                 embedding_dict, minimalist_finding_embedding_dict, devices_embedding_dict, descriptor_embedding_dict,
                 model, tokenizer, device):
    
    return_sentence_dict = {sentence: {"entities":{}, "relations":[]} 
                            for sentence in sentence_entities_dict}
    
    all_entities = []
    for sentence in sentence_entities_dict.keys():
        for entity, entity_type in sentence_entities_dict[sentence]['entities'].items():
            all_entities.append((sentence, entity, entity_type))

    entity_embeddings = generate_embeddings([entity for _, entity, _ in all_entities], model, tokenizer, device)
    
    entity_ontology_dict = {}
    for (sentence, entity, entity_type), query_embedding in zip(all_entities, entity_embeddings):
        if entity_type == "anatomy":
            most_similar_id, similarity = find_most_similar_entity(query_embedding, embedding_dict)
            return_sentence_dict[sentence]['entities'][fma_entity_dict[most_similar_id]] = sentence_entities_dict[sentence]['entities'][entity]
            entity_ontology_dict[entity] = fma_entity_dict[most_similar_id]
        elif entity_type in ["observation_present", "observation_notpresent"]:
            minimalist_findings = list(minimalist_finding_embedding_dict.keys())
            if entity in minimalist_findings:
                return_sentence_dict[sentence]['entities'][map_minimalist_findings_dict[entity]] = sentence_entities_dict[sentence]['entities'][entity]
                entity_ontology_dict[entity] = map_minimalist_findings_dict[entity]
            elif any([finding in entity for finding in minimalist_findings]):
                for finding in minimalist_findings:
                    if finding in entity:
                        # if only one words
                        if len(entity.split()) == len(finding.split()):
                            return_sentence_dict[sentence]['entities'][map_minimalist_findings_dict[finding]] = sentence_entities_dict[sentence]['entities'][entity]
                            entity_ontology_dict[entity] = map_minimalist_findings_dict[finding]
                        else:
                            # if multiple words
                            remain_entity = remove_findings(entity, minimalist_findings)
                            if remain_entity in anatomy_entity_dict:
                                return_sentence_dict[sentence]['entities'][map_minimalist_findings_dict[finding]] = sentence_entities_dict[sentence]['entities'][entity]
                                # need to get anatomy_ontology for remain_entity
                                return_sentence_dict[sentence]['entities'][anatomy_entity_dict[remain_entity]] = "anatomy"
                                entity_ontology_dict[entity] = [map_minimalist_findings_dict[finding],anatomy_entity_dict[remain_entity],'located_at']
                            elif remain_entity in map_descriptor_dict.keys():
                                return_sentence_dict[sentence]['entities'][map_minimalist_findings_dict[finding]] = sentence_entities_dict[sentence]['entities'][entity]
                                return_sentence_dict[sentence]['entities'][map_descriptor_dict[remain_entity][2]] = "descriptor"
                                entity_ontology_dict[entity] = [map_descriptor_dict[remain_entity][2],map_minimalist_findings_dict[finding],'modify']
                            else:
                                return_sentence_dict[sentence]['entities'][map_minimalist_findings_dict[finding]] = sentence_entities_dict[sentence]['entities'][entity]
                                entity_ontology_dict[entity] = map_minimalist_findings_dict[finding]
            else:    
                most_similar_id, similarity = find_most_similar_entity(query_embedding, minimalist_finding_embedding_dict)
                return_sentence_dict[sentence]['entities'][map_minimalist_findings_dict[most_similar_id]] = sentence_entities_dict[sentence]['entities'][entity]
                entity_ontology_dict[entity] = map_minimalist_findings_dict[most_similar_id]
        elif entity_type in ['device_present', 'device_notpresent', 'procedure']:
            map_device_observation = {
                "device_present": "observation_present",
                "device_notpresent": "observation_notpresent",
                "procedure": "observation_present"
            }
            most_similar_id, similarity = find_most_similar_entity(query_embedding, devices_embedding_dict)
            # print(sentence_entities_dict[sentence]['entities'],entity,most_similar_id)
            return_sentence_dict[sentence]['entities'][map_devices_dict[most_similar_id]] = map_device_observation[sentence_entities_dict[sentence]['entities'][entity]]
            entity_ontology_dict[entity] = map_devices_dict[most_similar_id]
        elif entity_type == "descriptor":
            most_similar_id, similarity = find_most_similar_entity(query_embedding, descriptor_embedding_dict)
            return_sentence_dict[sentence]['entities'][map_descriptor_dict[most_similar_id][2]] = sentence_entities_dict[sentence]['entities'][entity]
            entity_ontology_dict[entity] = map_descriptor_dict[most_similar_id][2]
        else:
            print(f"Unknown entity type: {entity_type}")
        # print(sentence, entity, entity_type,return_sentence_dict[sentence]['entities'])
        
    for sentence in sentence_entities_dict:
        for relation in sentence_entities_dict[sentence]['relations']:
            entity1, entity2, relation_type = relation['source_entity'], relation['target_entity'], relation['type']
            if entity1 in entity_ontology_dict.keys() and entity2 in entity_ontology_dict.keys():
                entity1_ontology = entity_ontology_dict[entity1]
                entity2_ontology = entity_ontology_dict[entity2]
                # if entity1_ontology is list then it is a modify relation
                if isinstance(entity1_ontology, list):
                    return_sentence_dict[sentence]['relations'].append({
                                                                        "source_entity": entity1_ontology[0],
                                                                        "target_entity": entity1_ontology[1],
                                                                        "type": entity1_ontology[2]
                                                                        })
                    source_entity = entity1_ontology[1] if entity1_ontology[2] == 'modify' else entity1_ontology[0]
                    # print(return_sentence_dict[sentence]['relations'])
                else:
                    source_entity = entity1_ontology
                if isinstance(entity2_ontology, list):
                    return_sentence_dict[sentence]['relations'].append({
                                                                        "source_entity": entity2_ontology[0],
                                                                        "target_entity": entity2_ontology[1],
                                                                        "type": entity2_ontology[2]
                                                                        })
                    target_entity = entity2_ontology[1] if entity2_ontology[2] == 'modify' else entity2_ontology[0]
                    # print(return_sentence_dict[sentence]['relations'])
                else:
                    target_entity = entity2_ontology
                
                return_sentence_dict[sentence]['relations'].append({
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "type": relation_type
                })
            else:
                # pass
                print(f"Entity not found in entity_ontology_dict: {sentence} {entity1}, {entity2}")
        unique_relations = deduplicate_relations(return_sentence_dict[sentence]['relations'])
        return_sentence_dict[sentence]['relations'] = unique_relations
    return return_sentence_dict


# Main processing function
def preprocess_json(input_json_file, save_json_file,anatomy_entity_dict, fma_entity_dict, map_minimalist_findings_dict, 
                    map_devices_dict,map_descriptor_dict, fma_embedding_dict, minimalist_finding_embedding_dict, 
                    devices_embedding_dict, descriptor_embedding_dict, model, tokenizer, device):
    # json_data = read_json(input_json_file)
    json_data = preprocessed_json(input_json_file)
    save_data_dict = {}
    
    for report_id in tqdm(json_data):
        sentence_entities_dict = json_data[report_id]['ner']
        map_fma_sentence_entities_dict = map_entities(sentence_entities_dict,anatomy_entity_dict, fma_entity_dict, 
                                                      map_minimalist_findings_dict, map_devices_dict, map_descriptor_dict,
                                                      fma_embedding_dict, minimalist_finding_embedding_dict,
                                                      devices_embedding_dict, descriptor_embedding_dict,
                                                      model, tokenizer, device)
        
        
        save_data_dict[report_id] = json_data[report_id]
        save_data_dict[report_id]['ner'] = map_fma_sentence_entities_dict
        save_json(save_data_dict, save_json_file)
    

def map_kg(input_json_file,save_json_file):
    model_path = "FremyCompany/BioLORD-2023-C"
    fma_csv_path = './kg/fma/fma_head_hierarchy_0821.csv'
    finding_ontology_path = './kg/observations/finding_ontology_0821.csv'
    descriptor_ontology_path = './kg/descriptors/headct_descriptors_ontology_gpt4.csv'
    anatomy_entity_path = './kg/anatomy/headct_anatomy.csv'
    anatomy_entity_df = pd.read_csv(anatomy_entity_path)
    anatomy_entity_dict = {row['Entity']: row['Mapped Entity'] for _, row in anatomy_entity_df.iterrows()}
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model and tokenizer initialization
    embedding_tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
    embedding_model = AutoModel.from_pretrained(model_path).to(device).eval()

    # Load FMA ontology and create entity dictionary
    fma_ontology = get_ontology("./kg/fma/fma.owl").load()
    head_fma_df = pd.read_csv(fma_csv_path)
    fma_entity_dict = {row['IRI']: row['Entity'] for _, row in head_fma_df.iterrows()}

    # Load or generate FMA embeddings
    fma_embedding_dict = load_pickle("./kg/fma/fma_headct_embeddings.pkl") if os.path.exists("./kg/fma/fma_headct_embeddings.pkl") else generate_and_save_embeddings(fma_entity_dict, embedding_model, embedding_tokenizer, "./kg/fma/fma_headct_embeddings.pkl", device)

    # Load and process ontology
    descriptor_df = pd.read_csv(descriptor_ontology_path)
    head_finding_df = pd.read_csv(finding_ontology_path)
    devices_df = head_finding_df[head_finding_df['device'] == 'yes']
    minimalist_findings_df = head_finding_df[(head_finding_df['device'] != 'yes')]
    
    # Create dictionaries
    # minimalist_child_dict = head_finding_df.groupby('finding')['modified_child_finding'].apply(lambda x: list(set(x))).to_dict()
    descriptor_dict,map_descriptor_dict = create_descriptor_dict(descriptor_df)
    minimalist_findings_dict, map_minimalist_findings_dict = create_findings_dicts(minimalist_findings_df)
    devices_dict, map_devices_dict = create_devices_dicts(devices_df)

    # Load or generate embeddings
    minimalist_finding_embedding_dict = load_or_generate_embeddings("./kg/embeddings/headct_minimalist_finding_embeddings.pkl", minimalist_findings_dict, embedding_model, embedding_tokenizer, device)
    devices_embedding_dict = load_or_generate_embeddings("./kg/embeddings/headct_device_embeddings.pkl", devices_dict, embedding_model, embedding_tokenizer, device)
    descriptor_embedding_dict = load_or_generate_embeddings("./kg/embeddings/headct_descriptor_embeddings.pkl", descriptor_dict, embedding_model, embedding_tokenizer, device)
    preprocess_json(input_json_file, save_json_file,anatomy_entity_dict, fma_entity_dict, map_minimalist_findings_dict, map_devices_dict,map_descriptor_dict, fma_embedding_dict, minimalist_finding_embedding_dict, devices_embedding_dict, descriptor_embedding_dict, embedding_model, embedding_tokenizer, device)