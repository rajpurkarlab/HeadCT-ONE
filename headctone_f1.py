import json
import csv
import argparse
from typing import Dict, List, Tuple, Set,Union
import pandas as pd

def parse_entities_relations(data: Dict[str, Dict]) -> Tuple[Set[Tuple], Set[Tuple]]:
    entities = set()
    relations = set()
    for sentence in data['ner'].values():
        entity_type_mapping = {entity: label for entity, label in sentence['entities'].items()}
        entities.update((entity, label) for entity, label in entity_type_mapping.items())
        
        for relation in sentence['relations']:
            source = (relation['source_entity'], entity_type_mapping.get(relation['source_entity'], 'unknown'))
            target = (relation['target_entity'], entity_type_mapping.get(relation['target_entity'], 'unknown'))
            relations.add((source, target, relation['type']))
    
    return entities, relations


def compute_weighted_f1(gt: Set, pred: Set, gt_weights: Dict, pred_weights: Dict) -> float:
    true_positives = gt.intersection(pred)
    weighted_tp = sum(gt_weights.get(item, 0) for item in true_positives)
    weighted_fp = sum(pred_weights.get(item, 0) for item in pred - gt)
    weighted_fn = sum(gt_weights.get(item, 0) for item in gt - pred)

    precision = weighted_tp / (weighted_tp + weighted_fp) if weighted_tp + weighted_fp else 0
    recall = weighted_tp / (weighted_tp + weighted_fn) if weighted_tp + weighted_fn else 0
    
    return 2 * precision * recall / (precision + recall) if precision + recall else 0



def calculate_radgraph_weighted_f1(gt_dict: Dict[str, Dict], pred_dict: Dict[str, Dict], entity_weights: Dict[str, float]) -> Tuple[float, float]:
    def parse_entities_relations(data: Dict[str, Dict]) -> Tuple[Dict[str, Set[Tuple]], Dict[str, Set[Tuple]]]:
        entities = {}
        relations = {}
        sentence_entities_items = data['ner']
        entities['current_case'] = set()
        relations['current_case'] = set()

        for sentence in sentence_entities_items:
            entity_type_mapping = {}
            for entity, label in sentence_entities_items[sentence]['entities'].items():
                entity_type_mapping[entity] = label
                entities['current_case'].add((entity, label))
                
            for relation_triplets in sentence_entities_items[sentence]['relations']:
                source_entity = relation_triplets['source_entity']
                target_entity = relation_triplets['target_entity']
                relation_type = relation_triplets['type']
                
                # Include entity types in the relation tuple
                source_type = entity_type_mapping.get(source_entity, 'unknown')
                target_type = entity_type_mapping.get(target_entity, 'unknown')
                
                relations['current_case'].add((
                    (source_entity, source_type),
                    (target_entity, target_type),
                    relation_type
                ))


        return entities, relations

    def compute_weighted_f1(
            gt: Set[Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str], str]]],
            pred: Set[Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str], str]]],
            gt_weights: Dict[Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str], str]], float],
            pred_weights: Dict[Union[Tuple[str, str], Tuple[Tuple[str, str], Tuple[str, str], str]], float]
        ) -> float:
        true_positives = gt.intersection(pred)
        false_positives = pred - gt
        false_negatives = gt - pred

        weighted_tp = sum(gt_weights.get(item, 0) for item in true_positives)
        weighted_fp = sum(pred_weights.get(item, 0) for item in false_positives)
        weighted_fn = sum(gt_weights.get(item, 0) for item in false_negatives)

        precision = weighted_tp / (weighted_tp + weighted_fp) if weighted_tp + weighted_fp != 0 else 0
        recall = weighted_tp / (weighted_tp + weighted_fn) if weighted_tp + weighted_fn != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        return f1

    def assign_weights(gt_entities: Set, gt_relations: Set, pred_entities: Set, pred_relations: Set, entity_weights: Dict[str, float]) -> Tuple[Dict[Tuple, float], Dict[Tuple, float], Dict[Tuple, float], Dict[Tuple, float]]:
        gt_entity_weights = {}
        gt_relation_weights = {}
        pred_entity_weights = {}
        pred_relation_weights = {}

        related_entities = set()
        pred_related_entities = set()


        # Assign weights for ground truth entities and relations
        for e, t in gt_entities:
            related_entities.add((e,t))
            gt_entity_weights[(e, t)] = entity_weights.get(t, 0.0)

        for (s, s_type), (t, t_type), r in gt_relations:
            gt_relation_weights[((s, s_type), (t, t_type), r)] = max(entity_weights.get(s_type, 0.0), entity_weights.get(t_type, 0.0))
            if gt_relation_weights[((s, s_type), (t, t_type), r)] > 0:
                related_entities.add((s, s_type))
                related_entities.add((t, t_type))

        for e, t in gt_entities:
            if (e,t) in related_entities:
                gt_entity_weights[(e, t)] = 1

        # Assign weights for predicted entities and relations
        for e, t in pred_entities:
            pred_entity_weights[(e, t)] = entity_weights.get(t, 0.0)

        for (s, s_type), (t, t_type), r in pred_relations:
            pred_relation_weights[((s, s_type), (t, t_type), r)] = max(entity_weights.get(s_type, 0.0), entity_weights.get(t_type, 0.0))
            if pred_relation_weights[((s, s_type), (t, t_type), r)] > 0:
                pred_related_entities.add((s, s_type))
                pred_related_entities.add((t, t_type))

        for e, t in pred_entities:
            if (e,t) in pred_related_entities:
                pred_entity_weights[(e, t)] = 1

        return gt_entity_weights, gt_relation_weights, pred_entity_weights, pred_relation_weights

    gt_entities, gt_relations = parse_entities_relations(gt_dict)
    pred_entities, pred_relations = parse_entities_relations(pred_dict)

    # Check if there are any weighted entities
    has_weighted_entities = any(t in entity_weights for _, t in gt_entities['current_case']) or \
                            any(t in entity_weights for _, t in pred_entities['current_case'])
    
    if not has_weighted_entities:
        print('No weighted entities found')
        return 1.0  # Return 1 if no weighted entities are present

    entity_f1s = []
    relation_f1s = []

    for report_id in gt_entities.keys():
        if report_id in pred_entities:
            gt_entity_weights, gt_relation_weights, pred_entity_weights, pred_relation_weights = assign_weights(
                gt_entities[report_id], gt_relations[report_id],
                pred_entities[report_id], pred_relations.get(report_id, set()),
                entity_weights
            )
            entity_f1 = compute_weighted_f1(
                gt_entities[report_id], 
                pred_entities[report_id], 
                gt_entity_weights,
                pred_entity_weights
            )
            entity_f1s.append(entity_f1)

            if report_id in pred_relations:
                relation_f1 = compute_weighted_f1(
                    gt_relations[report_id], 
                    pred_relations[report_id], 
                    gt_relation_weights,
                    pred_relation_weights
                )
                relation_f1s.append(relation_f1)
            else:
                relation_f1s.append(0)  # No match, F1 = 0
        else:
            entity_f1s.append(0)  # No match, F1 = 0
            relation_f1s.append(0)  # No match, F1 = 0

    avg_entity_f1 = sum(entity_f1s) / len(entity_f1s) if entity_f1s else 0
    avg_relation_f1 = sum(relation_f1s) / len(relation_f1s) if relation_f1s else 0

    return (avg_entity_f1 + avg_relation_f1) / 2

def create_radgraph_f1_matrix(gt_json_file: str, pred_json_file: str, output_csv_path: str, entity_weights: Dict[str, float]):
    # Read JSON file
    with open(gt_json_file, 'r') as f:
        gt_data = json.load(f)

    with open(pred_json_file, 'r') as f:
        pred_data = json.load(f)

    # Extract doc_keys and prepare data for F1 calculation
    doc_keys = list(gt_data.keys())
    n = len(doc_keys)

    # Initialize DataFrame for F1 scores
    df = pd.DataFrame(index=doc_keys, columns=['f1'])

    # Calculate F1 scores
    for i in range(n):
        if doc_keys[i] in pred_data:
            f1 = calculate_radgraph_weighted_f1(gt_data[doc_keys[i]], pred_data[doc_keys[i]], entity_weights)
            df.at[doc_keys[i], 'f1'] = f1
        else:
            print(f"Warning: {doc_keys[i]} not found in prediction data.")
            df.at[doc_keys[i], 'f1'] = None

    # Save to CSV
    df.to_csv(output_csv_path, index_label='ID')
    print(f"F1 score matrix has been saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NER model and generate knowledge graphs")
    parser.add_argument("--gt_kg_file", type=str, default='./kg/data/gt_kg.json', help="Path to input groundtruth json file")
    parser.add_argument("--pred_kg_file", type=str, default='./kg/data/pred_kg.json', help="Path to input prediction json file")
    parser.add_argument("--save_f1_file", type=str, default='./result/weighted_f1.csv', help="Path to save metric csv file")
    args = parser.parse_args()
    
    entity_weights_1 = {
        'observation_present': 1.0
    }
    create_radgraph_f1_matrix(args.gt_kg_file, args.pred_kg_file, args.save_f1_file, entity_weights_1)
    