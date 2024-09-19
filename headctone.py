import os
import argparse
from utils.structure_data import preprocess_sentences_test_withquery
from utils.reverse_structure_data import preprocess_sentences
from kg.get_metric_file import map_kg

def process_ner_model(input_jsonl_file, ner_data_dir, ner_result_dir,kg_result_dir, id_key, gt_key, pred_key):
    # Step 1: Preprocess input data
    gt_json_file = os.path.join(ner_data_dir, "gt.json")
    pred_json_file = os.path.join(ner_data_dir, "pred.json")
    preprocess_sentences_test_withquery(input_jsonl_file, gt_json_file, id_key, gt_key)
    preprocess_sentences_test_withquery(input_jsonl_file, pred_json_file, id_key, pred_key)

    # Step 2: Run NER and Relation Extraction models
    run_ner_and_relation_models(ner_data_dir)

    # Step 3: Postprocess results
    env_rel_gt_json_file = os.path.join(ner_result_dir, "env_rel_gt.json")
    env_rel_pred_json_file = os.path.join(ner_result_dir, "env_rel_pred.json")
    save_gt_json_file = os.path.join(kg_result_dir, "gt.json")
    save_pred_json_file = os.path.join(kg_result_dir, "pred.json")
    preprocess_sentences(env_rel_gt_json_file, save_gt_json_file)
    preprocess_sentences(env_rel_pred_json_file, save_pred_json_file)

    # Step 4: Generate knowledge graphs
    save_kg_gt_json_file = os.path.join(kg_result_dir, "gt_kg.json")
    save_kg_pred_json_file = os.path.join(kg_result_dir, "pred_kg.json")
    map_kg(save_gt_json_file, save_kg_gt_json_file)
    map_kg(save_pred_json_file, save_kg_pred_json_file)

    print(f"Processing complete. Results saved in {ner_result_dir}")

def run_ner_and_relation_models(ner_data_dir):
    # NER model command
    ner_command = f"""
    python ./ner/run_entity.py \
    --do_eval \
    --eval_test \
    --learning_rate=1e-5 \
    --task_learning_rate=5e-4 \
    --train_batch_size=8 \
    --context_window 100 \
    --task headct_version2 \
    --data_dir ./ner/data \
    --dev_data ./ner/data/gt.json \
    --test_data ./ner/data/gt.json \
    --test_pred_filename env_gt.json \
    --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --output_dir ./ner/result/run_entity
    """

    # Relation extraction model command
    relation_command = f"""
    python ./ner/run_relation.py \
    --task headct \
    --do_eval \
    --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --do_lower_case \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --context_window 100 \
    --max_seq_length 256 \
    --output_dir ./ner/result/0823_run_relation \
    --entity_output_dir ./ner/result/run_entity \
    --entity_predictions_dev env_gt.json \
    --entity_predictions_test env_gt.json \
    --prediction_file env_rel_gt.json
    """

    # Execute NER and Relation Extraction commands
    os.system(ner_command)
    os.system(relation_command)

    # Repeat the process for predictions
    ner_command = ner_command.replace("gt.json", "pred.json")
    relation_command = relation_command.replace("gt.json", "pred.json")
    
    os.system(ner_command)
    os.system(relation_command)

def main():
    parser = argparse.ArgumentParser(description="Process NER model and generate knowledge graphs")
    parser.add_argument("--input_jsonl_file", type=str, default='./data/headct_eval_annotations.jsonl', help="Path to input JSONL file")
    parser.add_argument("--ner_data_dir", type=str, default='./ner/data', help="Directory for NER data")
    parser.add_argument("--ner_result_dir",type=str, default='./ner/result/0823_run_relation', help="Directory for NER results")
    parser.add_argument("--kg_result_dir",type=str, default='./kg/data', help="Directory for NER results")
    parser.add_argument("--id_key", type=str, default='id',help="Key for ID in input data")
    parser.add_argument("--gt_key", type=str, default='original_report', help="Key for ground truth in input data")
    parser.add_argument("--pred_key", type=str, default='modified_report', help="Key for predictions in input data")
    args = parser.parse_args()
    
    if not os.path.exists(args.ner_data_dir):
        os.makedirs(args.ner_data_dir)
    
    if not os.path.exists(args.ner_result_dir):
        os.makedirs(args.ner_result_dir)
    
    if not os.path.exists(args.kg_result_dir):
        os.makedirs(args.kg_result_dir)
    
    process_ner_model(args.input_jsonl_file, args.ner_data_dir, args.ner_result_dir, args.kg_result_dir,
                      args.id_key, args.gt_key, args.pred_key)

if __name__ == "__main__":
    main()