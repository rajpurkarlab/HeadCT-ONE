# HeadCT-ONE (Ontology Normalization Extraction)

This project implements the Head CT Ontology Normalized Evaluation (HeadCT-ONE), a novel metric for evaluating head CT report generation through ontology-normalized entity and relation extraction. HeadCT-ONE represents a significant advancement in the automated evaluation of radiology reports, particularly for head CT scans.

This project provides the necessary tools and scripts to process head CT reports using the HeadCT-ONE metric, including Named Entity Recognition (NER) models and knowledge graph generation. 

## Environment Setup

To set up the required environment, use the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate ner_kg_env
```
## Model Checkpoints
The checkpoints of NER models can be found at [Google Drive](https://drive.google.com/drive/folders/1J3FLWLGYEhgnolkwsebWtpG6hAHpneKc?usp=sharing)

## Input File Format

The input file should be a JSONL (JSON Lines) file with the following structure for each line:

```json
{
  "id": "unique_identifier",
  "key1": "ground_truth_text",
  "key2": "predicted_text"
}
```

- `id`: A unique identifier for each entry
- `key1`: The ground truth text (default: "original_report")
- `key2`: The predicted text (default: "modified_report")

## Running the Code

To run the code, use the following command:

```bash
python main.py --input_jsonl_file path/to/your/input.jsonl
```

### Optional Arguments

- `--ner_data_dir`: Directory for NER data (default: './ner/data')
- `--ner_result_dir`: Directory for NER results (default: './ner/result/0823_run_relation')
- `--kg_result_dir`: Directory for knowledge graph results (default: './kg/data')
- `--id_key`: Key for ID in input data (default: 'id')
- `--gt_key`: Key for ground truth in input data (default: 'original_report')
- `--pred_key`: Key for predictions in input data (default: 'modified_report')

## Process Overview

1. Preprocess input data
2. Run NER and Relation Extraction models
3. Postprocess results
4. Generate knowledge graphs

## Output

The final results will be saved in the `./kg/data` directory, including:

- `gt.json`: Processed ground truth data
- `pred.json`: Processed prediction data
- `gt_kg.json`: Knowledge graph for ground truth data
- `pred_kg.json`: Knowledge graph for prediction data

## Note

Ensure that you have the necessary permissions to execute the Python scripts and access the specified directories.

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{HeadCT-ONE,
      title={HeadCT-ONE: Enabling Granular and Controllable Automated Evaluation of Head CT Radiology Report Generation}, 
      author={Julián N. Acosta and Xiaoman Zhang and Siddhant Dogra and Hong-Yu Zhou and Seyedmehdi Payabvash and Guido J. Falcone and Eric K. Oermann and Pranav Rajpurkar},
      year={2024},
      eprint={2409.13038},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2409.13038}, 
}

```

Please note that the journal, year, volume, number, pages, and publisher information are not provided in the given text. You should update these fields with the correct information when it becomes available.
