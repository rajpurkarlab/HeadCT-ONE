import json
import time
import pandas as pd
from openai import AzureOpenAI

def get_messages(queries, sentences, descriptor_string):
    system_message = f"""
    You are an AI trained to classify medical descriptors based on a predefined hierarchical category system.
    Your task is to analyze given medical descriptors and classify them into the most appropriate categories.

    Important notes:
    1. The category system is hierarchical with three levels: First level, Second level, and Third level.
    2. The full category system is provided here: {descriptor_string}
    3. A descriptor phrase can be mapped to multiple categories.
    4. For single-word descriptors, map to the most closely related category based on the given sentence context.

    Example input and output:
    Input: 
    1. Descriptor: normal in size and position, Sentence: the ventricles are normal in size and position.
    2. Descriptor: multi focal, Sentence: multi focal periventricular and subcortical white matter hypodensity is seen.

    Output: 
    {{
        "normal in size and position": [
            ["size", "qualitative", "medium"],
            ["position", "normal_position", "normal_position"]
        ],
        "multi focal": [
            ["distribution", "multifocal", "multifocal"]
        ]
    }}

    Given a list of input descriptors and sentences, reply with a JSON object containing classifications for all descriptors:
    {{
        "input descriptor 1": [[First level, Second level, Third level], ...],
        "input descriptor 2": [[First level, Second level, Third level], ...],
        ...
    }}

    If a level is not applicable, leave it as an empty string.
    If you're unsure or the descriptor doesn't fit any category, use ["Unclassified", "", ""].
    """
    
    user_message = "Classify the following medical descriptors based on their example sentences:\n"
    for query, sentence in zip(queries, sentences):
        user_message += f"Descriptor: {query}, Sentence: {sentence}\n"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    return messages

def chatgpt_input(messages):
    client = AzureOpenAI(
        api_version="2023-05-15",
        api_key="",
        azure_endpoint=""
    )

    response = client.chat.completions.create(
        model="gpt4o05132024",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    completion = response.choices[0].message.content
    cost = estimate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
    return json.loads(completion), cost

def estimate_cost(prompt_tokens, completion_tokens):
    input_cost = 0.005
    output_cost = 0.015
    return (input_cost * prompt_tokens / 1000 + output_cost * completion_tokens / 1000)

def process_batch(query_list, sentence_list, descriptor_string, processed_queries):
    # Filter out already processed queries
    new_queries = []
    new_sentences = []
    for query, sentence in zip(query_list, sentence_list):
        if query not in processed_queries:
            new_queries.append(query)
            new_sentences.append(sentence)
    
    if not new_queries:
        return {}, 0  # All queries in this batch were already processed
    
    messages = get_messages(new_queries, new_sentences, descriptor_string)
    try:
        res, cost = chatgpt_input(messages)
        results = {
            query: {
                'example sentence': sentence,
                'ontology': res[query]
            } for query, sentence in zip(new_queries, new_sentences)
        }
        return results, cost
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        time.sleep(1)
        return {}, 0

def load_processed_queries(save_json_file):
    try:
        with open(save_json_file, 'r') as infile:
            return json.load(infile)
    except FileNotFoundError:
        return {}

def batch_process_and_save(query_list, sentence_list, save_json_file, descriptor_string):
    batch_size = 10
    all_results = load_processed_queries(save_json_file)
    print(len(all_results.keys()))
    total_cost = 0
    
    for i in range(0, len(query_list), batch_size):
        batch_queries = query_list[i:i+batch_size]
        batch_sentences = sentence_list[i:i+batch_size]
        
        batch_results, batch_cost = process_batch(batch_queries, batch_sentences, descriptor_string, all_results)
        all_results.update(batch_results)
        total_cost += batch_cost
        print(len(all_results.keys()))
        # Save results after each batch
        with open(save_json_file, 'w') as outfile:
            json.dump(all_results, outfile, indent=4)
        
        print(f"Processed batch {i//batch_size + 1}. Total cost so far: {total_cost}")
    
    print(f"All batches processed. Total cost: {total_cost}")
    return all_results, total_cost

def main():
    # Load your CSV file
    df = pd.read_csv('./headct_descriptors_ontology_unmapped.csv')
    query_list = df['Entity'].tolist() 
    sentence_list = df['Example Sentence'].tolist() 
    
    # Load descriptor ontology
    descriptor_df = pd.read_csv('./descriptors_ontology.csv')
    descriptor_string = descriptor_df.to_csv(index=False).strip()
    
    save_json_file = './headct_descriptors_ontology_unmapped_results.json'
    
    results, total_cost = batch_process_and_save(query_list, sentence_list, save_json_file, descriptor_string)
    print(f"Processing complete. Total cost: {total_cost}")

if __name__ == '__main__':
    main()