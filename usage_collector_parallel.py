import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import BertModel, BertTokenizer
from nltk.tokenize import sent_tokenize
from concurrent.futures import ProcessPoolExecutor
import argparse


def get_context(token_ids, target_position, sequence_length=128):
    """
    Extracts the surrounding context of a target word from a list of token IDs.
    """
    window_size = int((sequence_length - 2) / 2)
    context_start = max(0, target_position - window_size)
    padding_offset = max(0, window_size - target_position)
    padding_offset += max(0, target_position + window_size - len(token_ids))

    context_ids = token_ids[context_start:target_position + window_size]
    context_ids += [0] * padding_offset
    new_target_position = target_position - context_start

    return context_ids, new_target_position


def process_bucket(args):
    """
    Processes a single decade's file, collecting usages of target words and saving the result to a unique file.
    :param args: A tuple of arguments (bucket, target_words, model_path, coha_dir, sequence_length, buffer_size, output_path)
    """
    bucket, target_words, model_path, coha_dir, sequence_length, buffer_size, output_path = args

    # Convert bucket to a plain integer or string for compatibility
    bucket_str = str(int(bucket))

    # Generate the decade-specific output path
    output_path = output_path.format(bucket=bucket_str)

    usages = defaultdict(list)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path, output_hidden_states=True)
    if torch.cuda.is_available():
        model.to("cuda")

    print(f"Processing bucket: {bucket_str}...")
    with open(f"{coha_dir}/all_{bucket_str}.txt", "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Bucket {bucket_str}"):
        # Split the line into sentences
        sentences = sent_tokenize(line)
        num_sentences = len(sentences)

        # Determine valid sentences based on rules
        if num_sentences == 5:
            valid_sentence_indices = [2]  # Only the 3rd sentence
        elif num_sentences == 4:
            valid_sentence_indices = [1, 2]  # 2nd and 3rd sentences
        elif num_sentences == 3:
            valid_sentence_indices = [0, 2]  # 1st and 3rd sentences
        else:
            valid_sentence_indices = []  # Skip

        for idx, sentence in enumerate(sentences):
            if idx not in valid_sentence_indices:
                continue

            tokens = tokenizer.encode(sentence)
            # Skip sentences longer than the allowed sequence length
            if len(tokens) > sequence_length:
                print(f"Skipping sentence with length {len(tokens)}")
                continue
            for pos, token in enumerate(tokens):
                if token not in target_words:
                    continue

                # Get context and position
                context_ids, pos_in_context = get_context(tokens, pos, sequence_length)
                input_ids = [101] + context_ids + [102]

                # Perform inference in batches
                batch_input_ids = [input_ids]
                input_ids_tensor = torch.tensor(batch_input_ids)
                if torch.cuda.is_available():
                    input_ids_tensor = input_ids_tensor.to("cuda")

                with torch.no_grad():
                    outputs = model(input_ids_tensor)
                    hidden_states = outputs.hidden_states
                    hidden_states = [layer.detach().cpu().numpy() for layer in hidden_states]
                    usage_vector = np.sum(np.stack(hidden_states)[1:], axis=0)[0, pos_in_context + 1, :]

                usages[token].append((usage_vector, context_ids, pos_in_context, bucket))

    # Save the result for this bucket
    with open(output_path, "wb") as f:
        pickle.dump(usages, f)

    print(f"Saved results for bucket {bucket_str} to {output_path}")
    return usages



def collect_from_coha_parallel(target_words, buckets, pretrained_weights, 
                               coha_dir, sequence_length=128, buffer_size=1024, 
                               output_path=None):
    """
    Collect usages of target words from the COHA dataset in parallel over decades.
    """
    args_list = [
        (bucket, target_words, pretrained_weights, coha_dir, 
         sequence_length, buffer_size, output_path)
        for bucket in buckets  # Ensure each process gets a single bucket
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_bucket, args_list))

    # Combine results from all decades
    final_usages = defaultdict(list)
    for usages in results:
        for key, value in usages.items():
            final_usages[key].extend(value)

    # Save final results to the output path
    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(final_usages, f)

    return final_usages



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seqlen', type=int, default=128)
    parser.add_argument('--bertdir', type=str, default='model/bert-base-german-cased')
    parser.add_argument('--cohadir', type=str, default='data/all_xxxx_files')
    parser.add_argument('--outdir', type=str, default='data/collect_out')
    parser.add_argument('--buffer', type=int, default=1024)

    args = parser.parse_args()


    TARGET_WORDS = ['Solidarität', 'Freiheit', 'Maus', 'Merkel']
    BUCKETS = list(np.arange(1990, 2024, 3))
    SEQUENCE_LENGTH = 512
    PRETRAINED_WEIGHTS = "model/bert-base-german-cased"
    COHA_DIR = "data/all_xxxx_files"
    OUTPUT_PATH = 'data/collect_out/concat/usages_16_len128_{bucket}.dict'
    BUFFER_SIZE = 1024

    collect_from_coha_parallel(
        TARGET_WORDS,
        BUCKETS,
        PRETRAINED_WEIGHTS,
        COHA_DIR,
        SEQUENCE_LENGTH,
        BUFFER_SIZE,
        OUTPUT_PATH
    )
    print("Processing complete. Results saved to:", OUTPUT_PATH)
