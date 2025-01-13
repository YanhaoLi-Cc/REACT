from dataclasses import dataclass, field
import logging
import json
import os

from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch
import tqdm

from presto.training import ModelArguments
from presto.inference import load_trained_lora_model, load_trained_model
from presto.data_tools import encode_chat, parse_chat_output, encode_interleaved_data
from presto.chemistry_tools import EVALUATOR_BUILDERS


@dataclass
class EvaluationArguments(ModelArguments):
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lora_enable: bool = field(default=True, metadata={"help": "Enable LoRA."})
    max_new_tokens: int = field(default=2048, metadata={"help": "Maximum number of new tokens to generate."})
    temperature: float = field(default=0.2, metadata={"help": "Temperature to use for sampling."})
    top_k: int = field(default=50, metadata={"help": "Top k to use for sampling."})
    top_p: float = field(default=0.8, metadata={"help": "Top p to use for sampling."})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample from the output distribution."})
    load_bits: int = field(default=16, metadata={"help": "Quantization bits to use."})
    parser: str = field(default='base', metadata={"help": "Parser for the generated output."})
    evaluator: str = field(default='smiles', metadata={"help": "Evaluator to use for the generated output."})
    cache_dir: str = field(default=None, metadata={"help": "Path to the cache directory."})
    output_dir: str = field(default=None, metadata={"help": "Path to the output file."})
    is_icl: bool = field(default=False, metadata={"help": "Whether ICL testing is enabled."})
    verbose: bool = field(default=False, metadata={"help": "Print verbose output."})


def _save_rows(rows: list, cache_dir: str, file_name: str = "rows.txt"):
    rows = [str(row) for row in rows]
    os.makedirs(cache_dir, exist_ok=True)
    if file_name.endswith(".txt"):
        with open(os.path.join(cache_dir, file_name), "w") as file:
            file.write("\n".join(rows))
    elif file_name.endswith(".json"):
        with open(os.path.join(cache_dir, file_name), "w") as file:
            json.dump(rows, file)
    else:
        raise ValueError(f"Unknown file format: {file_name}")

def _save_score(score: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "score.json"), "w") as file:
        json.dump(score, file)


# def _resolve_dataset(path: str) -> HFDataset:
#     if os.path.exists(path):
#         try:
#             return load_from_disk(path)
#         except:
#             return load_dataset(path, split="test")
#     else:
#         return load_dataset(path, split="test", data_files="*.arrow")


def _resolve_dataset(path: str) -> HFDataset:
    if os.path.exists(path):
        try:
            return load_from_disk(path)
        except:
            return load_dataset(path, split="train")
    else:
        return load_dataset(path, split="train", data_files="*.arrow")


def _load_cached_results(cache_dir: str) -> tuple[list, list, int]:
    """Load cached predictions and references if they exist."""
    try:
        with open(os.path.join(cache_dir, "cached_predictions.txt"), "r") as f:
            predictions = f.read().splitlines()
        with open(os.path.join(cache_dir, "cached_references.txt"), "r") as f:
            references = f.read().splitlines()
        with open(os.path.join(cache_dir, "last_index.txt"), "r") as f:
            last_index = int(f.read())
        return predictions, references, last_index
    except FileNotFoundError:
        return [], [], 0


def _save_cached_results(predictions: list, references: list, last_index: int, cache_dir: str):
    """Save current progress to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "cached_predictions.txt"), "w", encoding='utf-8') as f:
        f.write("\n".join(map(str, predictions)))
    with open(os.path.join(cache_dir, "cached_references.txt"), "w", encoding='utf-8') as f:
        f.write("\n".join(map(str, references)))
    with open(os.path.join(cache_dir, "last_index.txt"), "w", encoding='utf-8') as f:
        f.write(str(last_index))


def _evaluate(model, tokenizer, dataset, args, batch_size=5):
    if args.cache_dir:
        predictions, references, start_index = _load_cached_results(args.cache_dir)
    else:
        predictions, references = [], []
        start_index = 0
        
    evaluator = EVALUATOR_BUILDERS[args.evaluator]()
    
    # 用于存储DPO数据
    dpo_data = []
    
    dataset_slice = list(dataset)[start_index:]
    num_batches = (len(dataset_slice) + batch_size - 1) // batch_size
    batch_iter = tqdm.tqdm(range(num_batches), desc="Evaluating Batches", 
                          initial=start_index//batch_size, total=num_batches)
    
    for batch_idx in batch_iter:
        try:
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(dataset_slice))
            batch_entries = dataset_slice[batch_start:batch_end]
            
            batch_encoded = []
            batch_ground_truths = []
            batch_queries = []  # 存储原始查询
            
            for entry in batch_entries:
                if args.is_icl:
                    encoded = encode_interleaved_data(entry, tokenizer, model.modalities)
                else:
                    encoded = encode_chat(entry, tokenizer, model.modalities)
                batch_encoded.append(encoded)
                batch_ground_truths.append(entry['ground_truth'])
                batch_queries.append(entry.get('query', entry.get('instruction', '')))  # 获取查询文本
            
            max_length = max(encoded["input_ids"].size(0) for encoded in batch_encoded)
            
            padded_input_ids = []
            attention_mask = []
            for encoded in batch_encoded:
                input_length = encoded["input_ids"].size(0)
                padding_length = max_length - input_length
                
                padded_ids = torch.cat([
                    encoded["input_ids"],
                    torch.ones(padding_length, dtype=encoded["input_ids"].dtype) * tokenizer.pad_token_id
                ])
                padded_input_ids.append(padded_ids)
                
                mask = torch.cat([
                    torch.ones(input_length),
                    torch.zeros(padding_length)
                ])
                attention_mask.append(mask)
            
            batch_input_ids = torch.stack(padded_input_ids).to(model.device)
            batch_attention_mask = torch.stack(attention_mask).to(model.device)
            
            batch_modality_inputs = {
                m.name: [encoded[m.name] for encoded in batch_encoded] 
                for m in model.modalities
            }
            
            with torch.inference_mode():
                batch_output_ids = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    modality_inputs=batch_modality_inputs,
                )
            
            for i, output_ids in enumerate(batch_output_ids):
                input_length = batch_encoded[i]["input_ids"].shape[0]
                generated_output = tokenizer.decode(
                    output_ids[input_length:],
                    skip_special_tokens=True,
                ).strip()
                
                if args.parser:
                    try:
                        generated_output = parse_chat_output(generated_output, args.parser)["output"]
                    except:
                        pass
                
                if args.verbose:
                    print(f"Query: {batch_queries[i]}")
                    print(f"Ground Truth: {batch_ground_truths[i]}")
                    print(f"Generated Output: {generated_output}")
                
                # 构建DPO数据
                dpo_entry = {
                    "query": batch_queries[i],
                    "chosen": batch_ground_truths[i],  # ground truth作为正样本
                    "rejected": generated_output,      # 模型生成的答案作为负样本
                }
                dpo_data.append(dpo_entry)
                
                predictions.append(generated_output)
                references.append(batch_ground_truths[i])
            
            current_index = start_index + batch_start + len(batch_entries)
            if args.cache_dir and current_index % 5 == 0:
                _save_cached_results(predictions, references, current_index, args.cache_dir)
                # 同时保存DPO数据
                with open(os.path.join(args.cache_dir, "dpo_data.json"), "w", encoding='utf-8') as f:
                    json.dump(dpo_data, f, ensure_ascii=False, indent=2)
                
        except RuntimeError as e:
            if "probability tensor contains" in str(e):
                print("Encountered invalid probability tensor:")
                print(f"Batch causing error: {batch_entries}")
                if args.cache_dir:
                    _save_cached_results(predictions, references, 
                                      start_index + batch_start, args.cache_dir)
                continue
            else:
                if args.cache_dir:
                    _save_cached_results(predictions, references, 
                                      start_index + batch_start, args.cache_dir)
                raise e

    # 保存最终结果
    if args.cache_dir:
        _save_rows(predictions, args.cache_dir, "predictions.txt")
        _save_rows(references, args.cache_dir, "references.txt")
        # 保存最终的DPO数据
        with open(os.path.join(args.cache_dir, "dpo_data.json"), "w", encoding='utf-8') as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)
        # 清理缓存文件
        for f in ["cached_predictions.txt", "cached_references.txt", "last_index.txt"]:
            try:
                os.remove(os.path.join(args.cache_dir, f))
            except FileNotFoundError:
                pass

    return evaluator.evaluate(predictions, references, verbose=True)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((EvaluationArguments,))
    eval_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # Load the dataset
    dataset = _resolve_dataset(eval_args.dataset_path) 

    if eval_args.lora_enable:
        model, tokenizer = load_trained_lora_model(
            model_name_or_path=eval_args.model_name_or_path,
            model_lora_path=eval_args.model_lora_path,
            load_bits=eval_args.load_bits,
        )
    else:
        model, tokenizer = load_trained_model(
            model_name_or_path=eval_args.model_name_or_path,
            pretrained_projectors_path=eval_args.projectors_path,
            load_bits=eval_args.load_bits,
        )
    
    score = _evaluate(model, tokenizer, dataset, eval_args)

    if eval_args.output_dir:
        _save_score(score, eval_args.output_dir)
