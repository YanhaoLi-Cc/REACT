from datasets import load_from_disk, load_dataset, Dataset as HFDataset



print("start")
# dataset = load_dataset("OpenMol/USPTO_RXN_Interleaved")
# dataset = load_dataset('local_path', data_dir="/home/liyanhao/chemllm/REACT/datasets/USPTO_RXN_Interleaved")
dataset = load_dataset("/home/liyanhao/chemllm/REACT/datasets/USPTO_RXN_Interleaved")

# else:
#     dataset = load_dataset(args.repo_id)
    
    
# if split in dataset.keys():
#     return dataset[split]
# elif split == "eval" and "validation" in dataset.keys():
#     return dataset["validation"]
# elif split == "eval" and "valid" in dataset.keys():
#     return dataset["valid"]
# elif split == "eval" and "val" in dataset.keys():
#     return dataset["val"]
# else:
#     return dataset["train"]
