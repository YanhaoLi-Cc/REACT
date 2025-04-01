import pandas as pd
import json
import numpy as np

# 读取parquet文件
df = pd.read_parquet('/home/liyanhao/chemllm/REACT/datasets/MolInst_RS_125K_SMILES-MMChat/data/test-00000-of-00001.parquet')

# 获取前三条记录并转换为字典
first_three = df.head(3).to_dict('records')

# 创建一个函数来转换numpy类型
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# 转换数据
converted_data = [convert_numpy_types(record) for record in first_three]

# 保存为JSON文件
with open('逆合成.json', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)