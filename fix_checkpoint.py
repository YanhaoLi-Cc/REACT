import torch
import logging
import sys
import os

# 设置 PyTorch 的梯度检查点参数，解决在 DPO 训练中的梯度计算冲突
def fix_checkpoint():
    # 导入并应用我们创建的修补函数
    sys.path.append(os.getcwd())
    from presto.checkpoint_utils import fix_deepspeed_checkpointing
    fix_deepspeed_checkpointing()
    
    # 设置环境变量，告诉 DeepSpeed 允许梯度共享
    os.environ["DS_SKIP_GRAD_REDUCE_CHECK"] = "1"
    
    print("Gradient checkpointing patched successfully!")

if __name__ == "__main__":
    fix_checkpoint()
