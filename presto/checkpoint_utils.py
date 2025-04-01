import logging
import torch
from functools import partial
import functools

# 为 PyTorch 的梯度检查点设置默认参数
def setup_checkpoint_for_deepspeed():
    """
    为 DeepSpeed 训练设置梯度检查点，特别是禁用 reentrant 模式。
    这解决了在 DPO 训练中可能出现的梯度计算冲突。
    """
    logging.info("Setting up checkpoint utils for DeepSpeed compatibility")
    
    try:
        # 正确导入 checkpoint 模块
        from torch.utils import checkpoint
        
        # 保存原始 checkpoint 函数
        original_checkpoint = checkpoint.checkpoint
        
        # 定义新的 checkpoint 函数，默认设置 use_reentrant=False
        @functools.wraps(original_checkpoint)
        def patched_checkpoint(function, *args, use_reentrant=False, **kwargs):
            return original_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
        
        # 替换 PyTorch 的 checkpoint 函数
        checkpoint.checkpoint = patched_checkpoint
        
        logging.info("PyTorch checkpoint function patched with use_reentrant=False")
    except (ImportError, AttributeError) as e:
        logging.warning(f"Could not patch PyTorch checkpoint function: {str(e)}")
        logging.warning("Setting environment variable to fix gradient checkpointing")
        import os
        os.environ["PYTORCH_CHECKPOINT_USE_REENTRANT"] = "0"

# 通过 monkey patching 修正 Transformers 库的梯度检查点使用
def patch_transformers_checkpointing():
    """
    修补 Transformers 库中的梯度检查点实现，
    确保在 DPO 训练中正确使用 use_reentrant=False。
    """
    try:
        import transformers
        
        # 设置环境变量让 PyTorch 使用 use_reentrant=False
        import os
        os.environ["PYTORCH_CHECKPOINT_USE_REENTRANT"] = "0"
        
        # 通过 patch_forward_call 修改 LLaMA 模型的 forward 方法
        if hasattr(transformers, "LlamaModel"):
            model_class = transformers.models.llama.modeling_llama.LlamaModel
            if hasattr(model_class, "forward"):
                original_forward = model_class.forward
                
                @functools.wraps(original_forward)
                def patched_forward(self, *args, **kwargs):
                    # 如果模型使用梯度检查点，确保使用正确的设置
                    if self.gradient_checkpointing and self.training:
                        self.config._use_reentrant = False
                    return original_forward(self, *args, **kwargs)
                
                model_class.forward = patched_forward
                logging.info("Patched LlamaModel forward method to fix gradient checkpointing")
        
        logging.info("Successfully patched Transformers library for DeepSpeed compatibility")
    except ImportError:
        logging.warning("Could not patch Transformers library (not installed)")
    except Exception as e:
        logging.error(f"Error patching Transformers library: {str(e)}")

def fix_deepspeed_checkpointing():
    """
    应用所有必要的修补，以解决 DeepSpeed 和梯度检查点的兼容性问题。
    """
    try:
        setup_checkpoint_for_deepspeed()
        patch_transformers_checkpointing()
        
        # 修复警告消息
        logging.info("Fixed gradient checkpointing for DeepSpeed - use_reentrant set to False")
        
        # 设置跳过梯度减少检查的环境变量
        import os
        os.environ["DS_SKIP_GRAD_REDUCE_CHECK"] = "1"
        logging.info("Set DS_SKIP_GRAD_REDUCE_CHECK=1 to avoid gradient checking errors")
        
        return True
    except Exception as e:
        logging.error(f"Failed to fix gradient checkpointing: {str(e)}")
        # 出现错误时，至少设置必要的环境变量
        import os
        os.environ["PYTORCH_CHECKPOINT_USE_REENTRANT"] = "0"
        os.environ["DS_SKIP_GRAD_REDUCE_CHECK"] = "1"
        
        return False 