from presto.language_models.llama import (
    LlamaLMMForCausalLM, 
)
from presto.language_models.mistral import (
    MistralLMMForCausalLM,
)
from presto.language_models.phi import (
    PhiLMMForCausalLM,
)

LANGUAGE_MODEL_CLASSES = [MistralLMMForCausalLM, LlamaLMMForCausalLM, PhiLMMForCausalLM]

LANGUAGE_MODEL_NAME_TO_CLASS = {cls.__name__: cls for cls in LANGUAGE_MODEL_CLASSES}
