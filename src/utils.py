from src.phi3_vision import (
    Phi3VForCausalLM,
    Phi3VConfig,
    Phi3VProcessor,
    Phi3VImageProcessor,
)
import torch
from transformers import BitsAndBytesConfig
import warnings


from transformers import AutoTokenizer


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    print(f"Loading model from path: {model_path}")
    print(f"Base model: {model_base}")
    print(f"Model name: {model_name}")

    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["_attn_implementation"] = "flash_attention_2"

    if "lora" in model_name.lower() and model_base is None:
        warnings.warn(
            "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument."
        )

    if "lora" in model_name.lower() and model_base is not None:
        print("Loading LoRA model")
        lora_cfg_pretrained = Phi3VConfig.from_pretrained(model_path)
        print(f"Loading image processor from: {model_base}")
        image_processor = Phi3VImageProcessor.from_pretrained(model_base)
        print(f"Loading tokenizer from: {model_base}")
        tokenizer = AutoTokenizer.from_pretrained(model_base)
        print("Loading Phi3-Vision from base model...")
        model = Phi3VForCausalLM.from_pretrained(
            model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
        )

        # ... (rest of the LoRA loading code)

        processor = Phi3VProcessor(image_processor=image_processor, tokenizer=tokenizer)
    else:
        print("Loading full model")
        print(f"Loading image processor from: {model_path}")
        image_processor = Phi3VImageProcessor.from_pretrained(model_path)
        print(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = Phi3VProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = Phi3VForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )

    return processor, model


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
