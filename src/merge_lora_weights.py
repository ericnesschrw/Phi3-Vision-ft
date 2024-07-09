import argparse
from utils import get_model_name_from_path, load_pretrained_model


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map="cpu"
    )

    if args.safe_serialization:
        from accelerate import Accelerator

        accel = Accelerator()
        # You could set the shard size whatever you want
        accel.save(model, args.save_model_path)
  
        model.config.save_pretrained(args.save_model_path)
        processor.save_pretrained(args.save_model_path)

    else:
        model.save_pretrained(args.save_model_path, safe_serialization=False)
        processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--safe-serialization", action="store_true")

    args = parser.parse_args()

    merge_lora(args)
