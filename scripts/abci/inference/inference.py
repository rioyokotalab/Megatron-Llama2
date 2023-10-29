from transformers import (  # noqa: F401
    LlamaForCausalLM,
    LlamaTokenizer,
)
import torch
import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama-Recipes Inference")

    parser.add_argument("--hf-model-path", type=str, default=None, help="huggingface checkpoint path")
    parser.add_argument("--hf-tokenizer-path", type=str, default=None, help="huggingface tokenizer path")
    parser.add_argument("--hf-token", type=str, default=None, help="huggingface token")
    parser.add_argument("--hf-cache-dir", type=str, help="huggingface cache directory")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--limit-inference-case", type=int, default=None)

    args = parser.parse_args()
    return args


def load_jsonl(file_path: str) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main() -> None:
    # argument parse
    args = parse_args()

    # load model & tokenizer
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.hf_model_path, token=args.hf_token, cache_dir=args.hf_cache_dir
    )
    if torch.cuda.is_available():
        model.to('cuda')  # type: ignore

    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.hf_tokenizer_path, token=args.hf_token, cache_dir=args.hf_cache_dir
    )
    input_datasets: list[str] = ["United States of America"]

    # inference
    with torch.no_grad():
        for index, text in enumerate(input_datasets):

            input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

            output_ids = []
            for _ in range(args.num_samples):
                ids = model.generate(  # type: ignore
                    input_ids.to(model.device),  # type: ignore
                    max_length=2048,
                    pad_token_id=tokenizer.pad_token_id,
                )
                output_ids.append(ids)

            # デコードして結果を出力 (オプション)
            decoded_outputs = [tokenizer.decode(ids[0][len(input_ids[0]):], skip_special_tokens=True) for ids in output_ids]

            print(f"{index}: {decoded_outputs}")


if __name__ == "__main__":
    main()
