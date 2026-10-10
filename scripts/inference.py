"""
inference.py

Loads the base model plus the fine-tuned LoRA adapter and generates a
response to a user-supplied prompt in the proverb Q&A style used during
training.

Example:
    python scripts/inference.py \
        --base_model meta-llama/Llama-3.2-1B \
        --adapter_dir outputs/proverbs-lora \
        --prompt "Share a Yoruba proverb about patience and explain its meaning."
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--adapter_dir", type=str, default="outputs/proverbs-lora")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    prompt = PROMPT_TEMPLATE.format(instruction=args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = generated.split("### Response:\n", 1)[-1]
    print(response.strip())


if __name__ == "__main__":
    main()
