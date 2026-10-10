"""
app.py

Simple Gradio demo for the Nigerian Proverbs LLM. Loads the base model plus
the fine-tuned LoRA adapter and lets a user type a request (e.g. "Share an
Igbo proverb about honesty") and get a generated proverb + explanation.

Usage:
    python app.py --base_model meta-llama/Llama-3.2-1B --adapter_dir outputs/proverbs-lora

If no adapter is found at --adapter_dir, the app falls back to running the
base model directly (useful for testing the UI before training finishes).
"""

import argparse
import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

EXAMPLES = [
    "Share a Yoruba proverb about patience and explain its meaning.",
    "Give me an Igbo proverb about unity.",
    "Explain a Hausa proverb about honesty.",
    "Share a proverb about the importance of community.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--adapter_dir", type=str, default="outputs/proverbs-lora")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_model(base_model: str, adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if os.path.isdir(adapter_dir) and os.path.exists(
        os.path.join(adapter_dir, "adapter_config.json")
    ):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"Loaded LoRA adapter from {adapter_dir}")
    else:
        print(
            f"No adapter found at {adapter_dir} — running base model only. "
            "Train first with scripts/train_lora.py for proverb-tuned output."
        )

    model.eval()
    return tokenizer, model


def build_generate_fn(tokenizer, model):
    def generate(instruction: str, max_new_tokens: int, temperature: float):
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded.split("### Response:\n", 1)[-1].strip()

    return generate


def main():
    args = parse_args()
    tokenizer, model = load_model(args.base_model, args.adapter_dir)
    generate_fn = build_generate_fn(tokenizer, model)

    with gr.Blocks(title="Nigerian Proverbs LLM") as demo:
        gr.Markdown(
            "# 🇳🇬 Nigerian Proverbs LLM\n"
            "A LoRA-fine-tuned language model that generates Nigerian proverbs "
            "(Yoruba, Igbo, Hausa) with translations and explanations."
        )
        with gr.Row():
            with gr.Column(scale=2):
                instruction = gr.Textbox(
                    label="Your request",
                    placeholder="e.g. Share a Yoruba proverb about patience",
                    lines=2,
                )
                with gr.Row():
                    max_new_tokens = gr.Slider(
                        50, 400, value=200, step=10, label="Max new tokens"
                    )
                    temperature = gr.Slider(
                        0.1, 1.2, value=0.7, step=0.05, label="Temperature"
                    )
                submit_btn = gr.Button("Generate", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=instruction)
            with gr.Column(scale=2):
                output = gr.Textbox(label="Generated response", lines=12)

        submit_btn.click(
            fn=generate_fn,
            inputs=[instruction, max_new_tokens, temperature],
            outputs=output,
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
