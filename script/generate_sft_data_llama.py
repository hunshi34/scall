#currently it only support llama-3.1-instruct-8b


import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

def get 


def generate(arg):
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])
    
    


if __name__ == '__main__':
    parser.add_argument("--model-path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("data_path", type=str, default="")
    args = parser.parse_args()
    generate(args)