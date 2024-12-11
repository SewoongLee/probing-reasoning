import os
import torch
import transformers
from src.util.json_io import *
import random
from src.util.gsm8k_helper import *
import re
from datetime import datetime


rseed = 42
random.seed(rseed)

HF_TOKEN = "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"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"


model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(model, model.config)


def generate_text(input_text):
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 512,
            do_sample=True,
            top_k=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


question_start = '[INST] Question: '
question_end = ' [\\INST] '
answer_start = 'Answer: '
answer_end = '\n'

oneshot_start = ''
oneshot_end = ''

def generate_oneshot_prompt(question, answer):
    return f'{oneshot_start}{question_start}{question}{question_end}{answer_start}{answer}{answer_end}{oneshot_end}'


train_qnas = load_jsonlines(f'train.jsonl')
test_qnas = load_jsonlines(f'test.jsonl')


nshot_prompt = ""
for i in random.sample(range(len(train_qnas)), 8):
    nshot_prompt += generate_oneshot_prompt(train_qnas[i]['question'], train_qnas[i]['answer'])

def get_prompt(question, nshot_prompt=nshot_prompt):
    cot_prompt = " Let's think step by step."
    return f"{nshot_prompt}{oneshot_start}{question_start}{question}{cot_prompt}{question_end}{answer_start}"


current_time = datetime.now().strftime("%y%m%d_%H%M%S")
errors_log_path = f"log/errors-{model_name.replace('/', '-')}-{current_time}.txt"
with open(errors_log_path, 'w', encoding='utf-8') as log_file:
    log_file.write(f"{model}\n\n")
    log_file.write(f"{model.config}\n\n")
    log_file.write(f"[random seed]: {rseed}\n\n")
    log_file.write(f"[Nshot prompt]:\n{nshot_prompt}\n\n")
    log_file.write("=" * 40 + "\n\n")


correct_count = total_count = 0

for i, qna in enumerate(test_qnas[:100]):
    input_text = get_prompt(qna['question'])


    generated_text = generate_text(input_text)

    pattern = re.escape(question_end) + re.escape(answer_start) + r"(.*?)(?:\n|\Z)"
    match = re.search(pattern, generated_text, re.DOTALL)
    generated_answer = match.group(1).strip() if match else "Answer not found"

    generated_answer_int = extract_num_from_ans(generated_answer)
    ground_truth_int = extract_num_from_ans(qna['answer'])

    total_count += 1
    if generated_answer_int == ground_truth_int:
        correct_count += 1
        print('o', end='')
    else:
        print('x', end='')
        with open(errors_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"[Question (i:{i})]\n{qna['question']}\n\n")
            log_file.write(f"[Response {model_name}]\n{generated_answer}\n\n")
            log_file.write(f"[Ground Truth]\n{qna['answer']}\n\n")
            log_file.write(f"[Accuracy]\n{correct_count / total_count * 100:.2f}\n\n\n")


print(f"\nAccuracy: {correct_count / total_count * 100:.2f}%")
print(f"Errors logged to {errors_log_path}")