import os
import datasets
import transformers
import peft
import trl

import torch
import concurrent.futures
from tqdm import tqdm

EFFICIENT_TRAINING = True # We run out of GPU memory, otherwise!
SMALL_DATASET = True

if SMALL_DATASET:
    # Use the below dataset if GPU memory is limited
    dataset = datasets.load_from_disk('binned_dataset_1000_2000')
else:
    dataset = datasets.load_dataset("simplescaling/s1K_tokenized")['train']

import re
def check_text_structure(text):
    # Step 1: Split the text into sections
    sections = text.split("<|im_start|>")
    
    # Step 2: Define regex for each part
    system_pattern = r"system\nYou are.*\.\<\|im_end\|\>\n"
    user_pattern = r"user\n.*\n" # <|im_end|> before \n
    assistant_pattern = r"assistant\n"
    think_pattern = r"think\n.*\n"
    answer_pattern = r"answer\n.*\n" # <|im_end|> before last \n
    
    # Step 3: Validate sections one by one
    assert re.match(system_pattern, sections[1])
    assert re.match(user_pattern, sections[2])
    assert re.match(assistant_pattern, sections[3])
    assert re.match(think_pattern, sections[4])
    assert re.match(answer_pattern, sections[5])

reformatted_examples = []
for ex in tqdm(dataset['text']):
    sections = ex.split("<|im_start|>")
    assert len(sections) == 6
    # System: system\n...\n
    sections[1] = "system\n" + sections[1][len("system "):] + "\n"
    sections[2] = "user\n" + sections[2][len("user "):] + "\n"
    sections[3] = "assistant\n"
    sections[4] = "think\n" + sections[4][len("think "):] + "\n"
    sections[5] = "answer\n" + sections[5][len("answer "):] + "\n"
    ex = "<|im_start|>".join(sections)
    reformatted_examples.append(ex)
dataset = datasets.Dataset.from_dict({"text": reformatted_examples})

for ex in tqdm(dataset['text']):
    check_text_structure(ex)

print('System prompt:\n', dataset['text'][0].split("<|im_start|>")[1])
print('Question:\n', dataset['text'][0].split("<|im_start|>")[2][:100])
print('Thinking:\n', dataset['text'][0].split("<|im_start|>")[4][:300])
print('Answer:\n', dataset['text'][0].split("<|im_start|>")[5][:100])

base_model_name = "Qwen/Qwen2.5-3B-Instruct"

def get_tokenizer():
  tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|fim_pad|>'})
  return tokenizer

tokenizer = get_tokenizer()
instruction_template = "<|im_start|>user"
response_template = "<|im_start|>assistant\n"
tokenizer.pad_token = "<|fim_pad|>" # Use a token that is never used

for _prompt in tqdm(dataset['text']):
  _input_ids = tokenizer(_prompt)["input_ids"]
  _collator = trl.DataCollatorForCompletionOnlyLM(
      instruction_template=instruction_template,
      response_template=response_template,
      tokenizer=tokenizer,
      mlm=False
  )
  x = _collator([_input_ids])

import random
CUTOFF_LENGTH = 2300
if SMALL_DATASET:
    _examples_less_than_cutoff = []
    
    for example in tqdm(dataset["text"], desc="Calculating max sequence length"):
        # Tokenize without truncation; add special tokens if needed.
        tokenized = tokenizer(example, truncation=False, add_special_tokens=True)
        seq_length = len(tokenized["input_ids"])
        if seq_length < CUTOFF_LENGTH:
            _examples_less_than_cutoff.append((seq_length, example))

    examples_less_than_cutoff = sorted(_examples_less_than_cutoff, key=lambda x: x[0], reverse=True)
    
    sample_size = 1200
    weights = [ex[0] for ex in examples_less_than_cutoff]
    examples_less_than_cutoff = random.choices(
        examples_less_than_cutoff, weights=weights, k=sample_size
    )
    
    filtered_dataset = datasets.Dataset.from_dict({"text": [ex[1] for ex in examples_less_than_cutoff]})

    print(f"Full dataset size: {len(_examples_less_than_cutoff)}")
    print(f"Filtered dataset train size: {len(filtered_dataset)}")
else:
    filtered_dataset = dataset
    print(f"Full dataset size: {len(dataset)}")
    print(f"Filtered dataset train size: {len(filtered_dataset)}")

print(filtered_dataset['text'][0][:2000])
print("...")
print(filtered_dataset['text'][0][-3000:])

compute_dtype = getattr(torch, "float16")
quant_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype, # Run computations in 16-bit float
    bnb_4bit_use_double_quant=False, # Use True for additional memory saving
) if EFFICIENT_TRAINING else None

model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config if EFFICIENT_TRAINING else None,
    attn_implementation="flash_attention_2",
    device_map={"": 0}
)
if EFFICIENT_TRAINING:
    model.config.use_cache = False
    model.config.pretraining_tp = 1

output_dir = f"qwem7b_{'4bit' if EFFICIENT_TRAINING else 'full'}_s1"
vllm_dir = f"qwem7b_vllm"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(vllm_dir, exist_ok=True)

args = trl.SFTConfig(
    learning_rate=1e-5,
    num_train_epochs=5,
    weight_decay=1e-4,
    per_device_train_batch_size=1,
    max_steps=-1,
    dataset_text_field='text',
    logging_steps=25,
    save_steps=3000,
    output_dir=output_dir,
    max_seq_length=CUTOFF_LENGTH + 500 if SMALL_DATASET else 32768,
    gradient_accumulation_steps=1,
)

if EFFICIENT_TRAINING:
    args.optim = "paged_adamw_32bit"
    args.fp16=False
    args.bf16=False
    args.packing=False

peft_config = peft.LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.02,
) if EFFICIENT_TRAINING else None

if EFFICIENT_TRAINING:
    model = peft.get_peft_model(model, peft_config)

collator = trl.DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False
)

trainer = trl.SFTTrainer(
    model,
    train_dataset=filtered_dataset['train'] if 'train' in filtered_dataset else filtered_dataset,
    eval_dataset=filtered_dataset['eval'] if 'eval' in filtered_dataset else filtered_dataset,
    args=args,
    data_collator=collator,
    peft_config=peft_config if EFFICIENT_TRAINING else None,
)

trainer.remove_callback(transformers.integrations.WandbCallback)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
trainer.accelerator.wait_for_everyone()
