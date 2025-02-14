Required libraries:
pip install git+https://github.com/huggingface/transformers
pip install accelerate peft bitsandbytes trl 

Instructions:
1. Clone the repository.
2. Run `train_s1.py.` It only trains on one GPU.
3. Use `inference.ipynb` to load the finetuned model and make inferences with it.

HuggingFace model url: https://huggingface.co/jk23541/qwem7b_4bit_s1

Training details: Trained on one A40 GPU (48 GB) hosted from vast.ai. It's extremely easy to use.

Cost: Less than $5!
