Required libraries:
- pip install git+https://github.com/huggingface/transformers
- pip install accelerate peft bitsandbytes trl 

Instructions:
1. Clone the repository.
2. Run `train_s1.py.` It only trains on one GPU.
3. Use `inference.ipynb` to load the finetuned model and make inferences with it.

Training details: Trained on one A40 GPU (48 GB) hosted from https://cloud.vast.ai. It's extremely easy to use.

Cost: Less than $5!

HuggingFace model url: https://huggingface.co/jk23541/qwem7b_4bit_s1

Colab url (may have slight differences to `train_s1.py`): https://colab.research.google.com/drive/1Ju_wsjFVxAhGRGaoUg7OylnAOVroSo6L?usp=sharing
