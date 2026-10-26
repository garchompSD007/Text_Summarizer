# model dependencies
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.cuda import is_available
import torch
import re

from pathlib import Path
CURRENT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    CURRENT_ROOT / "saved_summary_model"
)

print("Loading model from:", MODEL_PATH)
print("Folder exists:", MODEL_PATH.is_dir())

# model and tokeniser
model=T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer=T5Tokenizer.from_pretrained(MODEL_PATH)
# model=T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
# tokenizer=T5Tokenizer.from_pretrained(MODEL_PATH)


def clean_data(text):
  text = re.sub(r'\r\n', ' ', text)
  text = re.sub(r'\s+', ' ', text)
  text = re.sub(r'<.*?>', ' ', text)
  text = text.strip().lower()
  return text
def summarize_dialogue(dialogue : str):      
    # # setting device
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")

    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")

    dialogue = clean_data(dialogue) # clean

    # tokenise
    inputs = tokenizer(dialogue, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
    
    # Generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_length=150, 
        num_beams=4, # it will create 4 different summaries and give the best out of it
        early_stopping=True
    )
    
    # token ids convert to summary => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    
    return summary


# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)
# print(device)