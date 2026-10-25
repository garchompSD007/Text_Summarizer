from fastapi import FastAPI, Request
from pydantic import BaseModel

# UI and client side
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# model dependencies
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.cuda import is_available
import torch
import re


app = FastAPI(title="text summarizer app", description="")

# model and tokeniser
model=T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer=T5Tokenizer.from_pretrained("./saved_summary_model")

# setting device
if torch.cuda.is_available():
  device = torch.device("cuda")

else:
  device = torch.device("cpu")

# templating
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


# Input Schema
# class DialogueInput(BaseModel):
#    dialogue: str



def clean_data(text):
  text = re.sub(r'\r\n', ' ', text)
  text = re.sub(r'\s+', ' ', text)
  text = re.sub(r'<.*?>', ' ', text)
  text = text.strip().lower()
  return text
def summarize_dialogue(dialogue : str):
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



# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request,name="index.html",context=None)

@app.post("/summarize/")
async def summarize(dialogue: str):
   summary = summarize_dialogue(dialogue)
   return {"summary":summary}
    