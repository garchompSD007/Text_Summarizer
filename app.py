from fastapi import FastAPI, Request
from pydantic import BaseModel

# UI and client side
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# from models-training.summary_model import summarize_dialogue
from models_training.summary_model import summarize_dialogue



app = FastAPI(title="text summarizer app", description="")


# templating
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


# Input Schema
class DialogueInput(BaseModel):
   dialogue: str



# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request,name="index.html",context=None)

@app.post("/summarize/")
async def summarize(dialogue: DialogueInput):
   summary = summarize_dialogue(dialogue.dialogue)
   return {"summary":summary}
    