import math
from collections import Counter
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import nltk
from pymorphy2 import MorphAnalyzer
import nltk
nltk.download('punkt_tab')

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

morph = MorphAnalyzer()

documents_words = []

def preprocess_text(text: str) -> List[str]:
    tokens = nltk.word_tokenize(text.lower(), language="russian")
    
    processed_tokens = []
    for token in tokens:
        if token.isalpha():
            lemma = morph.parse(token)[0].normal_form
            processed_tokens.append(lemma)
    
    return processed_tokens

def calculate_tf_idf(words: list):

    word_counts = Counter(words)
    total_words = len(words)
    tf = {word: count/total_words for word, count in word_counts.items()}
    
    idf = {}
    total_docs = len(documents_words)
    for word in word_counts:
        docs_with_word = sum(1 for doc in documents_words if word in doc)
        idf[word] = math.log(total_docs / (1 + docs_with_word)) if total_docs > 0 else 0
    
    return tf, idf


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    upload_dir = Path("static/uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    with file_path.open('r', encoding='utf-8') as f:
        words = preprocess_text(f.read())
        documents_words.append(words)

    tf, idf = calculate_tf_idf(words)
    
    sorted_words = sorted(idf.items(), key=lambda x: x[1], reverse=True)[:50]
    results = [{
        'word': word,
        'tf': tf.get(word, 0),
        'idf': idf_score
    } for word, idf_score in sorted_words]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "filename": file.filename
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)