# -*- coding: utf-8 -*-

from app import app
from flask import render_template, request
from model import embedder, tokenizer

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def index():
    exp = ""
    if request.method == 'POST':
        text_1 = tokenizer(request.form['entry_1'])
        text_2 = tokenizer(request.form['entry_2'])
        method = request.form['model']
        if any(not v for v in [text_1, text_2]):
            raise ValueError("Please do not leave text fields blank.")
         
        if method != "base":
            exp = embedder(method, text_1, text_2)
            
    return render_template('index.html', exp=exp, entry_1=text_1, entry_2=text_2, embed=method)

