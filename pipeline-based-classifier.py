# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 03:40:33 2021

@author: nashe
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from transformers import TextClassificationPipeline, TFAutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = r"D:\Fine-tuned Models\NLP\bert\tf-distilbert-base-uncased\epoch_1_loss_0.39"

# Feature extraction pipeline
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(r"D:\Models\NLP\bert\tf-distilbert-base-uncased")

pipeline = TextClassificationPipeline(model=model,
                                      tokenizer=tokenizer,
                                      framework='tf',
                                      device=0)

result = pipeline("It was a good watch. But a little boring.")[0]
