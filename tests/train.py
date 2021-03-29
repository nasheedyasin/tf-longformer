# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:49:18 2021

@author: nasheed
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
from typing import Dict
from transformers import (TFAutoModelForSequenceClassification,
                          AutoTokenizer)

from callbacks import HFModelCheckPoint

def get_model(base_path: str,
              num_classes: int = 2,
              class_map: Dict[str, str] = {},
              freeze_base: bool = True,
              show_summary: bool = False) -> TFAutoModelForSequenceClassification:

    if len(class_map) > 0:
        rev_class_map: Dict[str, str] = {v: k  for k, v in class_map.items()}
        model = TFAutoModelForSequenceClassification.from_pretrained(base_path,
                                                                     num_labels=num_classes,
                                                                     id2label=class_map,
                                                                     label2id=rev_class_map)
    else:
        model = TFAutoModelForSequenceClassification.from_pretrained(base_path,
                                                                     num_labels=num_classes)

    if freeze_base:
        model.layers[0].trainable = False    

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Set from_logits=True as there is no Softmax/Sigmoid layer added in Huggingface Models  
    model.compile(optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True) if num_classes > 2
                      else tf.keras.losses.BinaryCrossentropy(from_logits=True))

    if show_summary:
        model.summary()
    return model

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(r"D:\Models\NLP\longformer\tf-longformer-base-4096")
    model = get_model(r"D:\Models\NLP\longformer\tf-longformer-base-4096", 3,
                      {0: 'label1', 1: 'label2', 2: 'TomBombadil'})

    dummy_data = ["Some data.", "Some more data."]
    dummy_data = tokenizer(dummy_data, padding=True, return_tensors='tf')
    dummy_data = [dummy_data.input_ids, dummy_data.attention_mask]
    dummy_classes = tf.convert_to_tensor([[0, 0, 1], [0, 1, 0]],
                                         dtype=tf.float32)

    checkpoint = HFModelCheckPoint('.model/model/model_epoch_{epoch}_loss_{loss:.2f}')
    model.fit(dummy_data, dummy_classes, epochs=4, batch_size=64,
              callbacks=[checkpoint])
