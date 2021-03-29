# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 04:29:08 2021

@author: nasheed
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from typing import Tuple, Dict
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from datasets import load_dataset
from transformers import (TFAutoModelForSequenceClassification,
                          AutoTokenizer,
                          logging as transformers_logging)

from callbacks import HFModelCheckPoint

transformers_logging.set_verbosity_warning()
MODEL_PATH = r"D:\Models\NLP\longformer\tf-longformer-base-4096"
DATA_COLS = ['input_ids', 'attention_mask', 'label']
SAVE_PATH = r"D:\Fine-tuned Models\NLP\longformer\tf-longformer-base-4096"


def prep_data(fpath: str,
              tpath: str,
              text_col: str = 'text',
              label_col: str = 'label',
              *args, **kwargs) -> Tuple[tf.data.Dataset, int]:
    dataset = load_dataset('csv', data_files=fpath, split='train',
                           cache_dir=kwargs.get('cache_dir', r"D:\hf_cache"))

    # id2label: Dict = dict(enumerate())
    num_classes = len(dataset.unique(label_col))

    # Encoding the labels
    le = LabelEncoder()
    le.fit(dataset.unique(label_col))
    # The id to label mapping that will be used to a set the appropriate config
    # param.
    id2label: Dict = dict(enumerate(le.classes_))

    # Split to train and test data
    # NOTE: Always make mappings after the splitting is done
    test_size = kwargs.get('test_size', 0.2)
    dataset = dataset.train_test_split(test_size=test_size,
                                       shuffle=True, seed=88)

    tokenizer = AutoTokenizer.from_pretrained(tpath)

    # Getting the tokenized text
    dataset = dataset.map(lambda e: tokenizer(e[text_col], truncation=True,
                                              padding=True, max_length=500),
                          batched=True)

    if num_classes >= 2:
        dataset = dataset.map(lambda e: {'label': to_categorical(le.transform(e[label_col]),
                                                                 num_classes=num_classes)},
                              batched=True)
    else:
        raise NotImplementedError("Only one class found. One class \
                                  classification is not yet supported.")

    # Conversion to tf.data.Dataset
    framework = kwargs.get('framework', 'tf')
    dataset.set_format(type=framework, columns=DATA_COLS)

    tr_data, ts_data = dataset.values()

    # Multi-dimensional elements i.e ['input_ids', 'attention_mask', 'label']
    # are stored as tf.RaggedTensor and need to be converted to an EagerTensor
    tr_feat = {x: tr_data[x].to_tensor(default_value=0,
                                       shape=[None, tokenizer.model_max_length]) 
               for x in ['input_ids', 'attention_mask']}
    tr_labels = tr_data['label'].to_tensor(default_value=0,
                                           shape=[None, num_classes])    

    ts_feat = {x: ts_data[x].to_tensor(default_value=0,
                                       shape=[None, tokenizer.model_max_length]) 
               for x in ['input_ids', 'attention_mask']}
    ts_labels = ts_data['label'].to_tensor(default_value=0,
                                           shape=[None, num_classes])

    batch_size = kwargs.get('batch_size', 32)
    val_batch_size = kwargs.get('val_batch_size', 32)
    trdataset = tf.data.Dataset.from_tensor_slices((tr_feat, tr_labels)).batch(batch_size)
    tsdataset = tf.data.Dataset.from_tensor_slices((ts_feat, ts_labels)).batch(val_batch_size)

    return trdataset, tsdataset, num_classes, id2label


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
                                                                     label2id=rev_class_map,
                                                                     gradient_checkpointing=True)

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
    # Get Data
    trdataset, tsdataset, num_classes, id2label = prep_data(fpath=r"D:\Datasets\NLP\IMDB Dataset.csv",
                                                            tpath=MODEL_PATH,
                                                            text_col='review',
                                                            label_col='sentiment',
                                                            batch_size=4,
                                                            val_batch_size=2)
    
    model = get_model(MODEL_PATH, num_classes=num_classes, class_map=id2label)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10,
                                                      restore_best_weights=True)

    checkpoint = HFModelCheckPoint(os.path.join(SAVE_PATH, 'epoch_{epoch}_valloss_{val_loss:.2f}'),
                                   save_best_only=True)
    history = model.fit(trdataset,
                        validation_data=tsdataset,
                        epochs=5,
                        callbacks=[checkpoint],
                        verbose=True)
