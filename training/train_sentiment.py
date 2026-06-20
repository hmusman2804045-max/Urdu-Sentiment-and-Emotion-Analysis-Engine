import os 
import sys 
import numpy as np 
import torch 
import torch .nn as nn 
from transformers import (
AutoTokenizer ,AutoModelForSequenceClassification ,
Trainer ,TrainingArguments ,DataCollatorWithPadding ,
EarlyStoppingCallback ,
)
from sklearn .metrics import accuracy_score ,f1_score ,precision_recall_fscore_support 


sys .path .append (os .path .dirname (os .path .abspath (__file__ )))
from dataset import UrduTextDataset 

NUM_LABELS =3 
MAX_LENGTH =128 


def compute_metrics (eval_pred ):
    logits ,labels =eval_pred 
    preds =np .argmax (logits ,axis =-1 )
    precision ,recall ,f1 ,_ =precision_recall_fscore_support (
    labels ,preds ,average ='weighted',zero_division =0 )
    macro_f1 =f1_score (labels ,preds ,average ='macro',zero_division =0 )
    return {
    'accuracy':accuracy_score (labels ,preds ),
    'f1':f1 ,
    'macro_f1':macro_f1 ,
    'precision':precision ,
    'recall':recall ,
    }


class WeightedTrainer (Trainer ):
    """Trainer that applies class weights (+ optional label smoothing) in the loss."""
    def __init__ (self ,*args ,class_weights =None ,label_smoothing =0.0 ,**kwargs ):
        super ().__init__ (*args ,**kwargs )
        self .class_weights =class_weights 
        self .label_smoothing =label_smoothing 

    def compute_loss (self ,model ,inputs ,return_outputs =False ,**kwargs ):
        labels =inputs .pop ("labels")
        outputs =model (**inputs )
        weight =(self .class_weights .to (model .device )
        if self .class_weights is not None else None )
        loss_fn =nn .CrossEntropyLoss (
        weight =weight ,label_smoothing =self .label_smoothing )
        loss =loss_fn (outputs .logits .view (-1 ,NUM_LABELS ),labels .view (-1 ))
        return (loss ,outputs )if return_outputs else loss 


def main ():
    print ("="*60 )
    print ("  Phase 4: Training Sentiment Model")
    print ("="*60 )


    torch .set_num_threads (os .cpu_count ()or 1 )

    model_name ="xlm-roberta-base"


    print ("\n[1/5] Loading tokenizer and XLM-RoBERTa model...")
    tokenizer =AutoTokenizer .from_pretrained (model_name )
    model =AutoModelForSequenceClassification .from_pretrained (
    model_name ,num_labels =NUM_LABELS )


    print ("\n[2/5] Loading datasets...")
    base =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
    data_dir =os .path .join (base ,'data')
    train_files =[
    os .path .join (data_dir ,'roman_urdu_sentiment_train.csv'),
    os .path .join (data_dir ,'urdu_sentiment_corpus_train.csv'),
    ]
    val_files =[
    os .path .join (data_dir ,'roman_urdu_sentiment_val.csv'),
    os .path .join (data_dir ,'urdu_sentiment_corpus_val.csv'),
    ]

    train_dataset =UrduTextDataset (train_files ,tokenizer ,max_length =MAX_LENGTH ,task ="sentiment")
    val_dataset =UrduTextDataset (val_files ,tokenizer ,max_length =MAX_LENGTH ,task ="sentiment")
    print (f"Train samples: {len (train_dataset )} | Val samples: {len (val_dataset )}")


    class_weights =train_dataset .get_class_weights (NUM_LABELS )
    print (f"Class weights: {class_weights .tolist ()}")
    data_collator =DataCollatorWithPadding (tokenizer =tokenizer )


    print ("\n[3/5] Setting up Training Arguments...")
    output_dir =os .path .join (base ,'models','sentiment_model')

    training_args =TrainingArguments (
    output_dir =output_dir ,
    num_train_epochs =6 ,
    per_device_train_batch_size =8 ,
    per_device_eval_batch_size =16 ,
    gradient_accumulation_steps =4 ,
    learning_rate =2e-5 ,
    warmup_ratio =0.1 ,
    weight_decay =0.01 ,
    lr_scheduler_type ="cosine",
    max_grad_norm =1.0 ,

    dataloader_pin_memory =False ,
    dataloader_num_workers =0 ,
    logging_dir ='./logs/sentiment',
    logging_steps =50 ,
    eval_strategy ="epoch",
    save_strategy ="epoch",
    save_total_limit =2 ,
    load_best_model_at_end =True ,
    metric_for_best_model ="macro_f1",
    greater_is_better =True ,
    seed =42 ,
    report_to ="none",
    )


    print ("\n[4/5] Initializing Trainer...")
    trainer =WeightedTrainer (
    model =model ,
    args =training_args ,
    train_dataset =train_dataset ,
    eval_dataset =val_dataset ,
    processing_class =tokenizer ,
    data_collator =data_collator ,
    compute_metrics =compute_metrics ,
    class_weights =class_weights ,
    label_smoothing =0.1 ,
    callbacks =[EarlyStoppingCallback (early_stopping_patience =2 )],
    )


    print ("\n[5/5] Starting training loop...")
    trainer .train ()

    print (f"\nTraining complete! Saving best model to {output_dir }")
    trainer .save_model (output_dir )
    tokenizer .save_pretrained (output_dir )
    print ("Sentiment model saved successfully!")


if __name__ =="__main__":
    main ()
