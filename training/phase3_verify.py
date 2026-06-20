import os 
import sys 


sys .path .append (os .path .dirname (os .path .abspath (__file__ )))

from transformers import AutoTokenizer 
from dataset import UrduTextDataset 
from torch .utils .data import DataLoader 

def main ():
    print ("Loading XLM-RoBERTa Tokenizer...")
    tokenizer =AutoTokenizer .from_pretrained ("xlm-roberta-base")

    data_dir =os .path .join (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))),'data')

    print ("\n--- Testing Sentiment Dataset ---")
    sentiment_train_files =[
    os .path .join (data_dir ,'roman_urdu_sentiment_train.csv'),
    os .path .join (data_dir ,'urdu_sentiment_corpus_train.csv')
    ]

    sentiment_dataset =UrduTextDataset (
    csv_paths =sentiment_train_files ,
    tokenizer =tokenizer ,
    max_length =128 ,
    task ="sentiment"
    )

    print (f"Total Sentiment Training Samples: {len (sentiment_dataset )}")


    sample =sentiment_dataset [0 ]
    print (f"\nSample Text: {sample ['text']}")
    print (f"Sample Label: {sample ['label']} (0=Neg, 1=Neu, 2=Pos)")
    print (f"Input IDs shape: {sample ['input_ids'].shape }")
    print (f"Attention Mask shape: {sample ['attention_mask'].shape }")


    loader =DataLoader (sentiment_dataset ,batch_size =4 ,shuffle =True )
    batch =next (iter (loader ))
    print (f"\nBatch Input IDs shape: {batch ['input_ids'].shape }")
    print (f"Batch Labels: {batch ['label']}")

    print ("\n--- Testing Emotion Dataset ---")
    emotion_train_files =[
    os .path .join (data_dir ,'semeval_emotion_train.csv')
    ]

    emotion_dataset =UrduTextDataset (
    csv_paths =emotion_train_files ,
    tokenizer =tokenizer ,
    max_length =128 ,
    task ="emotion"
    )

    print (f"Total Emotion Training Samples: {len (emotion_dataset )}")
    emotion_sample =emotion_dataset [0 ]
    print (f"Sample Label: {emotion_sample ['label']} (0=Joy, 1=Anger, 2=Fear, 3=Sadness)")

    print ("\nPhase 3 Verification Successful! Dataset class and tokenization work perfectly.")

if __name__ =="__main__":
    main ()
