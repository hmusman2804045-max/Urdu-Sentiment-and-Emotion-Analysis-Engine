import os 
import sys 
import numpy as np 
import torch 
import matplotlib 


matplotlib .use ("Agg")
import matplotlib .pyplot as plt 
import seaborn as sns 

from torch .utils .data import DataLoader 
from transformers import (
AutoTokenizer ,AutoModelForSequenceClassification ,DataCollatorWithPadding ,
)
from sklearn .metrics import (
classification_report ,confusion_matrix ,accuracy_score ,f1_score ,
)


BASE_DIR =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
sys .path .append (os .path .join (BASE_DIR ,'training'))
from dataset import UrduTextDataset 

MAX_LENGTH =128 
BATCH_SIZE =16 














ROMAN_SENTIMENT_LABELS =["Positive","Negative","Neutral"]
URDU_SENTIMENT_LABELS =["Negative","Neutral","Positive"]
EMOTION_LABELS =["Joy","Anger","Fear","Sadness"]


def run_predictions (model ,dataset ,tokenizer ):
    """Run the model over the whole test set and return (predictions, true labels)."""


    collator =DataCollatorWithPadding (tokenizer =tokenizer )
    loader =DataLoader (dataset ,batch_size =BATCH_SIZE ,shuffle =False ,collate_fn =collator )

    model .eval ()
    all_preds =[]
    all_true =[]

    total_batches =len (loader )
    with torch .no_grad ():
        for i ,batch in enumerate (loader ,start =1 ):

            labels =batch .pop ("labels")
            logits =model (**batch ).logits 
            all_preds .append (logits .argmax (dim =-1 ).numpy ())
            all_true .append (labels .numpy ())

            if i %10 ==0 or i ==total_batches :
                print (f"      batch {i }/{total_batches }")

    return np .concatenate (all_preds ),np .concatenate (all_true )


def save_confusion_matrix (y_true ,y_pred ,label_names ,title ,out_path ):
    """Draw a confusion matrix heatmap (raw counts) and save it as a PNG."""
    cm =confusion_matrix (y_true ,y_pred ,labels =list (range (len (label_names ))))

    plt .figure (figsize =(7 ,6 ))
    sns .heatmap (
    cm ,annot =True ,fmt ='d',cmap ='Blues',
    xticklabels =label_names ,yticklabels =label_names ,cbar =False ,
    )
    plt .title (title )
    plt .xlabel ("Predicted label")
    plt .ylabel ("True label")
    plt .tight_layout ()
    plt .savefig (out_path ,dpi =150 )
    plt .close ()

    return cm 


def evaluate_task (task_name ,model_dir ,test_files ,task ,label_names ,results_dir ,plot_slug ):
    """Load one trained model, score it on its test set, save the plot, return a report string."""
    print (f"\n  Loading model + tokenizer from {os .path .relpath (model_dir ,BASE_DIR )}")
    tokenizer =AutoTokenizer .from_pretrained (model_dir )
    model =AutoModelForSequenceClassification .from_pretrained (model_dir )

    print ("  Loading test set...")
    test_dataset =UrduTextDataset (test_files ,tokenizer ,max_length =MAX_LENGTH ,task =task )
    print (f"  Test samples: {len (test_dataset )}")

    print ("  Running predictions...")
    y_pred ,y_true =run_predictions (model ,test_dataset ,tokenizer )


    accuracy =accuracy_score (y_true ,y_pred )
    macro_f1 =f1_score (y_true ,y_pred ,average ='macro',zero_division =0 )
    weighted_f1 =f1_score (y_true ,y_pred ,average ='weighted',zero_division =0 )


    report =classification_report (
    y_true ,y_pred ,
    labels =list (range (len (label_names ))),
    target_names =label_names ,
    digits =4 ,
    zero_division =0 ,
    )

    plot_path =os .path .join (results_dir ,f"{plot_slug }_confusion_matrix.png")
    cm =save_confusion_matrix (
    y_true ,y_pred ,label_names ,
    f"{task_name } - Confusion Matrix",plot_path ,
    )
    print (f"  Confusion matrix saved to {os .path .relpath (plot_path ,BASE_DIR )}")

    print (f"  Accuracy: {accuracy :.4f} | Macro F1: {macro_f1 :.4f} | Weighted F1: {weighted_f1 :.4f}")


    lines =[]
    lines .append ("="*70 )
    lines .append (f"  {task_name .upper ()}")
    lines .append ("="*70 )
    lines .append (f"Model directory : {os .path .relpath (model_dir ,BASE_DIR )}")
    lines .append ("Test files      : "+", ".join (os .path .basename (f )for f in test_files ))
    lines .append (f"Test samples    : {len (test_dataset )}")
    lines .append (f"Label order     : {label_names }")
    lines .append ("")
    lines .append (f"Accuracy        : {accuracy :.4f}")
    lines .append (f"Macro F1        : {macro_f1 :.4f}")
    lines .append (f"Weighted F1     : {weighted_f1 :.4f}")
    lines .append ("")
    lines .append ("Classification report:")
    lines .append (report )
    lines .append ("Confusion matrix (rows = true, columns = predicted):")
    header =" "*12 +"".join (f"{name :>12}"for name in label_names )
    lines .append (header )
    for name ,row in zip (label_names ,cm ):
        lines .append (f"{name :>12}"+"".join (f"{int (v ):>12}"for v in row ))
    lines .append ("")

    return "\n".join (lines )


def main ():
    print ("="*60 )
    print ("  Phase 5: Evaluating Sentiment & Emotion Models")
    print ("="*60 )

    torch .set_num_threads (os .cpu_count ()or 1 )

    data_dir =os .path .join (BASE_DIR ,'data')
    results_dir =os .path .join (BASE_DIR ,'results')
    os .makedirs (results_dir ,exist_ok =True )



    tasks =[
    {
    'task_name':"Sentiment Model - Roman Urdu test set",
    'model_dir':os .path .join (BASE_DIR ,'models','sentiment_model'),
    'test_files':[os .path .join (data_dir ,'roman_urdu_sentiment_test.csv')],
    'task':"sentiment",
    'label_names':ROMAN_SENTIMENT_LABELS ,
    'plot_slug':"sentiment_roman",
    },
    {
    'task_name':"Sentiment Model - Urdu script test set",
    'model_dir':os .path .join (BASE_DIR ,'models','sentiment_model'),
    'test_files':[os .path .join (data_dir ,'urdu_sentiment_corpus_test.csv')],
    'task':"sentiment",
    'label_names':URDU_SENTIMENT_LABELS ,
    'plot_slug':"sentiment_urdu_script",
    },
    {
    'task_name':"Emotion Model",
    'model_dir':os .path .join (BASE_DIR ,'models','emotion_model'),
    'test_files':[os .path .join (data_dir ,'semeval_emotion_test.csv')],
    'task':"emotion",
    'label_names':EMOTION_LABELS ,
    'plot_slug':"emotion",
    },
    ]

    all_reports =[]
    total_steps =len (tasks )+2 

    for step ,cfg in enumerate (tasks ,start =1 ):
        print (f"\n[{step }/{total_steps }] Evaluating: {cfg ['task_name']}")
        all_reports .append (evaluate_task (results_dir =results_dir ,**cfg ))

    print (f"\n[{total_steps -1 }/{total_steps }] Writing results.txt...")
    results_path =os .path .join (results_dir ,'results.txt')
    with open (results_path ,'w',encoding ='utf-8')as f :
        f .write ("Phase 5 Evaluation Results\n")
        f .write ("Urdu Sentiment & Emotion Analysis Engine (XLM-RoBERTa)\n\n")
        f .write ("NOTE ON SENTIMENT LABELS\n")
        f .write ("The two sentiment sources encode 0/1/2 differently (Positive and Negative\n")
        f .write ("are swapped between them), so they are scored separately, each under its own\n")
        f .write ("label names. See the comment block in evaluation/evaluate_models.py.\n\n")
        f .write ("\n".join (all_reports ))
    print (f"  Saved to {os .path .relpath (results_path ,BASE_DIR )}")

    print (f"\n[{total_steps }/{total_steps }] Done. Everything is in the results/ folder:")
    for name in sorted (os .listdir (results_dir )):
        print (f"  - results/{name }")


if __name__ =="__main__":
    main ()
