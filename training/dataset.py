import torch 
from torch .utils .data import Dataset 
import pandas as pd 


class UrduTextDataset (Dataset ):
    def __init__ (self ,csv_paths ,tokenizer ,max_length =128 ,task ="sentiment"):
        """
        Custom PyTorch Dataset for Urdu Sentiment and Emotion Analysis.

        NOTE: Tokenization here does NOT pad. Padding is applied per-batch by a
        DataCollatorWithPadding in the training script (dynamic padding), which
        is much faster than padding every sample to max_length -- especially on
        CPU, where wasted FLOPs on padding tokens dominate runtime.

        Args:
            csv_paths (list or str): Path(s) to the cleaned CSV files.
            tokenizer: HuggingFace tokenizer (XLM-RoBERTa).
            max_length (int): Max token length for truncation. 
            task (str): 'sentiment' or 'emotion'.
        """
        self .tokenizer =tokenizer 
        self .max_length =max_length 
        self .task =task 

        if isinstance (csv_paths ,str ):
            csv_paths =[csv_paths ]


        dfs =[pd .read_csv (path )for path in csv_paths ]
        self .data =pd .concat (dfs ,ignore_index =True )

        before =len (self .data )

        self .data =self .data .dropna (subset =['text','label'])


        if task =="sentiment":

            label_map ={
            'P':2 ,'O':1 ,'N':0 ,
            0 :2 ,1 :0 ,2 :1 ,
            '0':2 ,'1':0 ,'2':1 ,
            }
        elif task =="emotion":

            label_map ={'joy':0 ,'anger':1 ,'fear':2 ,'sadness':3 }
        else :
            raise ValueError ("Task must be either 'sentiment' or 'emotion'")

        self .data ['label']=self .data ['label'].map (label_map )


        unmapped =int (self .data ['label'].isna ().sum ())
        self .data =self .data .dropna (subset =['label'])
        self .data ['label']=self .data ['label'].astype (int )
        after =len (self .data )
        print (f"  [{task }] kept {after }/{before } rows "
        f"({before -after } dropped, {unmapped } had invalid labels)")
        print (f"  [{task }] class counts: "
        f"{self .data ['label'].value_counts ().sort_index ().to_dict ()}")

        self .texts =self .data ['text'].astype (str ).tolist ()
        self .labels =self .data ['label'].tolist ()

    def __len__ (self ):
        return len (self .texts )

    def __getitem__ (self ,idx ):


        encoding =self .tokenizer (
        self .texts [idx ],
        add_special_tokens =True ,
        max_length =self .max_length ,
        truncation =True ,
        )
        return {
        'input_ids':encoding ['input_ids'],
        'attention_mask':encoding ['attention_mask'],
        'labels':torch .tensor (self .labels [idx ],dtype =torch .long ),
        }

    def get_class_weights (self ,num_labels ):
        """Inverse-frequency class weights for a weighted loss (handles imbalance)."""
        counts =self .data ['label'].value_counts ().sort_index ()
        counts =counts .reindex (range (num_labels ),fill_value =0 )
        total =counts .sum ()

        weights =total /(num_labels *counts .replace (0 ,1 ))
        return torch .tensor (weights .values ,dtype =torch .float )
