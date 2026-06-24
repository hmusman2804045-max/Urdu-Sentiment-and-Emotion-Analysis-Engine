
import os 
import re 
import sys 
import requests 
import pandas as pd 
from datasets import load_dataset 
from sklearn .model_selection import train_test_split 

DATA_DIR =os .path .dirname (os .path .abspath (__file__ ))

def separator (title ):
    print ("\n"+"="*60 )
    print (f"  {title }")
    print ("="*60 )

def clean_text (text ):
    if not isinstance (text ,str ):
        return ""
    text =re .sub (r'http\S+','',text )
    text =re .sub (r'@\w+','',text )
    text =re .sub (r'#\w+','',text )
    text =re .sub (r'\s+',' ',text )
    return text .strip ()

def split_and_save (df ,name ,text_col ,label_col ):
    df =df [[text_col ,label_col ]].rename (columns ={text_col :'text',label_col :'label'})
    df =df .dropna (subset =['text','label'])
    df ['text']=df ['text'].apply (clean_text )
    df =df [df ['text'].str .len ()>2 ].reset_index (drop =True )

    train ,temp =train_test_split (df ,test_size =0.20 ,random_state =42 ,stratify =df ['label'])
    val ,test =train_test_split (temp ,test_size =0.50 ,random_state =42 ,stratify =temp ['label'])

    for split ,data in [('train',train ),('val',val ),('test',test )]:
        path =os .path .join (DATA_DIR ,f'{name }_{split }.csv')
        data .to_csv (path ,index =False ,encoding ='utf-8-sig')
        print (f"  Saved {path }  ({len (data )} rows)")

    return train ,val ,test 

separator ("DATASET 1: Roman Urdu Sentiment (HuggingFace)")
try :
    ds1 =load_dataset ('community-datasets/roman_urdu',trust_remote_code =True )
    print ("Available splits:",list (ds1 .keys ()))
    df1 =ds1 ['train'].to_pandas ()
    print (f"Columns      : {df1 .columns .tolist ()}")
    print (f"Shape        : {df1 .shape }")
    print (f"Sample row   :\n{df1 .iloc [0 ]}")
    print (f"\nLabel dist.  :\n{df1 .iloc [:,-1 ].value_counts ()}")

    text_col =df1 .columns [0 ]
    label_col =df1 .columns [-1 ]
    print (f"\nUsing text='{text_col }'  label='{label_col }'")
    split_and_save (df1 ,'roman_urdu_sentiment',text_col ,label_col )
    print ("Dataset 1 DONE")
except Exception as e :
    print (f"ERROR loading Dataset 1: {e }")

separator ("DATASET 2: SemEval 2018 Task 1 - Emotion (HuggingFace)")
try :
    ds2 =load_dataset ('SemEvalWorkshop/sem_eval_2018_task_1','subtask5.english',trust_remote_code =True )
    print ("Available splits:",list (ds2 .keys ()))

    frames =[]
    for split_name in ds2 .keys ():
        tmp =ds2 [split_name ].to_pandas ()
        frames .append (tmp )
    df2_all =pd .concat (frames ,ignore_index =True )

    print (f"Columns : {df2_all .columns .tolist ()}")
    print (f"Shape   : {df2_all .shape }")
    print (f"Sample  :\n{df2_all .iloc [0 ]}")

    emotion_cols =['anger','anticipation','disgust','fear','joy',
    'love','optimism','pessimism','sadness','surprise','trust']
    target_emotions =['joy','anger','fear','sadness']
    available =[c for c in target_emotions if c in df2_all .columns ]

    if available :
        df2_all ['label']=df2_all [available ].idxmax (axis =1 )
        df2_all ['max_score']=df2_all [available ].max (axis =1 )
        df2_all =df2_all [df2_all ['max_score']>0 ]
        text_col2 ='Tweet'if 'Tweet'in df2_all .columns else df2_all .columns [0 ]
        print (f"\nEmotion dist.:\n{df2_all ['label'].value_counts ()}")
        split_and_save (df2_all ,'semeval_emotion',text_col2 ,'label')
        print ("Dataset 2 DONE")
    else :
        print (f"Emotion columns not found. Available: {df2_all .columns .tolist ()}")
except Exception as e :
    print (f"ERROR loading Dataset 2: {e }")
    print ("Trying alternate config...")
    try :
        print ("Available configs:")
        from datasets import get_dataset_config_names 
        configs =get_dataset_config_names ('SemEvalWorkshop/sem_eval_2018_task_1')
        print (configs )
    except Exception as e2 :
        print (f"Could not list configs: {e2 }")

separator ("DATASET 3: mirfan899 Urdu Sentiment TSV (GitHub Download)")
tar_path =os .path .join (DATA_DIR ,'urdu.tsv.tar.gz')
tsv_path =os .path .join (DATA_DIR ,'urdu_v1.tsv')
try :
    if not os .path .exists (tsv_path ):
        url ='https://raw.githubusercontent.com/mirfan899/Urdu/master/sentiment/urdu.tsv.tar.gz'
        print (f"Downloading from: {url }")
        r =requests .get (url ,timeout =30 )
        r .raise_for_status ()
        with open (tar_path ,'wb')as f :
            f .write (r .content )
        print (f"Downloaded successfully ({len (r .content )} bytes)")

        import tarfile 
        with tarfile .open (tar_path ,"r:gz")as tar :
            tar .extractall (path =DATA_DIR )
        print ("Extracted urdu_v1.tsv")

    for sep in ['\t',',',';']:
        try :
            df3 =pd .read_csv (tsv_path ,sep =sep ,header =0 ,encoding ='utf-8')
            if df3 .shape [1 ]>=2 :
                break 
        except :
            continue 

    print (f"Columns : {df3 .columns .tolist ()}")
    print (f"Shape   : {df3 .shape }")
    print (f"\nLabel dist.:\n{df3 .iloc [:,-1 ].value_counts ()}")

    split_and_save (df3 ,'mirfan_urdu_sentiment',df3 .columns [0 ],df3 .columns [-1 ])
    print ("Dataset 3 DONE")
except Exception as e :
    print (f"ERROR loading Dataset 3: {e }")

separator ("DATASET 4: Urdu Sentiment Corpus (GitHub Download)")
tsv4_path =os .path .join (DATA_DIR ,'urdu-sentiment-corpus-v1.tsv')
try :
    if not os .path .exists (tsv4_path ):
        url4 ='https://raw.githubusercontent.com/MuhammadYaseenKhan/Urdu-Sentiment-Corpus/master/urdu-sentiment-corpus-v1.tsv'
        print (f"Downloading from: {url4 }")
        r4 =requests .get (url4 ,timeout =30 )
        r4 .raise_for_status ()
        with open (tsv4_path ,'wb')as f :
            f .write (r4 .content )
        print (f"Downloaded successfully ({len (r4 .content )} bytes)")

    for enc in ['utf-8','utf-8-sig','cp1252','latin-1']:
        try :
            df4 =pd .read_csv (tsv4_path ,sep ='\t',encoding =enc )
            break 
        except :
            continue 

    print (f"Columns  : {df4 .columns .tolist ()}")
    print (f"Shape    : {df4 .shape }")

    text_col4 =df4 .columns [0 ]
    label_col4 =df4 .columns [-1 ]
    print (f"\nLabel dist.:\n{df4 [label_col4 ].value_counts ()}")
    split_and_save (df4 ,'urdu_sentiment_corpus',text_col4 ,label_col4 )
    print ("Dataset 4 DONE")
except Exception as e :
    print (f"ERROR loading Dataset 4: {e }")

separator ("PHASE 2 COMPLETE — Summary of saved files")
all_files =[f for f in os .listdir (DATA_DIR )if f .endswith ('.csv')]
print (f"{'File':<45} {'Rows':>6}")
print ("-"*55 )
for f in sorted (all_files ):
    path =os .path .join (DATA_DIR ,f )
    try :
        n =len (pd .read_csv (path ,encoding ='utf-8-sig'))
        print (f"{f :<45} {n :>6}")
    except :
        print (f"{f :<45}  (error reading)")
print ("\nAll datasets collected, cleaned, and split. Ready for Phase 3.")
