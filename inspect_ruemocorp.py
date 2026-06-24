from datasets import load_dataset 
import pandas as pd 
from sklearn .model_selection import train_test_split 
import os 
import re 


DATA_DIR ='data'
os .makedirs (DATA_DIR ,exist_ok =True )


print ("Downloading RUEmoCorp...")
ds =load_dataset ("Khubaib01/RUEmoCorp","ruemocorp-annotated")
df =ds ['train'].to_pandas ()
print (f"Downloaded {len (df )} rows")


df =df .rename (columns ={
'message':'text',
'emotion_label':'label'
})




label_map ={
'happy':'joy',
'sad':'sadness',
'anger':'anger',
'fear':'fear',
}
df ['label']=df ['label'].map (label_map )


before =len (df )
df =df .dropna (subset =['label'])
print (f"Dropped {before -len (df )} rows (none/surprise/disgust)")


def clean_text (text ):
    if not isinstance (text ,str ):
        return ""
    text =re .sub (r'http\S+','',text )
    text =re .sub (r'@\w+','',text )
    text =re .sub (r'#\w+','',text )
    text =re .sub (r'\s+',' ',text )
    return text .strip ()

df ['text']=df ['text'].apply (clean_text )


df =df .dropna (subset =['text'])
df =df [df ['text'].str .len ()>2 ].reset_index (drop =True )
print (f"Final clean dataset: {len (df )} rows")
print (f"Class distribution:\n{df ['label'].value_counts ()}")


train ,temp =train_test_split (
df ,test_size =0.20 ,random_state =42 ,stratify =df ['label']
)
val ,test =train_test_split (
temp ,test_size =0.50 ,random_state =42 ,stratify =temp ['label']
)


for split_name ,data in [('train',train ),('val',val ),('test',test )]:
    path =os .path .join (DATA_DIR ,f'roman_urdu_emotion_{split_name }.csv')
    data .to_csv (path ,index =False ,encoding ='utf-8-sig')
    print (f"Saved {path }  ({len (data )} rows)")

print ("\nDone! RUEmoCorp cleaned and split.")
print ("Now combine with SemEval and retrain emotion model on Kaggle.")
