#!/usr/bin/env python
# coding: utf-8

# # This notebook is for pre- processing Text Data from a folder and and store the output in a specified directory

# ## Objectives
# This notebook aims to give the user the ability to read the code from a folder of .txt files, perform the steps for pre processing on selected .txt file, store the output in a separate folder. 

# IMPORTING PACKAGES AS REQUIRED

# In[1]:


#importing libraries

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime


# PATH OF A FOLDER WHERE UN-PROCESED  TEXT FILES ARE KEPT

# In[3]:


#importing data from 'input' folder

path=r"C:\\Users\\im_ak\\Desktop\\input\\"
files=os.listdir(path)


# Below code functionality as follows
# 
# 1) Take multiple unprocessed text from a specific folder
# 
# 2) Removing white spaces between them
# 
# 3) Removing the words and punctuation as per requirement.
# 
# 
# 4) Storing the process data in a specific output folder with name and timestamp
# 
# 

# In[4]:


#looping the file
for i in [x for x in files]:
    f=open(path+i,'r',encoding='latin-1')

    #files=[x for x in files if '.txt' in x]

    #removing white spaces  between paragraph
    text=' '.join([line.strip() for line in f if line.strip()!=''])
    
    

    
    #renmoving the special character with that word from a text
    special_character_list = ['â','\x80','sss','SSS','g88','88','59g','gg','8g', 'g89','8999','R.A.F.','G88','GG','59G','GG','8G', 'G89']
    text=' '.join([ele for ele in text.split() if all(sp_character not in ele for sp_character in special_character_list)])
    
    #removing only that punctuation which are necessary only other punctuaation makes more readable
    punc=['-',':','| |','@','#','/','\\','\'','//','','â','|','<','>',';','?','!','(',')']
    for ele in text:
        if ele in punc:
            text = text.replace(ele, "")


    
    
    k=[]
    for word in text.split():
        if '.'in word and len(word)>5:
            #print(word)
            k.append( word.replace('.', '.\n\n'))
        else:
            k.append(word)
    text=' '.join(k)
    #storing the process_data each file in output folder
    time1=datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_name=i.replace('.txt','')+"_processed"+"_"+str(time1)+'.txt'
    
    output_path=r"C:\Users\im_AK\Desktop\processdata"+"\\"+str(output_name)
    file = open(output_path, 'w',encoding='latin-1')
    
    
    file.write(text)   

    file.close()

