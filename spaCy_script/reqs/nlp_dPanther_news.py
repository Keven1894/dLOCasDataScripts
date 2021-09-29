#!/usr/bin/env python
# coding: utf-8

# # dPanther News articles demo test with Spacy Library
# 
# >This notebook is designed to test some NLP functions from Spacy with dPanther's news and article data
# 
# 1. FIU dPanther repository: http://dpanther.fiu.edu
# 2. spaCy python library for NLP in Python: https://spacy.io/ 
# 
# - Author: Boyuan (Keven) Guan
# - Copyright: Florida International University Library
# - Last update: 09/29/2021
# 

# In[2]:


# import spacy libarary
import spacy
# load the english model
nlp=spacy.load('en_core_web_sm')


# In[22]:


# load data from txt file
with open('CA03400001_00001.txt','r', encoding="utf8") as file:
    data = file.read().replace('\n', '')
    print(data)


# In[23]:


# create a nlp object
doc = nlp(data)


# In[24]:


# check the pipe names
nlp.pipe_names


# In[25]:


# use case no.1 Part-of-Speech (POS) 
# In English grammar, the parts of speech tell us what is the function of a word and how it is used in a sentence.
# In this project, this will help to capture the story by fined the pattern of 'who-did-what'

# let's try a simple sample before jump into the actual dataset
testor = nlp("Boyuan Guan was working all day at my home in west Miami and then go to my office at Florida International University to prepare a meeting. Some of my workmates from University of Florida will also join the meeting.")
for token in testor:
    print(token.text, " [parts]--> ", token.pos_, " [dependency]--> ",  token.dep_, ", position: ", token.i, "-", token.idx)


# In[26]:


print(testor.sentiment)


# In[27]:


testor11 = nlp("I am super happy!!")
print(testor11.sentiment)


# In[28]:


# EXTRACT ONLY THE VERB FROM THE REAL DATA
for token in doc:
    if token.pos_ == 'VERB':
        print(token.text, " [parts]--> ", token.pos_, " [dependency]--> ",  token.dep_, ", position: ", token.i, "-", token.idx)


# In[29]:


from IPython.display import Image
Image("dep_tree.png")


# In[30]:


spacy.explain("nsubj"), spacy.explain("ROOT"), spacy.explain("aux"), spacy.explain("advcl"), spacy.explain("dobj")


# In[31]:


# Now, let's work on some NER 
for ent in testor.ents:
    print(ent.text, ent.label_)


# In[32]:


spacy.explain("GPE"), spacy.explain("CARDINAL"), spacy.explain("WORK_OF_ART"), spacy.explain("LOC"), spacy.explain("EVENT")


# In[33]:


# try real data instead
target_ner = 'PERSON'
print (target_ner, ": ", end='')
for ent in doc.ents:
    if ent.label_ == target_ner:
        print(ent.text, ", ", ent.label, "; ", end='')


# In[34]:


Image("ner_label.png")


# In[15]:


Image("ner_label2.png")


# In[35]:


from spacy.lang.en import English
from spacy.pipeline import EntityRuler

nlp = English()
ruler = EntityRuler(nlp)
patterns = [{"label": "GPE", "pattern": "Mississippi"}]
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)

doc = nlp(data)

print([(ent.text, ent.label_) for ent in doc.ents])


# In[36]:


from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(data)
displacy.serve(doc, style="ent")


# In[38]:


from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Boyuan Guan was working all day at my home in west Miami and then go to my office at Florida International University to prepare a meeting. Some of my workmates from University of Florida will also join the meeting.")
displacy.serve(doc, style="ent")


# In[20]:


testor2 = nlp("Tropical Storm Eta could intensify in the Gulf of Mexico through Wednesday, but is then expected to weaken as it potentially approaches the U.S. Gulf Coast later in the week.Eta is now centered in the southern Gulf of Mexico, where it has now stalled out near the western tip of Cuba.")
for ent in testor2.ents:
    print(ent.text, ent.label_)


# In[14]:


from spacy import displacy

nlp = spacy.load("en_core_web_sm")
# Eta is not recognized as a hurricane 
doc = nlp("After a historical hurricane, Irma, hit south florida, tropical Storm Eta could intensify in the Gulf of Mexico through Wednesday, but is then expected to weaken as it potentially approaches the U.S. Gulf Coast later in the week.Eta is now centered in the southern Gulf of Mexico, where it has now stalled out near the western tip of Cuba.")
displacy.serve(doc, style="ent")


# In[ ]:


# todo: re-train the NER model to recognize the hurricane name


# In[15]:


from spacy import displacy

nlp = spacy.load("en_core_web_sm")
# Eta is not recognized as a hurricane 
doc = nlp("Tropical Storm Eta could intensify in the Gulf of Mexico through Wednesday, but is then expected to weaken as it potentially approaches the U.S. Gulf Coast later in the week.Eta is now centered in the southern Gulf of Mexico, where it has now stalled out near the western tip of Cuba.")
displacy.render(doc, style="ent")


# ## Start Traning the existing model with new Entity: Hurricane

# In[7]:


#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.1.0+
Last tested with: v2.2.4
"""
from __future__ import unicode_literals, print_function

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# In[8]:


# new entity label
LABEL = "HURRICANE"


# In[9]:


# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    (
        "Tropical Storm Eta could intensify in the Gulf of Mexico through Wednesday",
        {"entities": [(14, 18, LABEL)]},
    ),
    ("Do they bite?", {"entities": []}),
    (
        "Eta originated from a vigorous tropical wave in the eastern Caribbean Sea on October 31",
        {"entities": [(0, 3, LABEL)]},
    ),
    ("Eta is a Tropical Strom in 2020", {"entities": [(0, 2, LABEL)]}),
    (
        "The last Tropical Strom, that Eta",
        {"entities": [(29, 32, LABEL)]},
    ),
    ("Eta?", {"entities": [(0, 3, LABEL)]}),
]


# In[16]:


nlp2 = spacy.load('hurricane')


# In[17]:



nlp2 = spacy.load("hurricane")
# Eta is not recognized as a hurricane 
doc = nlp2("Tropical Storm Eta could intensify in the Gulf of Mexico through Wednesday, but is then expected to weaken as it potentially approaches the U.S. Gulf Coast later in the week.Eta is now centered in the southern Gulf of Mexico, where it has now stalled out near the western tip of Cuba.")
displacy.render(doc, style="ent")


# In[8]:


doc = nlp2("Hurricane Eta was a devastating Category 4 hurricane that wreaked havoc across parts of Central America in early November 2020")
displacy.render(doc, style="ent")


# In[13]:


import urllib.request  # the lib that handles the url stuff

for line in urllib.request.urlopen("https://en.wikipedia.org/wiki/Hurricane_Eta"):
    doc = nlp2(line.decode('utf-8'))
    displacy.render(doc, style="ent")


# In[ ]:




