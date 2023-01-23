import pandas as pd
import matplotlib as plt
import numpy as np 
import scipy as sp
import streamlit as st

def confirmation(button,action):
    button = st.button("Confirm")
    action = button.append("Confirmed")
    
    if button == False:
        st.write("Succusfully Submmited")
        
    else:
        st.write("Unsuccusfull Submition")
        
        
def checkTheImage(image,Path):
     image = pd.difference(X,Y)
     
     if image == True:
         image.append()
         Path = "./SuggestionML/Train/Model.py"
                 
    