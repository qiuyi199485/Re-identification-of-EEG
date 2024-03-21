import sys
import os
import numpy as np
os.environ['MNE_USE_NUMBA'] = 'false'
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap, )
import tensorflow as tf
from tensorflow.keras import backend as K

sys.path.insert(1, 'C:\\Users\\49152\\Desktop\\MA\\Code')  
import settings as st
import Build_Database as bd

import autoreject

model_path = 'C:\\Users\\49152\\Desktop\\MA\\Code\\pretrained_net_ica1-40Hz\\'
try:
    model = tf.keras.models.load_model(
        #model_path,
        'C:\\Users\\49152\\Desktop\\MA\\Code\\pretrained_net_ica1-40Hz',
        custom_objects={'K': K},
        compile=False)
    print("loaded successfully")
    model.summary()
    
except Exception as e:
    print(f"Error: {e}")
    

try:
    model = tf.keras.models.load_model(model_path)
    model.summary()
    print("loaded successfully")
except Exception as e:
    print(f"Error: {e}")

try:
    input_shape = model.input_shape  
    input_data = np.random.random(size=input_shape[1:])  
    input_data = np.expand_dims(input_data, axis=0)  
    prediction = model.predict(input_data)
    print("loaded successfully")
except Exception as e:
    print(f"Error: {e}")
    
print(f"TF version = ", tf.__version__)

