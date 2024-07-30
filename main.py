import sys                                                                       
import os
os.environ['MNE_USE_NUMBA'] = 'false'                                            #避免使用numba speedup
import mne
import numpy as np
from IPython.display import clear_output
import pandas as pd
#Import of self created python script
sys.path.insert( 1,'C:\\Users\\49152\\Documents\\GitHub\\Re-identification-of-EEG')                              # 允许脚本导入一个特定路径下的自定义Python脚本，例如settings模块和tools模块里的函数。
#sys.path.insert(1, 'C:\\Users\\49152\\Desktop\\MA\\Code')       
import settings
#from tools import test_edf_corrupted_info, get_date_edf                           ## (edf_corrupted, edf_info) false没坏，edf_metadata; edf measurment date yyyy-mm-dd
from tools import get_date_edf 
from Preprocessing_MA import load_excel_file, process_edf_file, extract_and_normalize_data, segment_data, apply_autoreject


# Define file paths
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")

# Load Excel file
df = load_excel_file(challenges_subset_path)

# Channel sets
channels_standard = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'A1', 'T3', 'C3',
                     'Cz', 'C4', 'T4', 'A2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
# Create and set montage
#montage = mne.channels.make_standard_montage('standard_1020')
#print(f"Montage channels for file {i+1}: {montage.ch_names}")

# Process each EDF file

for i in range(min(3, len(df))):  # Ensure we only process up to 3 files
    selected_row = df.iloc[i]
    edf_path = selected_row['path_to_edf']
    
    raw = process_edf_file(edf_path)
    normalized_data, times = extract_and_normalize_data(raw)
    
    epochs_array = segment_data(normalized_data)
    
    epochs_clean = apply_autoreject(epochs_array, channels_standard, raw.get_montage())
    
    print(f"Processed file {i+1}: {edf_path}")