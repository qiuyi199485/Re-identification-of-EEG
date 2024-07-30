import os
from Preprocessing_MA import load_excel_file, process_edf_file, extract_and_normalize_data, segment_data, apply_autoreject

# Define file paths
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
challenges_subset_path = os.path.join(desktop_path, "challenges_subset.xlsx")

# Load Excel file
df = load_excel_file(challenges_subset_path)

# Channel sets
channels_standard = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'A1', 'T3', 'C3',
                     'Cz', 'C4', 'T4', 'A2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']


# Process each EDF file
for i in range(min(3, len(df))):  # Ensure we only process up to 3 files
    selected_row = df.iloc[i]
    edf_path = selected_row['path_to_edf']
    
    raw = process_edf_file(edf_path)
    normalized_data, times = extract_and_normalize_data(raw)
    
    epochs_array = segment_data(normalized_data)
    
    epochs_clean = apply_autoreject(epochs_array, channels_standard, raw.get_montage())
    
    print(f"Processed file {i+1}: {edf_path}")
