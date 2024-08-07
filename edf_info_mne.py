import mne

# load EDF
raw = mne.io.read_raw_edf('C://Users/49152/Desktop/MA/Code/000/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf', preload=True)
path_to_edf_files = 'C:\\Users\\49152\\Desktop\\MA\\Code\\000\\aaaaaaab\\s001_2002\\02_tcp_le\\aaaaaaab_s001_t000.edf'

print(raw.info)

# duration
print(raw.times[-1])

if raw.times[-1] is not None:
    print(raw.times[-1])
else:
    print('No time')
#print(str(raw['time']))

# Open edf as plain text to get attributes from header
with open(path_to_edf_files, 'rb') as h:
 header = h.read(123).decode('utf-8')
# Modify header to get attributes
modified_header = header.split(' ')
sex_str = modified_header[8]
if sex_str in ['F', 'f']:
 sex ='f'
elif sex_str in ['M', 'm']:
 sex = 'm'
else:
 sex = 'x'
            
age_str = modified_header[11][4:]
if age_str.isdigit():
 age = int(age_str)
else: # missing/wrong value
 age = -1