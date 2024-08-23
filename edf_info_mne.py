import mne
import re
# load EDF
raw = mne.io.read_raw_edf('E:\\EEG data\\edf\\132\\aaaaatpq\\s002_2015\\01_tcp_ar\\aaaaatpq_s002_t000.edf', preload=True)
path_to_edf_files = 'E:\\EEG data\\edf\\132\\aaaaatpq\\s002_2015\\01_tcp_ar\\aaaaatpq_s002_t000.edf'

print(raw.info)
session_data = str(raw.info['meas_date'])[0:10]
session_year = str(raw.info['meas_date'])[0:4]
print(str(raw.info['meas_date'])[0:4])

match = re.search(r's\d{3}_(\d{4})', path_to_edf_files)
year = match.group(1)
print(year)

if year==session_year:
 print('yes')
else:
 print('no')


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