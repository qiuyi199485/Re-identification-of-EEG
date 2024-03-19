import mne

# load EDF
raw = mne.io.read_raw_edf('C://Users/49152/Desktop/MA/Code/000/aaaaaaab/s001_2002/02_tcp_le/aaaaaaab_s001_t000.edf', preload=True)


print(raw.info)

# duration
print(raw.times[-1])

if raw.times[-1] is not None:
    print(raw.times[-1])
else:
    print('No time')
#print(str(raw['time']))

