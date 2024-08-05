import numpy as np
import matplotlib.pyplot as plt

# parameter
num_channels = 5  # 
num_points = 100  # 
time = np.linspace(0, 5, num_points)  

# Generation of random EEG signals
eeg_signals = np.random.randn(num_channels, num_points)

# Plot
plt.figure(figsize=(4, 8))
for i in range(num_channels):
    plt.plot(time, eeg_signals[i] + 10 * i, color='black')


plt.axis('off')

plt.show()

