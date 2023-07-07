import matplotlib.pyplot as plt

# Data
frames = range(0, 6)
mva = [6.14, 18.68, 26.34, 30.75, 32.93, 34.82]

# Plotting
plt.plot(frames, mva, marker='o', linestyle='-', color='b')

# Labels and Title
plt.xlabel('Frame')
plt.ylabel('MVA')
plt.title('MVA over Time')

# Display the plot
plt.show()
