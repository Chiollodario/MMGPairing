import numpy as np
import matplotlib.pyplot as plt
import csv

# Smartphone coordinate lists
x_phone = list()
y_phone = list()
# Timestamp lists
phone_timestamp = list()
# For changing the colors in the Movement plots (2D and 3D)
boundary = 20;
# Specify the linewidth in plots (for mathplotlib options)
lw = 1

# Figures
fig = plt.figure(figsize=plt.figaspect(1.))
fig.suptitle("Drawing on smartphone")

# CSV file parsing
with open("C:\\Users\\DARIO-DELL\\Desktop\\Collected_Data\\2018-09-21_7_smartphone_sample.csv", "r") as phone_file:
    content = csv.reader(phone_file)
    next(content)
    for row in content:
        phone_timestamp.append(float(row[0]))
        x_phone.append(float(row[1]))
        y_phone.append(-float(row[2])) # -float(...) used for showing the drawing as users normally see the screen
#        smartphone_magnitude.append(float(row[7]))

x_phone_data = {"position": x_phone}
y_phone_data = {"position": y_phone}

ax = fig.add_subplot(1, 1, 1)
# Plot creation
ax.set_title("Phone movement")
ax.set_xlabel("X samples")
ax.set_ylabel("Y samples")
ax.plot(np.asarray(x_phone_data.get("position")[:boundary]),
        np.asarray(y_phone_data.get("position")[:boundary]), 
        linewidth=lw, color="green",
        label="beginning")
ax.plot(np.asarray(x_phone_data.get("position")[boundary-1:-boundary]),
        np.asarray(y_phone_data.get("position")[boundary-1:-boundary]), 
        linewidth=lw, color="blue",
        label="middle")
ax.plot(np.asarray(x_phone_data.get("position")[-1-boundary:]),
        np.asarray(y_phone_data.get("position")[-1-boundary:]), 
        linewidth=lw, color="red",
        label="end")
plt.legend()

plt.show()