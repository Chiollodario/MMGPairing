import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import csv

# Smartphone coordinate lists
x_phone = list()
y_phone = list()
# Smartwatch coordinate lists
x_watch = list()
y_watch = list()
z_watch = list()
# Timestamp lists
phone_timestamp = list()
watch_timestamp = list()
# Magnitude lists
smartphone_magnitude = list()
watch_magnitude = list()
# For changing the colors in the Movement plots (2D and 3D)
boundary = 20;
# Specify the linewidth in plots (for mathplotlib options)
lw = 1
# integers containing the mean of each list values (for removing the gravity)
mean_x_watch = 0
mean_y_watch = 0
mean_z_watch = 0
# 1st derivative of positions
velocity_x_watch = list()
velocity_y_watch = list()
velocity_z_watch = list()
velocity_x_phone = list()
velocity_y_phone = list()
# 2nd derivative of positions
acceleration_x_watch = list()
acceleration_y_watch = list()
acceleration_z_watch = list()
acceleration_x_phone = list()
acceleration_y_phone = list()

# Figures
fig = plt.figure(figsize=plt.figaspect(4.))
fig.suptitle("Movement and Magnitude analysis")

# CSV file parsing
with open("C:\\Users\\DARIO-DELL\\Desktop\\Collected_Data\\2018-06-19_3_smartphone_sample.csv", "r") as phone_file:
    content = csv.reader(phone_file)
    next(content)
    for row in content:
        phone_timestamp.append(float(row[0]))
        x_phone.append(float(row[1]))
        y_phone.append(-float(row[2])) # -float(...) used for showing the drawing as users normally see the screen
        smartphone_magnitude.append(float(row[7]))

with open("C:\\Users\\DARIO-DELL\\Desktop\\Collected_Data\\2018-06-19_3_watch_sample.csv", "r") as watch_file:
    content = csv.reader(watch_file)
    next(content)
    for row in content:
        watch_timestamp.append(float(row[0]))
        x_watch.append(float(row[1]))
        y_watch.append(float(row[2]))
        z_watch.append(float(row[3]))
        watch_magnitude.append(float(row[4]))

# Mean of positions in order to get rid of the gravity
mean_x_watch = float(sum(x_watch))/float(len(x_watch))
mean_y_watch = float(sum(y_watch))/float(len(y_watch))
mean_z_watch = float(sum(z_watch))/float(len(z_watch))

# Gravity removal from smartwatch data
for value in x_watch:
    value -= mean_x_watch
for value in y_watch:
    value -= mean_y_watch
for value in z_watch:
    value -= mean_z_watch

def calculateDerivativeList( timestampList , toCalcList , resultList ):
    resultList.append(0)
    for i in range(len(toCalcList)-1):
        resultList.append((toCalcList[i+1] - toCalcList[i])/(timestampList[i+1] - timestampList[i]))

calculateDerivativeList( watch_timestamp, x_watch, velocity_x_watch )
calculateDerivativeList( watch_timestamp, velocity_x_watch, acceleration_x_watch )
calculateDerivativeList( watch_timestamp, y_watch, velocity_y_watch )
calculateDerivativeList( watch_timestamp, velocity_y_watch, acceleration_y_watch )
calculateDerivativeList( watch_timestamp, z_watch, velocity_z_watch )
calculateDerivativeList( watch_timestamp, velocity_z_watch, acceleration_z_watch )

calculateDerivativeList( phone_timestamp, x_phone, velocity_x_phone )
calculateDerivativeList( phone_timestamp, velocity_x_phone, acceleration_x_phone )
calculateDerivativeList( phone_timestamp, y_phone, velocity_y_phone )
calculateDerivativeList( phone_timestamp, velocity_y_phone, acceleration_y_phone )

x_watch_data = {"position": x_watch, "velocity": velocity_x_watch, "acceleration": acceleration_x_watch}
y_watch_data = {"position": y_watch, "velocity": velocity_y_watch, "acceleration": acceleration_y_watch}
z_watch_data = {"position": z_watch, "velocity": velocity_z_watch, "acceleration": acceleration_z_watch}
x_phone_data = {"position": x_phone, "velocity": velocity_x_phone, "acceleration": acceleration_x_phone}
y_phone_data = {"position": y_phone, "velocity": velocity_y_phone, "acceleration": acceleration_y_phone}

print("==========  acceleration_x_watch   ==============")
print(acceleration_x_watch)
print("==========  acceleration_y_watch   ==============")
print(acceleration_y_watch)
print("==========  acceleration_z_watch   ==============")
print(acceleration_z_watch)
print("==========  acceleration_x_phone   ==============")
print(acceleration_x_phone)
print("==========  acceleration_y_phone   ==============")
print(acceleration_y_phone)
print("==========  velocity_x_phone   ==============")
print(velocity_x_phone)
print("==========  velocity_y_phone   ==============")
print(velocity_y_phone)


ax = fig.add_subplot(2, 2, 1)
# Plot creation
ax.set_title("Axis accelerations")
ax.set_xlabel("Time")
ax.set_ylabel("Accelerations")
ax.plot(np.asarray(x_watch_data.get("acceleration")), label="x_watch", linewidth=1)
ax.plot(np.asarray(y_watch_data.get("acceleration")), label="y_watch", linewidth=1)
#ax.plot(np.asarray(z_watch_data.get("acceleration")), label="z_watch", linewidth=1)
ax.plot(np.asarray(x_phone_data.get("acceleration")), label="x_phone", linewidth=1)
ax.plot(np.asarray(y_phone_data.get("acceleration")), label="y_phone", linewidth=1)
plt.legend()

ax = fig.add_subplot(2, 2, 2)
# Plot creation
ax.set_title("Devices magnitudes")
ax.set_xlabel("Time")
ax.set_ylabel("Magnitude")
ax.plot(np.asarray(smartphone_magnitude), label="smartphone_magnitude", linewidth=1)
ax.plot(np.asarray(watch_magnitude), label="watch_magnitude", linewidth=1)
plt.legend()

ax = fig.add_subplot(2, 2, 3)
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

"""# First subplot (3D)
ax = fig.add_subplot(2, 2, 1, projection="3d")
# Plot creation
ax.set_title("Watch movement")
ax.set_xlabel("X samples")
ax.set_ylabel("Y samples")
ax.set_zlabel("Z samples")

ax.plot(np.asarray(x_watch[:boundary]),
        np.asarray(y_watch[:boundary]),
        np.asarray(z_watch[:boundary]), 
        linewidth=lw, color="green",
        label="beginning")
ax.plot(np.asarray(x_watch[boundary-1:-boundary]),
        np.asarray(y_watch[boundary-1:-boundary]),
        np.asarray(z_watch[boundary-1:-boundary]), 
        linewidth=lw, color="blue",
        label="middle")
ax.plot(np.asarray(x_watch[-1-boundary:]),
        np.asarray(y_watch[-1-boundary:]),
        np.asarray(z_watch[-1-boundary:]), 
        linewidth=lw, color="red",
        label="end")
plt.legend()"""
"""
# First subplot (2D - only x,y samples)
ax = fig.add_subplot(2, 2, 1)
# Plot creation
ax.set_title("Watch movement")
ax.set_xlabel("X samples")
ax.set_ylabel("Y samples")

ax.plot(np.asarray(x_watch[:boundary]),
        np.asarray(y_watch[:boundary]), 
        linewidth=lw, color="green",
        label="beginning")
ax.plot(np.asarray(x_watch[boundary-1:-boundary]),
        np.asarray(y_watch[boundary-1:-boundary]), 
        linewidth=lw, color="blue",
        label="middle")
ax.plot(np.asarray(x_watch[-1-boundary:]),
        np.asarray(y_watch[-1-boundary:]), 
        linewidth=lw, color="red",
        label="end")
plt.legend()


# Second subplot
ax = fig.add_subplot(2, 2, 2)
# Plot creation
ax.set_title("Phone movement")
ax.set_xlabel("X samples")
ax.set_ylabel("Y samples")
ax.plot(np.asarray(x_phone[:boundary]),
        np.asarray(y_phone[:boundary]), 
        linewidth=lw, color="green",
        label="beginning")
ax.plot(np.asarray(x_phone[boundary-1:-boundary]),
        np.asarray(y_phone[boundary-1:-boundary]), 
        linewidth=lw, color="blue",
        label="middle")
ax.plot(np.asarray(x_phone[-1-boundary:]),
        np.asarray(y_phone[-1-boundary:]), 
        linewidth=lw, color="red",
        label="end")
plt.legend()

# Third subplot
ax = fig.add_subplot(2, 2, 3)
# Plot creation
ax.set_title("Watch magnitude")
ax.set_xlabel("Time")
ax.set_ylabel("Magnitude values")
ax.plot(np.asarray(watch_magnitude), linewidth=1)

# Fourth subplot
ax = fig.add_subplot(2, 2, 4)
# Plot creation
ax.set_title("Phone magnitude")
ax.set_xlabel("Time")
ax.set_ylabel("Magnitude values")
ax.plot(np.asarray(smartphone_magnitude), linewidth=1)

plt.show()"""