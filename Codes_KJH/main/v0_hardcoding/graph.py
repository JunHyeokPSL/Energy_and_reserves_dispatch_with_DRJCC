# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 03:47:57 2023

@author: user
"""

cases = ['Case 1', 'Case 2', 'Case 3', 'Case 4']
active_power_data = {
    'Case 1': case1_PG[0][0],  # Replace with actual values for Case 1
    'Case 2': case2_PG[0][0],  # Replace with actual values for Case 2
    'Case 3': case3_PG[0][0],
    'Case 4': case4_PG[0][0]
}

bar_width = 0.2
index = np.arange(len(case1_PG[0][0]))

# Plotting the grouped bar graph
fig, ax = plt.subplots(figsize=(8, 6))

for i, (case, data) in enumerate(active_power_data.items()):
    ax.bar(index + i * bar_width, data, bar_width, label=case)

ax.set_xlabel('Categories')
ax.set_ylabel('Active Power')
ax.set_title('Active Power for Cases 1, 2, 3')
ax.set_xticks(index + bar_width * (len(active_power_data) - 1) / 2)
ax.set_xticklabels([f"Data {i+1}" for i in range(len(active_power_data['Case 1']))])
ax.legend()

plt.tight_layout()
plt.show()









cases = ['Case 1', 'Case 2', 'Case 3']
active_power_data = {
    'Case 1': case1_PG[0][1],  # Replace with actual values for Case 1
    'Case 2': case2_PG[0][1],  # Replace with actual values for Case 2
    'Case 3': case3_PG[0][1]
}

bar_width = 0.2
index = np.arange(len(case1_PG[0][1]))

# Plotting the grouped bar graph
fig, ax = plt.subplots(figsize=(8, 6))

for i, (case, data) in enumerate(active_power_data.items()):
    ax.bar(index + i * bar_width, data, bar_width, label=case)

ax.set_xlabel('Categories')
ax.set_ylabel('Ramp')
ax.set_title('Ramp Up for Cases 1, 2, 3')
ax.set_xticks(index + bar_width * (len(active_power_data) - 1) / 2)
ax.set_xticklabels([f"Data {i+1}" for i in range(len(active_power_data['Case 1']))])
ax.legend()

plt.tight_layout()
plt.show()
