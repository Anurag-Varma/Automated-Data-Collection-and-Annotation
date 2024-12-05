import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'transformed_end_effector_poses.csv'
data = np.loadtxt(file_path, delimiter=',')

def plotReferenceFrame(R, p, scale_value, length_value, ax, color):
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 0], scale_value*R[1, 0], scale_value*R[2, 0], color=color[0], arrow_length_ratio=length_value)
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 1], scale_value*R[1, 1], scale_value*R[2, 1], color=color[1], arrow_length_ratio=length_value)
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 2], scale_value*R[1, 2], scale_value*R[2, 2], color=color[2], arrow_length_ratio=length_value)
    return ax


transformation_matrices = [data[i].reshape(4, 4) for i in range(data.shape[0])]


groups = {}
current_group = []
group_index = 1

# Loop through each row
for i in range(len(transformation_matrices)):
    tx_value = transformation_matrices[i][0][3]  # Extract Tx (4th column)
    tx_squared = tx_value ** 2  # Square Tx

    if i == 0:
        # Start with the first squared value
        current_group.append(i)
    else:
        prev_tx_squared = transformation_matrices[i-1][0][3] ** 2
        
        if tx_squared < prev_tx_squared:
            # If current squared value is less than previous, continue adding to current group
            current_group.append(i)
        else:
            # If there's an increase, finalize current group and start a new one
            groups[group_index] = current_group
            group_index += 1
            current_group = [i]

# Add the last group if not empty
if current_group:
    groups[group_index] = current_group

# Now, plotting these matrices as reference frames (with different colors for each group)
fig10 = plt.figure()
ax10 = fig10.add_subplot(projection='3d')

# Define a list of colors (can be extended as needed)
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# Plot each transformation matrix from the CSV as a sampled end-effector pose with different colors for each group
for group_index, indices in groups.items():
    color_idx = (group_index - 1) % len(colors)  # Cycle through colors if more groups than colors
    color_set = [colors[color_idx]] * 3          # Use the same color for all axes of the frame
    
    for idx in indices:
        pose = transformation_matrices[idx]
        R = pose[0:3, 0:3]  # Extract rotation matrix (top-left 3x3 part)
        p = pose[0:3, 3]    # Extract position vector (first 3 elements of the last column)
        ax10 = plotReferenceFrame(R, p, 0.15, 0.15, ax10, color_set)

# Set labels and axis limits
ax10.set_xlabel('X')
ax10.set_ylabel('Y')
ax10.set_zlabel('Z')
ax10.set_xlim(-0.8, 1)
ax10.set_ylim(-0.8, 1)
ax10.set_zlim(-0.8, 1)

plt.show()


import json

# Prepare data to save
grouped_poses = {}
for group_index, indices in groups.items():
    grouped_poses[group_index] = [transformation_matrices[idx].tolist() for idx in indices[::-1]]

# Define the path for the output JSON file
output_file_path = 'grouped_end_effector_poses.json'

# Write the grouped data to the JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(grouped_poses, json_file, indent=4)

print(f"Grouped end effector poses saved to {output_file_path}")
