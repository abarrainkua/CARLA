import matplotlib.pyplot as plt
import numpy as np

# The data for your 6 pairs of bars
data = [[8.06, 7.72],
        [8.18, 7.63],
        [8.16, 7.79],
        [2.90, 5.71],
        [2.78, 5.64],
        [5.38, 2.96]]

# The delta values to display on top of each pair
delta_values = [0.34, 0.55, 0.38, -2.81, -2.86, 2.42]

# Separate the values for easier plotting
first_values = [pair[0] for pair in data]
second_values = [pair[1] for pair in data]

# Number of pairs
num_pairs = len(data)

# Set the width of the bars
bar_width = 0.35
# Define the gap between the first 3 and last 3 pairs
group_gap = 0.8 # Additional space between the 3rd and 4th pairs

# Set the positions of the bars on the x-axis
# This creates groups of two bars for each pair, with an added gap
r1 = np.array([0, 1, 2, 2 + group_gap + 1, 2 + group_gap + 2, 2 + group_gap + 3])
r2 = r1 + bar_width

# Define the colors for the bars
color_first_bar = 'skyblue' # A light blue for the first value in each pair
color_second_bar = 'lightcoral' # A light red/pink for the second value in each pair

# Create the plot
plt.figure(figsize=(12, 7)) # Adjust figure size for better readability

# Plot the first set of bars
plt.bar(r1, first_values, color=color_first_bar, width=bar_width, edgecolor='grey', label='S = 1')

# Plot the second set of bars, offset
plt.bar(r2, second_values, color=color_second_bar, width=bar_width, edgecolor='grey', label='S = -1')

# Add labels, title, and legend
plt.xlabel('', fontweight='bold') # X-axis label is removed as per request
plt.ylabel('av. cost', fontsize=20)
plt.title('', fontweight='bold') # Title is removed as per request

# Define the new x-axis labels
new_x_labels = [r'SVM(X,A)', r'SVM(X)', r'SVM$_{causal}$', r'SVM(X,A)', r'SVM(X)', r'SVM$_{causal}$']

# Set the x-axis ticks with the new labels
plt.xticks([pos + bar_width / 2 for pos in r1], new_x_labels, fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid for easier reading
# plt.ylim(0,9)

# Add vertical line to separate the first 3 and last 3 pairs
# The line's x-position is at the midpoint of the gap
line_x_pos = r1[3] - group_gap / 2 # r1[3] is the start of the 4th group
plt.axvline(x=line_x_pos - 0.3, color='gray', linestyle='--', linewidth=1.5)

# Add delta values on top of each pair
text_offset = 0.3 # Adjust this value to move text higher/lower
for i in range(num_pairs):
    # Calculate the center x-position for the text above the pair
    center_x = r1[i] + bar_width / 2
    # Determine the higher bar in the pair to place the text above it
    max_y = max(first_values[i], second_values[i])
    # Format the delta text, including the LaTeX delta symbol
    plt.text(center_x, max_y + text_offset,
             r'$\Delta$ = ' + f'{delta_values[i]:.2f}',
             ha='center', va='bottom', fontsize=20)
    
# --- Add arrow and text across the vertical line ---
# Determine the maximum y-value among all bars to set the arrow/text position
max_overall_y = max(max(first_values), max(second_values))
# Set the y-position for the arrow and text, ensuring it's above all bars and delta values
arrow_y_pos = max_overall_y + 1.0 # Adjust this value as needed for vertical placement
text_arrow_offset = 0.4 # Vertical offset for the text above the arrow

# Draw the horizontal arrow crossing the vertical line
plt.annotate('', # No text with the arrow itself
             xy=(line_x_pos - 0.8 - 0.3, arrow_y_pos), # Start point of the arrow (left of line)
             xycoords='data',
             xytext=(line_x_pos + 0.8 - 0.28, arrow_y_pos), # End point of the arrow (right of line)
             textcoords='data',
             arrowprops=dict(arrowstyle="<-", color='black', lw=1.5, ls='-'))

# Add the text on top of the arrow
plt.text(line_x_pos - 0.3, arrow_y_pos + text_arrow_offset - 0.1,
         r'$\cdot (1 - \mathbb{P} \{ f(x)=1 \})$',
         ha='center', va='bottom', fontsize=22, color='black',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=1))

# Adjust y-axis limits to ensure everything fits, if necessary
plt.ylim(0, max_overall_y + 1.0 + text_arrow_offset + 0.5) # Add extra padding


# 2. Get the current axes
ax = plt.gca()
# Get rid of the top spine
ax.spines['top'].set_visible(False)

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

plt.savefig("eq_cost_limitations.pdf", format="pdf")
# Added plt.show() to explicitly display the plot.
plt.show()