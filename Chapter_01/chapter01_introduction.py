# Extracted from chapter1_Introduction.qmd
# Do not edit the source .qmd file directly.

import matplotlib.pyplot as plt

colors = ["black", "#003DFD", "#b512b8", "#11a9ba", "#0d780f", "#f77f07", "#ba0f0f"]

fig, ax = plt.subplots(figsize=(8, 8))

circles = [
    {'radius': 3.5, 'center': (0, -1), 'label': 'Artificial Intelligence', 'color': '#ff6933'},
    {'radius': 2.8, 'center': (0, -1.7), 'label': 'Machine Learning', 'color': '#ff9c33'},
    {'radius': 2.0, 'center': (0, -2.5), 'label': 'Neural Networks', 'color': '#ffcf33'},
    {'radius': 1.2, 'center': (0, -3.3), 'label': 'Deep Learning', 'color': '#fcff33'}
]

for circle in circles:
    circ = plt.Circle(circle['center'], circle['radius'], color=circle['color'], alpha=1)
    ax.add_patch(circ)

title_positions = [0.5, -1.5, -4, -6.5] 
for idx, circle in enumerate(circles):
    ax.text(0, title_positions[idx] - circle['center'][1], circle['label'], fontsize=14 + idx, ha='center')

ax.set_xlim(-4, 4)
ax.set_ylim(-5, 3)
ax.set_aspect('equal')
ax.axis('off')

plt.show()