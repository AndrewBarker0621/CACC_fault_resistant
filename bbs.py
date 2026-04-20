import matplotlib.pyplot as plt
import matplotlib.patches as patches

# size
car_length = 5.0  # meter
car_width = 2.5

# init
gaps = [19, 4.5, 8, 6.5]  # Car 0~4 dis
speeds = [17.5, 16, 15.5, 12.0]  # Car 1~4 vel

x_positions = [100.0]
for gap in gaps:
    x_positions.append(x_positions[-1] - gap - car_length)

fig, ax = plt.subplots(figsize=(12, 3))
y_center = 0

for i, x in enumerate(x_positions):
    # bbs
    rect = patches.Rectangle((x, y_center - car_width/2), car_length, car_width,
                             edgecolor='black', facecolor='skyblue', linewidth=2)
    ax.add_patch(rect)

    # no.
    speed = speeds[i - 1] if i > 0 else 10.0  # Leader vel = 10
    ax.text(x + car_length/2, y_center + car_width/2 + 0.5,
            f"Car {i}\nv={speed} m/s", ha='center', fontsize=9)

# dis
for i in range(1, len(x_positions)):
    x_front = x_positions[i - 1]
    x_back = x_positions[i] + car_length
    mid = (x_front + x_back) / 2
    ax.annotate("", xy=(x_back, y_center - 2), xytext=(x_front, y_center - 2),
                arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(mid, y_center - 1, f"{gaps[i - 1]} m", ha='center', fontsize=9)

# cs
ax.set_xlim(x_positions[-1] - 5, x_positions[0] + car_length + 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()

# save
plt.savefig("initial_vehicle_layout.pdf", format='pdf')
print("Saved as initial_vehicle_layout.pdf")
