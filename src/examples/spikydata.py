import csv

import matplotlib.pyplot as plt
import numpy as np


def generate_spiky_data(
        length: int = 1000,
        base_value: float = 250.0,
        max_spike_magnitude: float = 150.0,
        min_plateau_length: int = 5,
        max_plateau_length: int = 35) -> list[float]:
    data = []
    current_value = base_value
    i = 0

    while i < length:
        plateau_length = np.random.randint(
            min_plateau_length, max_plateau_length)
        spike = (np.random.rand() - 0.5) * 2 * max_spike_magnitude
        for _ in range(plateau_length):
            noisy_value = current_value + (np.random.rand() - 0.5) * 5
            data.append(noisy_value)
            i += 1
            if i >= length:
                break

        # Apply the spike after each plateau
        current_value += spike

    return data


length = 7500
data = generate_spiky_data(length=length)
csv_filename = "spiky_data.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    for index, value in enumerate(data):
        writer.writerow([index, value])

print(f"Data has been saved to {csv_filename}")
plt.figure(figsize=(10, 5))
plt.plot(data, color="blue", label="Spiky Data with Random Plateaus")
plt.title("Generated Spiky Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
