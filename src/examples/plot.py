import csv
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

from downsample import ltd, ltob, lttb

colors = ["blue", "red", "green", "purple", "orange",
          "brown", "pink", "gray", "olive", "cyan"]


def plot_data(data: dict) -> None:
    plt.figure(figsize=(10, 6))

    for i, (key, value) in enumerate(data.items()):
        x = value[:, 0]
        y = value[:, 1]
        print(f"Key: {key} with Shape {len(x), len(y)}")
        color = colors[i % len(colors)]
        plt.plot(x, y,
                 label=key,
                 linestyle="-",
                 color=color)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Original Spiky Data - Downsampled Data")
    plt.legend()
    plt.show()


def main():
    path = pathlib.Path(__file__).parent.joinpath("spiky_data.csv")
    data = np.genfromtxt(path, delimiter=",", skip_header=0)
    threshold = 500
    start_time = time.perf_counter()
    ltd_sampled = ltd(data[:, 0], data[:, 1], threshold)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Before execution: Start time = {start_time}")
    print(f"After execution: End time = {end_time}")
    print(f"Execution time: {execution_time:.10f} seconds")
    csv_filename = "sampled_data.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for index, value in zip(ltd_sampled[0], ltd_sampled[1]):
            writer.writerow([index, value])
    merged_ltd_array = np.column_stack((ltd_sampled[0], ltd_sampled[1]))

    start_time = time.perf_counter()
    ltob_sampled = ltob(data[:, 0], data[:, 1], threshold)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Before execution: Start time = {start_time}")
    print(f"After execution: End time = {end_time}")
    print(f"Execution time: {execution_time:.10f} seconds")
    merged_ltob_array = np.column_stack((ltob_sampled[0], ltob_sampled[1]))

    start_time = time.perf_counter()
    lttb_sampled = lttb(data[:, 0], data[:, 1], threshold)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Before execution: Start time = {start_time}")
    print(f"After execution: End time = {end_time}")
    print(f"Execution time: {execution_time:.10f} seconds")
    merged_lttb_array = np.column_stack((lttb_sampled[0], lttb_sampled[1]))

    plot_data(
        {"Original": data,
         "Sampled LTD": merged_ltd_array,
         "Sampled LTOB": merged_ltob_array,
         "Sampled LTTB": merged_lttb_array,
         }
    )


if __name__ == "__main__":
    main()
