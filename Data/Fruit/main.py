import csv
import random
import matplotlib.pyplot as plt


def generate_data(num_samples=1000):
    with open('fruit_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["spike_length", "size", "is_poisonous"])

        # Generate the data
        for _ in range(num_samples):
            spike_length = random.uniform(0, 10)  # Spike length in cm
            fruit_size = random.uniform(0, 1000)  # Fruit size in grams

            is_poisonous = int((spike_length/10)**3 + (fruit_size/1000)**2 > 0.2)

            writer.writerow([spike_length, fruit_size, is_poisonous])


def visualize_data():
    with open('fruit_dataset.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)

        # Read the data
        spike_lengths = []
        sizes = []
        colors = []
        for row in reader:
            spike_length, size, is_poisonous = row
            spike_lengths.append(float(spike_length))
            sizes.append(float(size))
            colors.append('red' if int(is_poisonous) else 'blue')

    # Create the scatter plot
    plt.scatter(spike_lengths, sizes, c=colors)
    plt.xlabel('Spike Length')
    plt.ylabel('Size')
    plt.show()

if __name__ == '__main__':
    generate_data(500)
    visualize_data()