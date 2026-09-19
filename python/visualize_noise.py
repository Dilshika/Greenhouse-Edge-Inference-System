import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_noise():
    clean_path = './data/train_clean.csv'
    noisy_path = './data/train_noisy.csv' 

    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        print("Error: Could not find the respective files")
        return

    print('Loading Data set')
    clean_df = pd.read_csv(clean_path)
    noisy_df = pd.read_csv(noisy_path)

    #Setup visualization parameters (Change the cloum_to_plot as the X axis params you need)
    column_to_plot = 'Temperature'
    window_size = 500

    clean_subset = clean_df[column_to_plot].iloc[:window_size]
    noisy_subset = noisy_df[column_to_plot].iloc[:window_size]

    #create the plot
    plt.figure(figsize=(14,6))

    #Plot Noisy data first (color: red, slightly transparent)
    plt.plot(noisy_subset, label='Noisy Sensor Data (Gaussian + Drift + Spikes)', color='red', alpha=0.7, linewidth=1.5)

    #Plot Clean data
    plt.plot(clean_subset, label='Original Ground Truth', color='green', linewidth=2.5)

    #Formatting
    plt.title(f"Clean vs. Noisy Data; {column_to_plot} (First {window_size} Readings)")
    plt.xlabel("Time Step (Index)", fontsize=12)
    plt.ylabel(f"{column_to_plot} value", fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    #Save and show
    output_image ='./data/temp_noise_visualization.png' #rename as the x axis
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Saved visualization to {output_image}")

    #Popup
    plt.show()


if __name__ == "__main__":
    plot_noise()