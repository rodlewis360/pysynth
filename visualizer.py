import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, Entry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to plot the waveform
def plot_wave(wave_data, sample_rate=44100, zoom_start=0, zoom_end=1):
    """
    Plot the waveform with zoom functionality.
    
    :param wave_data: The waveform data as a numpy ndarray.
    :param sample_rate: The sample rate (default: 44100 Hz).
    :param zoom_start: The start time of the zoom window in seconds.
    :param zoom_end: The end time of the zoom window in seconds.
    """
    # Close any previous figures to avoid overlap
    plt.close('all')
    
    # Time array corresponding to the wave_data
    time = np.linspace(0, len(wave_data) / sample_rate, len(wave_data), endpoint=False)
    
    # Apply zooming by selecting the portion of the wave_data based on time
    start_idx = int(zoom_start * sample_rate)
    end_idx = int(zoom_end * sample_rate)
    zoomed_wave = wave_data[start_idx:end_idx]
    zoomed_time = time[start_idx:end_idx]
    
    # Plot the zoomed section of the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(zoomed_time, zoomed_wave, color='b', label='Waveform')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform Zoom')
    plt.grid(True)
    plt.legend()
    plt.show()

# GUI Application for Visualizing the Waveform
class WaveformVisualizerApp:
    def __init__(self, root, wave_data, wave_name="Waveform", sample_rate=44100):
        self.root = root
        self.root.title(wave_name)  # Set the window title
        
        # Waveform data, sample rate, and wave name
        self.wave_data = wave_data
        self.sample_rate = sample_rate
        self.wave_name = wave_name
        
        # Initial zoom window (display the entire waveform initially)
        self.zoom_start = 0
        self.zoom_end = len(wave_data) / sample_rate
        
        # Create the controls
        self.create_widgets()
    
    def create_widgets(self):
        # Label and input fields for zooming
        Label(self.root, text="Zoom Start (s):").grid(row=0, column=0, padx=10, pady=5)
        self.start_entry = Entry(self.root)
        self.start_entry.grid(row=0, column=1, padx=10, pady=5)
        self.start_entry.insert(0, str(self.zoom_start))
        
        Label(self.root, text="Zoom End (s):").grid(row=1, column=0, padx=10, pady=5)
        self.end_entry = Entry(self.root)
        self.end_entry.grid(row=1, column=1, padx=10, pady=5)
        self.end_entry.insert(0, str(self.zoom_end))
        
        # Buttons for zooming in/out
        self.zoom_in_button = Button(self.root, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.grid(row=2, column=0, padx=10, pady=10)
        
        self.zoom_out_button = Button(self.root, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.grid(row=2, column=1, padx=10, pady=10)
        
        self.plot_button = Button(self.root, text="Plot Wave", command=self.update_plot)
        self.plot_button.grid(row=3, column=0, columnspan=2, pady=10)

    def update_plot(self):
        try:
            # Get the zoom start and end times from the entries
            self.zoom_start = float(self.start_entry.get())
            self.zoom_end = float(self.end_entry.get())
            # Plot the waveform with the updated zoom window
            plot_wave(self.wave_data, self.sample_rate, self.zoom_start, self.zoom_end)
        except ValueError:
            print("Invalid input for zoom start or end.")

    def zoom_in(self):
        # Reduce the zoom window by 50%
        self.zoom_end = self.zoom_start + (self.zoom_end - self.zoom_start) / 2
        self.update_plot()

    def zoom_out(self):
        # Increase the zoom window by 50%
        self.zoom_end = self.zoom_start + (self.zoom_end - self.zoom_start) * 2
        self.update_plot()

def visualize_wave_with_app(wave_data, wave_name, sample_rate):
    root = Tk()
    app = WaveformVisualizerApp(root, wave_data, wave_name, sample_rate)
    root.mainloop()
