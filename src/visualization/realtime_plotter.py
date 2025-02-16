import matplotlib.pyplot as plt

class RealTimePlotter:
    def __init__(self, ear_threshold: float, mar_threshold: float):
        plt.ion()  # Turn on interactive mode
        # Create a figure with two vertical subplots
        self.fig, (self.ax_ear, self.ax_mar) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Configure the EAR subplot
        self.ax_ear.set_xlabel("Frame")
        self.ax_ear.set_ylabel("EAR")
        self.ax_ear.set_title("Real-time EAR")
        self.ear_line, = self.ax_ear.plot([], [], label="EAR", color='blue')
        # Draw horizontal threshold line for EAR
        self.ax_ear.axhline(ear_threshold, color='green', linestyle='--', label="EAR Threshold")
        self.ax_ear.legend()
        
        # Configure the MAR subplot
        self.ax_mar.set_xlabel("Frame")
        self.ax_mar.set_ylabel("MAR")
        self.ax_mar.set_title("Real-time MAR")
        self.mar_line, = self.ax_mar.plot([], [], label="MAR", color='red')
        # Draw horizontal threshold line for MAR
        self.ax_mar.axhline(mar_threshold, color='green', linestyle='--', label="MAR Threshold")
        self.ax_mar.legend()
        
        self.fig.tight_layout()
        self.fig.show()

        # Lists to store data points
        self.x_data = []
        self.ear_data = []
        self.mar_data = []

    def update(self, frame_idx: int, ear: float, mar: float):
        # Append new measurement values
        self.x_data.append(frame_idx)
        self.ear_data.append(ear)
        self.mar_data.append(mar)

        # Update EAR subplot
        self.ear_line.set_data(self.x_data, self.ear_data)
        self.ax_ear.relim()
        self.ax_ear.autoscale_view()

        # Update MAR subplot
        self.mar_line.set_data(self.x_data, self.mar_data)
        self.ax_mar.relim()
        self.ax_mar.autoscale_view()

        # Redraw the figure canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001) 
        
        
# How to use the RealTimePlotter
# Initialize the real-time plotter and frame counter
# plotter = RealTimePlotter(config.EAR_THRESHOLD, config.MAR_THRESHOLD)
# frame_count = 0
# Update real-time plot with the latest EAR and MAR values
# plotter.update(frame_count, metrics.avg_ear, metrics.mar)
# frame_count += 1  # Increment frame counter

