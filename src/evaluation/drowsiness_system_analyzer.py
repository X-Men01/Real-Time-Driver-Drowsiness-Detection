import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

class DrowsinessSystemAnalyzer:
    """
    Analyzer for drowsiness detection system evaluation results.
    Processes multiple evaluation runs with different parameter combinations.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Path to directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        self.all_results = []
        self.metrics_df = None
        self.csv_data = {}
        
        # Verify the results directory exists
        if not self.results_dir.exists() or not self.results_dir.is_dir():
            raise ValueError(f"Results directory not found: {self.results_dir}")
            
        print(f"Initialized analyzer for: {self.results_dir}") 

    def collect_data(self):
        """
        Traverse all evaluation directories and extract results.
        
        This method:
        1. Scans all subdirectories in the results directory
        2. Extracts window size and threshold from directory names
        3. Loads the evaluation metrics from JSON files
        4. Stores the combined data for further analysis
        
        Returns:
            self for method chaining
        """
        print("Collecting data from evaluation directories...")
        
        # Pattern to extract window size and threshold from directory names
        pattern = r'window_(\d+)_threshold_(\d+)'
        
        # Count for progress reporting
        total_dirs = len([d for d in self.results_dir.iterdir() if d.is_dir()])
        processed_dirs = 0
        
        # Traverse all subdirectories
        for eval_dir in self.results_dir.iterdir():
            if not eval_dir.is_dir():
                continue
                
            # Extract window size and threshold from directory name
            match = re.search(pattern, eval_dir.name)
            if not match:
                print(f"Warning: Could not parse parameters from directory: {eval_dir.name}")
                continue
                
            window_size = int(match.group(1))
            threshold = int(match.group(2)) / 100  # Convert from percentile notation
            
            # Path to metrics file
            metrics_file = eval_dir / "evaluation_metrics.json"
            config_file = eval_dir / "evaluation_config.json"
            csv_file = eval_dir / "video_evaluation_results.csv"
            
            # Skip if metrics file doesn't exist
            if not metrics_file.exists():
                print(f"Warning: No metrics file found in {eval_dir.name}")
                continue
                
            try:
                # Load metrics
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Load configuration if available
                config = {}
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                
                # Store the result
                result = {
                    'directory': eval_dir.name,
                    'window_size': window_size,
                    'threshold': threshold,
                    **metrics
                }
                
                # Add config parameters if available
                if config:
                    result['config'] = config
                
                # Save CSV file path for later detailed analysis
                if csv_file.exists():
                    result['csv_file'] = str(csv_file)
                
                self.all_results.append(result)
                
                # Update progress
                processed_dirs += 1
                if processed_dirs % 10 == 0 or processed_dirs == total_dirs:
                    print(f"Processed {processed_dirs}/{total_dirs} directories")
                
            except Exception as e:
                print(f"Error processing {eval_dir.name}: {str(e)}")
        
        print(f"Data collection complete. Found {len(self.all_results)} valid evaluation runs.")
        return self 

    def create_metrics_dataframe(self):
        """
        Convert collected results into a structured pandas DataFrame.
        
        Creates a DataFrame with columns for parameters (window_size, threshold)
        and all evaluation metrics. Captures configuration once.
        
        Returns:
            self for method chaining
        """
        if not self.all_results:
            print("No results to process. Run collect_data() first.")
            return self
            
        print("Creating metrics DataFrame...")
        
        # Create DataFrame from collected results
        self.metrics_df = pd.DataFrame(self.all_results)
        
        # Extract configuration from the first result only
        if 'config' in self.metrics_df.columns and not self.metrics_df['config'].empty:
            # Save the first config as reference
            self.config = self.metrics_df['config'].iloc[0]
            
            # Print configuration parameters
            print("\nConfiguration parameters:")
            for param, value in self.config.items():
                print(f"  {param}: {value}")
            
            # Remove config column since we've saved it as a class attribute
            self.metrics_df = self.metrics_df.drop('config', axis=1)
        
        # Calculate additional derived metrics
        # Latency to accuracy ratio
        if 'average_detection_latency' in self.metrics_df.columns and 'accuracy' in self.metrics_df.columns:
            self.metrics_df['latency_to_accuracy_ratio'] = (
                self.metrics_df['average_detection_latency'] / self.metrics_df['accuracy']
            )
        
        # F1-score like metric combining drowsy detection and false positive rates
        if 'drowsy_detection_rate' in self.metrics_df.columns and 'false_positive_rate' in self.metrics_df.columns:
            # Convert false_positive_rate to precision (1 - false_positive_rate)
            precision = 1 - self.metrics_df['false_positive_rate'] 
            recall = self.metrics_df['drowsy_detection_rate']  # No division by 100
            
            # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
            # Handle division by zero
            denominator = precision + recall
            self.metrics_df['detection_f1_score'] = np.where(
                denominator > 0,
                2 * (precision * recall) / denominator,
                0
            )
        
        print(f"Created DataFrame with {len(self.metrics_df)} rows and {len(self.metrics_df.columns)} columns")
        return self 

    def generate_heatmaps(self, metrics=None, save_dir=None):
        """
        Generate heatmaps for specified metrics across window sizes and thresholds.
        
        Args:
            metrics: List of metrics to visualize. If None, uses default key metrics.
            save_dir: Directory to save plots. If None, plots are displayed but not saved.
            
        Returns:
            self for method chaining
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
            
        print("Generating heatmaps...")
        
        # Default metrics if none specified
        if metrics is None:
            metrics = [
                'accuracy', 
                'drowsy_detection_rate', 
                'false_positive_rate',
                'average_detection_latency', 
                'multiple_detection_rate'
            ]
        
        # Define optimization direction for each metric
        optimization_directions = {
            'accuracy': 'higher',
            'drowsy_detection_rate': 'closer to 1.0',
            'false_positive_rate': 'lower',
            'average_detection_latency': 'lower',
            'multiple_detection_rate': 'lower',
            'detection_f1_score': 'higher',
            'latency_to_accuracy_ratio': 'lower'
        }
        
        # Create save directory if specified
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
        
        # Create a pivot table for each metric
        for metric in metrics:
            if metric not in self.metrics_df.columns:
                print(f"Warning: Metric '{metric}' not found in DataFrame. Skipping.")
                continue
                
            print(f"Creating heatmap for: {metric}")
            
            # Get the optimization direction
            if metric in optimization_directions:
                direction = optimization_directions[metric]
            else:
                direction = 'higher'  # Default
                print(f"  Warning: Optimization direction not defined for {metric}, assuming higher is better.")
            
            # Create pivot table
            pivot = self.metrics_df.pivot_table(
                values=metric,
                index='window_size',
                columns='threshold',
                aggfunc='mean'
            )
            
            # Sort indices to ensure correct ordering
            pivot = pivot.sort_index(ascending=True)
            pivot = pivot.sort_index(axis=1, ascending=True)
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Determine colormap based on optimization direction
            if direction == 'lower':
                cmap = 'viridis_r'  # Reversed colormap for metrics where lower is better
                title_direction = "Lower is Better"
            elif direction == 'closer to 1.0':
                # Custom colormap centered at 1.0
                # Values closer to 1.0 will be dark green, values further away will be yellow/red
                cmap = 'RdYlGn_r'  # Red-Yellow-Green reversed
                title_direction = "Closer to 1.0 is Better"
            else:  # 'higher'
                cmap = 'viridis'
                title_direction = "Higher is Better"
            
            # Create heatmap
            ax = sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                linewidths=0.5,
                cbar_kws={'label': metric}
            )
            
            # Handle potential NaN values in the pivot table
            pivot_values = pivot.values.copy()
            
            # Replace NaN values with appropriate extreme values for finding min/max
            if direction == 'higher':
                pivot_values[np.isnan(pivot_values)] = -np.inf  # NaN becomes worst value
                best_idx = np.unravel_index(np.nanargmax(pivot_values), pivot_values.shape)
            elif direction == 'lower':
                pivot_values[np.isnan(pivot_values)] = np.inf  # NaN becomes worst value
                best_idx = np.unravel_index(np.nanargmin(pivot_values), pivot_values.shape)
            elif direction == 'closer to 1.0':
                # Find the value closest to 1.0, ignoring NaNs
                distance_from_1 = np.abs(pivot_values - 1.0)
                distance_from_1[np.isnan(distance_from_1)] = np.inf  # NaN becomes worst value
                best_idx = np.unravel_index(np.nanargmin(distance_from_1), distance_from_1.shape)
            
            # Ensure the indices are within range
            best_idx = (
                min(best_idx[0], len(pivot.index) - 1),
                min(best_idx[1], len(pivot.columns) - 1)
            )
            
            # Mark the best value with a box
            ax.add_patch(plt.Rectangle(
                (best_idx[1], best_idx[0]), 
                1, 1, 
                fill=False, 
                edgecolor='white', 
                lw=2, 
                alpha=0.8
            ))
            
            # Set labels and title
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Window Size')
            ax.set_title(f'Effect of Window Size and Threshold on {metric.replace("_", " ").title()}\n({title_direction})')
            
            # Add text annotation for the best value
            best_window = pivot.index[best_idx[0]]
            best_threshold = pivot.columns[best_idx[1]]
            best_value = pivot.loc[best_window, best_threshold]
            
            # Only display if value is not NaN
            if not np.isnan(best_value):
                plt.figtext(
                    0.5, 0.01, 
                    f'Best Value: {best_value:.4f} at Window Size={best_window}, Threshold={best_threshold:.2f}',
                    ha='center', 
                    fontsize=12,
                    bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}
                )
            else:
                plt.figtext(
                    0.5, 0.01, 
                    f'Warning: Best value could not be determined (missing data at Window={best_window}, Threshold={best_threshold:.2f})',
                    ha='center', 
                    fontsize=12,
                    bbox={'facecolor': 'yellow', 'alpha': 0.8, 'pad': 5}
                )
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            # Save or show plot
            if save_dir:
                file_path = save_path / f'heatmap_{metric}.png'
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"  Saved to: {file_path}")
            else:
                plt.show()
                
            plt.close()
        
        print("Heatmap generation complete.")
        return self 

    