# DON'T RUN THIS FILE, IT'S FOR REFERENCE ONLY
# AND DON'T TRY TO UNDERSTAND IT, BECAUSE I DON'T KNOW HOW IT WORKS LOL :) 

# WARNING: This code was written by an AI, modified by a human who was pretending to understand it.
# If you're reading this comment, you're already in too deep.

# Legend says if you stare at this code for too long, you'll start seeing matrix code.
# My debugging strategy: add print statements and hope for enlightenment.

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

    def create_combined_key_metrics_plot(self, save_path=None):
        """
        Creates a specialized multi-panel visualization focusing on accuracy and multiple_detection_rate.
        
        Args:
            save_path: Path to save the plot. If None, the plot is displayed but not saved.
        
        Returns:
            self for method chaining
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
        
        print("Creating combined visualization for key metrics...")
        
        # Create figure with 2x2 grid layout
        fig = plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
        
        # Define metrics and their properties
        metrics = ['accuracy', 'multiple_detection_rate']
        titles = ['Accuracy (Higher is Better)', 'Multiple Detection Rate (Higher is Better)']
        cmaps = ['viridis', 'viridis']  # Assuming higher is better for both
        
        # Create pivot tables
        pivots = []
        best_params = []
        
        for i, metric in enumerate(metrics):
            pivot = self.metrics_df.pivot_table(
                values=metric, 
                index='window_size', 
                columns='threshold', 
                aggfunc='mean'
            ).sort_index().sort_index(axis=1)
            pivots.append(pivot)
            
            # Find best parameters for this metric
            best_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
            best_window = pivot.index[best_idx[0]]
            best_threshold = pivot.columns[best_idx[1]]
            best_value = pivot.loc[best_window, best_threshold]
            best_params.append((best_window, best_threshold, best_value))
        
        # 1. Top left: Accuracy heatmap
        ax1 = plt.subplot(gs[0, 0])
        sns.heatmap(pivots[0], annot=True, fmt='.3f', cmap=cmaps[0], linewidths=0.5, ax=ax1)
        ax1.set_title(titles[0], fontsize=14)
        
        # Mark the best value
        best_idx = np.unravel_index(np.nanargmax(pivots[0].values), pivots[0].values.shape)
        ax1.add_patch(plt.Rectangle((best_idx[1], best_idx[0]), 1, 1, fill=False, edgecolor='white', lw=2))
        
        # 2. Top right: Multiple Detection Rate heatmap
        ax2 = plt.subplot(gs[0, 1])
        sns.heatmap(pivots[1], annot=True, fmt='.3f', cmap=cmaps[1], linewidths=0.5, ax=ax2)
        ax2.set_title(titles[1], fontsize=14)
        
        # Mark the best value
        best_idx = np.unravel_index(np.nanargmax(pivots[1].values), pivots[1].values.shape)
        ax2.add_patch(plt.Rectangle((best_idx[1], best_idx[0]), 1, 1, fill=False, edgecolor='white', lw=2))
        
        # 3. Bottom left: Combined metric scatter plot
        ax3 = plt.subplot(gs[1, 0])
        scatter = ax3.scatter(
            self.metrics_df['accuracy'], 
            self.metrics_df['multiple_detection_rate'],
            c=self.metrics_df['window_size'],  # Color by window size
            s=self.metrics_df['threshold']*100,  # Size by threshold
            alpha=0.7,
            cmap='viridis'
        )
        
        # Add colorbar for window size
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Window Size')
        
        # Add annotations for extreme points
        max_acc_idx = self.metrics_df['accuracy'].idxmax()
        max_mdr_idx = self.metrics_df['multiple_detection_rate'].idxmax()
        
        # Find the index that maximizes both metrics (simple sum approach)
        combined_score = self.metrics_df['accuracy'] + self.metrics_df['multiple_detection_rate']
        max_combined_idx = combined_score.idxmax()
        
        for idx, label in [(max_acc_idx, 'Max Accuracy'), 
                        (max_mdr_idx, 'Max Multiple Detection'), 
                        (max_combined_idx, 'Best Combined')]:
            x = self.metrics_df.loc[idx, 'accuracy']
            y = self.metrics_df.loc[idx, 'multiple_detection_rate']
            w = self.metrics_df.loc[idx, 'window_size']
            t = self.metrics_df.loc[idx, 'threshold']
            
            ax3.annotate(
                f'{label}\nWindow={w}, Thresh={t:.2f}',
                xy=(x, y),
                xytext=(10, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5)
            )
        
        ax3.set_xlabel('Accuracy', fontsize=12)
        ax3.set_ylabel('Multiple Detection Rate', fontsize=12)
        ax3.set_title('Relationship Between Key Metrics', fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Bottom right: Summary text
        ax4 = plt.subplot(gs[1, 1])
        ax4.axis('off')
        
        # Add summary text
        summary_text = (
            "SUMMARY OF KEY METRICS\n\n"
            f"Best Accuracy: {best_params[0][2]:.4f}\n"
            f"  Window Size: {best_params[0][0]}\n"
            f"  Threshold: {best_params[0][1]:.2f}\n\n"
            f"Best Multiple Detection Rate: {best_params[1][2]:.4f}\n"
            f"  Window Size: {best_params[1][0]}\n"
            f"  Threshold: {best_params[1][1]:.2f}\n\n"
            "Best Combined Configuration:\n"
            f"  Window Size: {self.metrics_df.loc[max_combined_idx, 'window_size']}\n"
            f"  Threshold: {self.metrics_df.loc[max_combined_idx, 'threshold']:.2f}\n"
            f"  Accuracy: {self.metrics_df.loc[max_combined_idx, 'accuracy']:.4f}\n"
            f"  Multiple Detection Rate: {self.metrics_df.loc[max_combined_idx, 'multiple_detection_rate']:.4f}"
        )
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=12, va='top', family='monospace')
        
        # Adjust layout and add super title
        plt.suptitle('Optimizing Accuracy and Multiple Detection Rate', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined visualization to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return self

    def create_metric_scatter_plot(self, save_path=None):
        """
        Create a scatter plot showing accuracy vs multiple_detection_rate.
        
        Args:
            save_path: Path to save the plot. If None, plot is displayed but not saved.
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
            
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with window size as color and threshold as size
        scatter = plt.scatter(
            self.metrics_df['accuracy'],
            self.metrics_df['multiple_detection_rate'],
            c=self.metrics_df['window_size'],
            s=self.metrics_df['threshold']*100,
            alpha=0.7,
            cmap='viridis'
        )
        
        # Add colorbar and legend
        cbar = plt.colorbar(scatter)
        cbar.set_label('Window Size')
        
        # Add a size legend
        for threshold in [0.7, 0.8, 0.9]:
            plt.scatter([], [], s=threshold*100, c='gray', alpha=0.7, 
                        label=f'Threshold = {threshold}')
        
        # Find optimal points
        best_acc_idx = self.metrics_df['accuracy'].idxmax()
        best_mdr_idx = self.metrics_df['multiple_detection_rate'].idxmin()  # Lower is better
        
        # Calculate a combined metric (normalize both metrics to [0,1] range)
        acc_norm = (self.metrics_df['accuracy'] - self.metrics_df['accuracy'].min()) / (self.metrics_df['accuracy'].max() - self.metrics_df['accuracy'].min())
        mdr_norm = 1 - (self.metrics_df['multiple_detection_rate'] - self.metrics_df['multiple_detection_rate'].min()) / (self.metrics_df['multiple_detection_rate'].max() - self.metrics_df['multiple_detection_rate'].min())
        combined_score = acc_norm + mdr_norm
        best_combined_idx = combined_score.idxmax()
        
        # Highlight optimal points
        for idx, label, marker in [
            (best_acc_idx, 'Best Accuracy', '^'), 
            (best_mdr_idx, 'Best MDR (Lower)', 's'),
            (best_combined_idx, 'Best Combined', '*')
        ]:
            row = self.metrics_df.loc[idx]
            plt.scatter(
                row['accuracy'], 
                row['multiple_detection_rate'],
                s=200, 
                marker=marker, 
                edgecolors='black', 
                linewidths=2,
                label=f"{label}: W={row['window_size']}, T={row['threshold']:.2f}"
            )
        
        # Add labels and title
        plt.xlabel('Accuracy (Higher is Better)', fontsize=12)
        plt.ylabel('Multiple Detection Rate (Lower is Better)', fontsize=12)
        plt.title('Accuracy vs Multiple Detection Rate', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved scatter plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return self

    def create_bubble_chart(self, save_path=None):
        """
        Create a bubble chart with contour lines showing the parameter landscape.
        
        Args:
            save_path: Path to save the plot. If None, plot is displayed but not saved.
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create a meshgrid for contour plotting
        window_sizes = sorted(self.metrics_df['window_size'].unique())
        thresholds = sorted(self.metrics_df['threshold'].unique())
        
        X, Y = np.meshgrid(thresholds, window_sizes)
        Z_acc = np.zeros(X.shape)
        Z_mdr = np.zeros(X.shape)
        
        # Fill in the grid with values
        for i, w in enumerate(window_sizes):
            for j, t in enumerate(thresholds):
                subset = self.metrics_df[(self.metrics_df['window_size'] == w) & 
                                        (self.metrics_df['threshold'] == t)]
                if not subset.empty:
                    Z_acc[i, j] = subset['accuracy'].values[0]
                    Z_mdr[i, j] = subset['multiple_detection_rate'].values[0]
        
        # Create contour lines for accuracy
        CS1 = ax.contour(X, Y, Z_acc, levels=10, colors='blue', alpha=0.6)
        ax.clabel(CS1, inline=True, fontsize=8, fmt='%.3f')
        
        # Create contour lines for multiple detection rate
        CS2 = ax.contour(X, Y, Z_mdr, levels=10, colors='red', alpha=0.6)
        ax.clabel(CS2, inline=True, fontsize=8, fmt='%.3f')
        
        # Add bubble plot on top
        for _, row in self.metrics_df.iterrows():
            size = 1000 * (row['accuracy'] / row['multiple_detection_rate'])  # Size based on ratio (higher is better)
            ax.scatter(
                row['threshold'], 
                row['window_size'],
                s=size,
                alpha=0.5,
                edgecolors='gray'
            )
        
        # Find and mark optimal points
        best_acc_idx = self.metrics_df['accuracy'].idxmax()
        best_mdr_idx = self.metrics_df['multiple_detection_rate'].idxmin()
        
        # Calculate combined score (higher acc, lower mdr)
        self.metrics_df['combined_score'] = self.metrics_df['accuracy'] / self.metrics_df['multiple_detection_rate']
        best_combined_idx = self.metrics_df['combined_score'].idxmax()
        
        # Highlight best points
        for idx, label, marker, color in [
            (best_acc_idx, 'Best Accuracy', '^', 'blue'),
            (best_mdr_idx, 'Best MDR (Lower)', 's', 'red'),
            (best_combined_idx, 'Best Combined', '*', 'green')
        ]:
            row = self.metrics_df.loc[idx]
            ax.scatter(
                row['threshold'],
                row['window_size'],
                s=300,
                marker=marker,
                color=color,
                edgecolors='black',
                label=f"{label}: Acc={row['accuracy']:.3f}, MDR={row['multiple_detection_rate']:.3f}"
            )
        
        # Add labels and title
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Window Size', fontsize=12)
        ax.set_title('Parameter Bubble Chart with Accuracy (blue) and MDR (red) Contours', fontsize=14)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved bubble chart to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return self
    
    def create_performance_table(self, save_path=None):
        """
        Create a visually formatted table showing the top performing configurations.
        
        Args:
            save_path: Path to save the plot. If None, plot is displayed but not saved.
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
        
        # Calculate performance score (higher accuracy, lower multiple_detection_rate)
        df = self.metrics_df.copy()
        
        # Calculate ranges for normalization, handling division by zero cases
        acc_range = df['accuracy'].max() - df['accuracy'].min()
        mdr_range = df['multiple_detection_rate'].max() - df['multiple_detection_rate'].min()
        
        # Normalize metrics to [0, 1] range for fair comparison
        if acc_range > 0:
            df['norm_accuracy'] = (df['accuracy'] - df['accuracy'].min()) / acc_range
        else:
            df['norm_accuracy'] = 0.5  # Default if all values are the same
        
        if mdr_range > 0:
            df['norm_mdr'] = 1 - (df['multiple_detection_rate'] - df['multiple_detection_rate'].min()) / mdr_range
        else:
            df['norm_mdr'] = 0.5  # Default if all values are the same
        
        # Calculate combined score
        df['score'] = 0.5*df['norm_accuracy'] + 0.5*df['norm_mdr']
        
        # Sort by score
        df = df.sort_values('score', ascending=False)
        
        # Select top 20 configurations
        top_df = df.head(10)[['window_size', 'threshold', 'accuracy', 'multiple_detection_rate', 'score']]
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        cell_text = []
        for _, row in top_df.iterrows():
            cell_text.append([
                f"{row['window_size']:.0f}",
                f"{row['threshold']:.2f}",
                f"{row['accuracy']:.4f}",
                f"{row['multiple_detection_rate']:.4f}",
                f"{row['score']:.4f}"
            ])
        
        # Create table with conditional coloring
        table = plt.table(
            cellText=cell_text,
            colLabels=['Window Size', 'Threshold', 'Accuracy\n(Higher Better)', 'Multiple Detection Rate\n(Lower Better)', 'Combined Score'],
            loc='center',
            cellLoc='center'
        )
        
        # Set font sizes
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Apply conditional formatting with safety bounds
        for i in range(len(cell_text)):
            # Accuracy cell (higher is better - more green)
            try:
                acc_val = float(cell_text[i][2])
                if acc_range > 0:
                    acc_norm = (acc_val - df['accuracy'].min()) / acc_range
                    # Ensure value is within 0-1 range
                    acc_norm = max(0, min(1, acc_norm))
                else:
                    acc_norm = 0.5
                acc_color = (1-acc_norm, 1, 1-acc_norm)  # Green gradient
                table[(i+1, 2)].set_facecolor(acc_color)
            except Exception as e:
                print(f"Skipping accuracy color for row {i}: {e}")
            
            # MDR cell (lower is better - less red)
            try:
                mdr_val = float(cell_text[i][3])
                if mdr_range > 0:
                    mdr_norm = (mdr_val - df['multiple_detection_rate'].min()) / mdr_range
                    # Ensure value is within 0-1 range
                    mdr_norm = max(0, min(1, mdr_norm))
                else:
                    mdr_norm = 0.5
                mdr_color = (1, 1-mdr_norm, 1-mdr_norm)  # Red gradient
                table[(i+1, 3)].set_facecolor(mdr_color)
            except Exception as e:
                print(f"Skipping MDR color for row {i}: {e}")
            
            # Combined score
            try:
                score_val = float(cell_text[i][4])
                score_range = df['score'].max() - df['score'].min()
                if score_range > 0:
                    score_norm = (score_val - df['score'].min()) / score_range
                    # Ensure value is within 0-1 range
                    score_norm = max(0, min(1, score_norm))
                else:
                    score_norm = 0.5
                score_color = (1-score_norm, 1-0.5*score_norm, 1)  # Purple gradient
                table[(i+1, 4)].set_facecolor(score_color)
            except Exception as e:
                print(f"Skipping score color for row {i}: {e}")
        
        # Highlight the best row
        for j in range(5):
            table[(1, j)].set_edgecolor('darkblue')
            table[(1, j)].set_linewidth(3)
        
        # Add a title
        plt.title('Top 10 Parameter Configurations by Performance (0.5 Acc, 0.5 MDR)', fontsize=16, pad=20)
        
        # Add explanatory text
        explanatory_text = (
            "Accuracy: Higher values indicate better overall classification performance.\n"
            "Multiple Detection Rate: Lower values are better (fewer redundant detections).\n"
            "Combined Score: 0.5 × normalized accuracy + 0.5 × normalized (1-MDR) (higher is better)."
        )
        plt.figtext(0.5, 0.01, explanatory_text, ha='center', fontsize=10, 
                    bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved performance table to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return self

    def visualize_top_configs_simple(self, top_n=20, acc_weight=0.7, mdr_weight=0.3, save_path=None):
        """
        Create a simple bar chart visualization of top configurations.
        
        Args:
            top_n: Number of top configurations to display (default: 20)
            acc_weight: Weight for accuracy in score calculation (default: 0.7)
            mdr_weight: Weight for multiple detection rate in score calculation (default: 0.3)
            save_path: Path to save the plot. If None, plot is displayed but not saved.
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
        
        # Prepare data
        df = self.metrics_df.copy()
        
        # Normalize metrics - with safe division
        acc_range = df['accuracy'].max() - df['accuracy'].min()
        mdr_range = df['multiple_detection_rate'].max() - df['multiple_detection_rate'].min()
        
        if acc_range > 0:
            df['norm_accuracy'] = (df['accuracy'] - df['accuracy'].min()) / acc_range
        else:
            df['norm_accuracy'] = 0.5
        
        if mdr_range > 0:
            df['norm_mdr'] = 1 - (df['multiple_detection_rate'] - df['multiple_detection_rate'].min()) / mdr_range
        else:
            df['norm_mdr'] = 0.5
        
        # Calculate weighted score
        df['score'] = acc_weight * df['norm_accuracy'] + mdr_weight * df['norm_mdr']
        
        # Get top configurations
        top_df = df.sort_values('score', ascending=False).head(top_n)
        
        # Create a simple figure
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        y_pos = range(len(top_df))
        
        # Create bar labels
        labels = [f"W={row['window_size']:.0f}, T={row['threshold']:.2f}" for _, row in top_df.iterrows()]
        
        # Create a simple horizontal bar chart
        bars = plt.barh(y_pos, top_df['score'], height=0.6)
        
        # Add data labels to bars
        for i, (_, row) in enumerate(top_df.iterrows()):
            plt.text(
                row['score'] + 0.01, 
                i, 
                f"Acc={row['accuracy']:.3f}, MDR={row['multiple_detection_rate']:.3f}",
                va='center'
            )
        
        # Add configuration labels to y-axis
        plt.yticks(y_pos, labels)
        
        # Add a title and labels
        plt.title(f'Top {top_n} Configurations (Acc Weight: {acc_weight:.1f}, MDR Weight: {mdr_weight:.1f})', fontsize=14)
        plt.xlabel('Combined Score', fontsize=12)
        
        # Add grid lines
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved simple top configurations chart to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return self

    def plot_top_config_distributions(self, top_n=20, acc_weight=0.7, mdr_weight=0.3, save_path=None):
        """
        Plot distributions of parameters and metrics for top configurations.
        
        Args:
            top_n: Number of top configurations to analyze (default: 20)
            acc_weight: Weight for accuracy in combined score (default: 0.7)
            mdr_weight: Weight for multiple detection rate in combined score (default: 0.3)
            save_path: Path to save the plot. If None, plot is displayed but not saved.
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
        
        print(f"Analyzing distributions for top {top_n} configurations...")
        
        # Prepare data
        df = self.metrics_df.copy()
        
        # Normalize metrics for fair comparison
        acc_range = df['accuracy'].max() - df['accuracy'].min()
        mdr_range = df['multiple_detection_rate'].max() - df['multiple_detection_rate'].min()
        
        if acc_range > 0:
            df['norm_accuracy'] = (df['accuracy'] - df['accuracy'].min()) / acc_range
        else:
            df['norm_accuracy'] = 0.5
        
        if mdr_range > 0:
            df['norm_mdr'] = 1 - (df['multiple_detection_rate'] - df['multiple_detection_rate'].min()) / mdr_range
        else:
            df['norm_mdr'] = 0.5
        
        # Calculate weighted score
        df['score'] = acc_weight * df['norm_accuracy'] + mdr_weight * df['norm_mdr']
        
        # Get top configurations
        top_df = df.sort_values('score', ascending=False).head(top_n)
        
        # Create a 2x2 grid of histograms/distributions
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Window Size Distribution (top left)
        axs[0, 0].hist(top_df['window_size'], bins=10, color='royalblue', alpha=0.7, edgecolor='black')
        axs[0, 0].set_title('Window Size Distribution', fontsize=12)
        axs[0, 0].set_xlabel('Window Size')
        axs[0, 0].set_ylabel('Frequency')
        
        # Add vertical line for most frequent window size
        window_mode = top_df['window_size'].value_counts().idxmax()
        axs[0, 0].axvline(window_mode, color='red', linestyle='--', 
                          label=f'Most Common: {window_mode}')
        axs[0, 0].legend()
        
        # 2. Threshold Distribution (top right)
        axs[0, 1].hist(top_df['threshold'], bins=10, color='green', alpha=0.7, edgecolor='black')
        axs[0, 1].set_title('Threshold Distribution', fontsize=12)
        axs[0, 1].set_xlabel('Threshold')
        axs[0, 1].set_ylabel('Frequency')
        
        # Add vertical line for most frequent threshold
        threshold_mode = top_df['threshold'].value_counts().idxmax()
        axs[0, 1].axvline(threshold_mode, color='red', linestyle='--', 
                          label=f'Most Common: {threshold_mode:.2f}')
        axs[0, 1].legend()
        
        # 3. Accuracy Distribution (bottom left)
        sns.histplot(top_df['accuracy'], bins=10, kde=True, color='purple', 
                    alpha=0.6, ax=axs[1, 0], edgecolor='black')
        axs[1, 0].set_title('Accuracy Distribution', fontsize=12)
        axs[1, 0].set_xlabel('Accuracy')
        axs[1, 0].set_ylabel('Frequency')
        
        # Add vertical line for mean accuracy
        mean_acc = top_df['accuracy'].mean()
        axs[1, 0].axvline(mean_acc, color='red', linestyle='--', 
                         label=f'Mean: {mean_acc:.4f}')
        axs[1, 0].legend()
        
        # 4. Multiple Detection Rate Distribution (bottom right)
        sns.histplot(top_df['multiple_detection_rate'], bins=10, kde=True, 
                    color='orange', alpha=0.6, ax=axs[1, 1], edgecolor='black')
        axs[1, 1].set_title('Multiple Detection Rate Distribution', fontsize=12)
        axs[1, 1].set_xlabel('Multiple Detection Rate')
        axs[1, 1].set_ylabel('Frequency')
        
        # Add vertical line for mean MDR
        mean_mdr = top_df['multiple_detection_rate'].mean()
        axs[1, 1].axvline(mean_mdr, color='red', linestyle='--', 
                         label=f'Mean: {mean_mdr:.4f}')
        axs[1, 1].legend()
        
        # Add overall title
        plt.suptitle(f'Parameter and Metric Distributions for All Configurations', 
                    fontsize=14, y=0.98)
        
        # Add footnote
        footnote = (
            f"Combined Score: {acc_weight:.1f} × norm_accuracy + {mdr_weight:.1f} × norm(1-MDR)\n"
            "These histograms show the distribution of parameters and metrics among the best configurations."
        )
        plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=10, 
                   bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plots to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return self

    def plot_window_vs_accuracy(self, save_path=None):
        """
        Create a simple plot showing how window size affects accuracy and multiple detection rate.
        
        Args:
            save_path: Path to save the plot. If None, plot is displayed but not saved.
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Group data by window size
        window_groups = self.metrics_df.groupby('window_size')
        
        # Plot 1: Window Size vs Accuracy
        window_accuracy = window_groups['accuracy'].mean()
        ax1.plot(window_accuracy.index, window_accuracy.values, marker='o', linestyle='-', color='blue')
        ax1.set_xlabel('Window Size')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Window Size vs. Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Add text showing the best window size for accuracy
        best_window_acc = window_accuracy.idxmax()
        best_acc = window_accuracy.max()
        ax1.annotate(
            f'Best: {best_window_acc} (Acc={best_acc:.4f})',
            xy=(best_window_acc, best_acc),
            xytext=(5, -15),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red')
        )
        
        # Plot 2: Window Size vs Multiple Detection Rate (lower is better)
        window_mdr = window_groups['multiple_detection_rate'].mean()
        ax2.plot(window_mdr.index, window_mdr.values, marker='o', linestyle='-', color='red')
        ax2.set_xlabel('Window Size')
        ax2.set_ylabel('Multiple Detection Rate')
        ax2.set_title('Window Size vs. Multiple Detection Rate (Lower is Better)')
        ax2.grid(True, alpha=0.3)
        
        # Add text showing the best window size for MDR
        best_window_mdr = window_mdr.idxmin()  # Min for MDR since lower is better
        best_mdr = window_mdr.min()
        ax2.annotate(
            f'Best: {best_window_mdr} (MDR={best_mdr:.4f})',
            xy=(best_window_mdr, best_mdr),
            xytext=(5, 10),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='blue')
        )
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved window size analysis to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return self
    
    def find_best_parameters(self):
        """
        Find and print the best parameter combinations for different metrics.
        
        This method identifies:
        1. Parameters for best accuracy
        2. Parameters for minimum multiple detection rate
        3. Parameters for best combined score (weighted accuracy and MDR)
        
        Returns:
            Dict containing the best parameters and their values
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return None
            
        print("\n=== Best Parameter Combinations ===\n")
        
        # Find best accuracy
        best_acc_idx = self.metrics_df['accuracy'].idxmax()
        best_acc_row = self.metrics_df.loc[best_acc_idx]
        
        print(f"Best Accuracy: {best_acc_row['accuracy']:.4f}")
        print(f"  Window Size: {best_acc_row['window_size']}")
        print(f"  Threshold: {best_acc_row['threshold']:.2f}")
        print(f"  Multiple Detection Rate: {best_acc_row['multiple_detection_rate']:.4f}")
        print(f"  False Positive Rate: {best_acc_row['false_positive_rate']:.4f}")
        print(f"  Average Detection Latency: {best_acc_row['average_detection_latency']:.2f} seconds")
        
        # Find best (minimum) MDR
        best_mdr_idx = self.metrics_df['multiple_detection_rate'].idxmin()
        best_mdr_row = self.metrics_df.loc[best_mdr_idx]
        
        print("\nBest Multiple Detection Rate: {:.4f}".format(best_mdr_row['multiple_detection_rate']))
        print(f"  Window Size: {best_mdr_row['window_size']}")
        print(f"  Threshold: {best_mdr_row['threshold']:.2f}")
        print(f"  Accuracy: {best_mdr_row['accuracy']:.4f}")
        print(f"  False Positive Rate: {best_mdr_row['false_positive_rate']:.4f}")
        print(f"  Average Detection Latency: {best_mdr_row['average_detection_latency']:.2f} seconds")
        
        # Calculate combined score (weighted accuracy and MDR)
        # Normalize metrics to 0-1 range
        df = self.metrics_df.copy()
        acc_range = df['accuracy'].max() - df['accuracy'].min()
        mdr_range = df['multiple_detection_rate'].max() - df['multiple_detection_rate'].min()
        
        if acc_range > 0:
            df['norm_accuracy'] = (df['accuracy'] - df['accuracy'].min()) / acc_range
        else:
            df['norm_accuracy'] = 0.5
            
        if mdr_range > 0:
            df['norm_mdr'] = 1 - (df['multiple_detection_rate'] - df['multiple_detection_rate'].min()) / mdr_range
        else:
            df['norm_mdr'] = 0.5
        
        # Calculate weighted score (0.6 accuracy, 0.4 MDR)
        df['combined_score'] = 0.6 * df['norm_accuracy'] + 0.4 * df['norm_mdr']
        
        # Find best combined score
        best_combined_idx = df['combined_score'].idxmax()
        best_combined_row = self.metrics_df.loc[best_combined_idx]
        
        print("\nBest Combined Score:")
        print(f"  Window Size: {best_combined_row['window_size']}")
        print(f"  Threshold: {best_combined_row['threshold']:.2f}")
        print(f"  Accuracy: {best_combined_row['accuracy']:.4f}")
        print(f"  Multiple Detection Rate: {best_combined_row['multiple_detection_rate']:.4f}")
        print(f"  False Positive Rate: {best_combined_row['false_positive_rate']:.4f}")
        print(f"  Average Detection Latency: {best_combined_row['average_detection_latency']:.2f} seconds")
        
        # Return results as a dictionary for easier access
        results = {
            'best_accuracy': {
                'window_size': best_acc_row['window_size'],
                'threshold': best_acc_row['threshold'],
                'accuracy': best_acc_row['accuracy'],
                'mdr': best_acc_row['multiple_detection_rate'],
                'fpr': best_acc_row['false_positive_rate'],
                'latency': best_acc_row['average_detection_latency']
            },
            'best_mdr': {
                'window_size': best_mdr_row['window_size'],
                'threshold': best_mdr_row['threshold'],
                'accuracy': best_mdr_row['accuracy'],
                'mdr': best_mdr_row['multiple_detection_rate'],
                'fpr': best_mdr_row['false_positive_rate'],
                'latency': best_mdr_row['average_detection_latency']
            },
            'best_combined': {
                'window_size': best_combined_row['window_size'],
                'threshold': best_combined_row['threshold'],
                'accuracy': best_combined_row['accuracy'],
                'mdr': best_combined_row['multiple_detection_rate'],
                'fpr': best_combined_row['false_positive_rate'],
                'latency': best_combined_row['average_detection_latency']
            }
        }
        
        return results
    
    def plot_metrics_distributions(self, metrics=None, top_n=20, acc_weight=0.7, mdr_weight=0.3, save_path=None):
        """
        Plot distributions of parameters and metrics for top configurations.
        
        Args:
            metrics: List of metrics to include in distributions. If None, uses defaults.
            top_n: Number of top configurations to analyze (default: 20)
            acc_weight: Weight for accuracy in combined score (default: 0.7)
            mdr_weight: Weight for multiple detection rate in combined score (default: 0.3)
            save_path: Path to save the plot. If None, plot is displayed but not saved.
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
        
        print(f"Analyzing distributions for top {top_n} configurations...")
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ['accuracy', 'multiple_detection_rate', 'drowsy_detection_rate', 'false_positive_rate']
        
        # Prepare data
        df = self.metrics_df.copy()
        
        # Normalize metrics for fair comparison
        acc_range = df['accuracy'].max() - df['accuracy'].min()
        mdr_range = df['multiple_detection_rate'].max() - df['multiple_detection_rate'].min()
        
        if acc_range > 0:
            df['norm_accuracy'] = (df['accuracy'] - df['accuracy'].min()) / acc_range
        else:
            df['norm_accuracy'] = 0.5
        
        if mdr_range > 0:
            df['norm_mdr'] = 1 - (df['multiple_detection_rate'] - df['multiple_detection_rate'].min()) / mdr_range
        else:
            df['norm_mdr'] = 0.5
        
        # Calculate weighted score
        df['score'] = acc_weight * df['norm_accuracy'] + mdr_weight * df['norm_mdr']
        
        # Get top configurations
        top_df = df.sort_values('score', ascending=False).head(top_n)
        
        # Calculate rows and columns for subplot grid
        n_metrics = len(metrics)  # +2 for window_size and threshold
        n_cols = 2
        n_rows = (n_metrics + 1) // n_cols  # Round up
        
        # Create figure with subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axs = axs.flatten()  # Flatten for easier indexing
        
        # # Plot window size distribution
        # axs[0].hist(top_df['window_size'], bins=10, color='royalblue', alpha=0.7, edgecolor='black')
        # axs[0].set_title('Window Size Distribution', fontsize=12)
        # axs[0].set_xlabel('Window Size')
        # axs[0].set_ylabel('Frequency')
        
        # # Add vertical line for most frequent window size
        # window_mode = top_df['window_size'].value_counts().idxmax()
        # axs[0].axvline(window_mode, color='red', linestyle='--', 
        #                 label=f'Most Common: {window_mode}')
        # axs[0].legend()
        
        # # Plot threshold distribution
        # axs[1].hist(top_df['threshold'], bins=10, color='green', alpha=0.7, edgecolor='black')
        # axs[1].set_title('Threshold Distribution', fontsize=12)
        # axs[1].set_xlabel('Threshold')
        # axs[1].set_ylabel('Frequency')
        
        # # Add vertical line for most frequent threshold
        # threshold_mode = top_df['threshold'].value_counts().idxmax()
        # axs[1].axvline(threshold_mode, color='red', linestyle='--', 
        #                 label=f'Most Common: {threshold_mode:.2f}')
        # axs[1].legend()
        
        # Colors for different metrics
        colors = ['purple', 'orange', 'teal', 'brown', 'olive', 'pink']
        
        # Plot distributions for each metric
        for i, metric in enumerate(metrics):
            if metric in top_df.columns:
                idx = i   # Start after window size and threshold
                sns.histplot(top_df[metric], bins=10, kde=True, 
                            color=colors[i % len(colors)], 
                            alpha=0.6, ax=axs[idx], edgecolor='black')
                axs[idx].set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=12)
                axs[idx].set_xlabel(metric.replace("_", " ").title())
                axs[idx].set_ylabel('Frequency')
                
                # Add vertical line for mean value
                mean_val = top_df[metric].mean()
                axs[idx].axvline(mean_val, color='red', linestyle='--', 
                                label=f'Mean: {mean_val:.4f}')
                axs[idx].legend()
        
        # Hide any unused subplots
        for i in range(2 + len(metrics), len(axs)):
            axs[i].axis('off')
        
        # Add overall title
        plt.suptitle(f'Parameter and Metric Distributions for Top {top_n} Configurations', 
                    fontsize=14, y=0.98)
        
        # Add footnote
        footnote = (
            f"Combined Score: {acc_weight:.1f} × norm_accuracy + {mdr_weight:.1f} × norm(1-MDR)\n"
            "These histograms show the distribution of parameters and metrics among the best configurations."
        )
        plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=10, 
                    bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plots to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return self


    def create_improved_metric_scatter_plot(self, acc_weight=0.7, mdr_weight=0.3, save_path=None):
        """
        Create an improved scatter plot showing accuracy vs multiple_detection_rate with clear optimal points.
        
        Args:
            acc_weight: Weight for accuracy in combined score (default: 0.7)
            mdr_weight: Weight for multiple detection rate in combined score (default: 0.3)
            save_path: Path to save the plot. If None, plot is displayed but not saved.
        """
        if self.metrics_df is None:
            print("No metrics data available. Run create_metrics_dataframe() first.")
            return self
            
        # Create a copy of the dataframe and calculate normalized metrics
        df = self.metrics_df.copy()
        
        # Normalize metrics for fair comparison
        acc_range = df['accuracy'].max() - df['accuracy'].min()
        mdr_range = df['multiple_detection_rate'].max() - df['multiple_detection_rate'].min()
        
        if acc_range > 0:
            df['norm_accuracy'] = (df['accuracy'] - df['accuracy'].min()) / acc_range
        else:
            df['norm_accuracy'] = 0.5
        
        if mdr_range > 0:
            df['norm_mdr'] = 1 - (df['multiple_detection_rate'] - df['multiple_detection_rate'].min()) / mdr_range
        else:
            df['norm_mdr'] = 0.5
        
        # Calculate weighted score
        df['combined_score'] = acc_weight * df['norm_accuracy'] + mdr_weight * df['norm_mdr']
        
        plt.figure(figsize=(14, 10))
        
        # Create a custom colormap for window sizes (10-100)
        norm = plt.Normalize(10, 100)
        cmap = plt.cm.viridis
        
        # Create scatter plot with window size as color and threshold as size
        scatter = plt.scatter(
            df['accuracy'],
            df['multiple_detection_rate'],
            c=df['window_size'],
            s=df['threshold']*100 + 50,  # Scale thresholds for better visibility
            alpha=0.7,
            cmap=cmap,
            norm=norm,
            edgecolors='gray',
            linewidths=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Window Size', fontsize=12)
        
        # Find optimal points
        best_acc_idx = df['accuracy'].idxmax()
        best_mdr_idx = df['multiple_detection_rate'].idxmin()
        best_combined_idx = df['combined_score'].idxmax()
        
        # Store best config information
        best_acc = df.loc[best_acc_idx]
        best_mdr = df.loc[best_mdr_idx]
        best_combined = df.loc[best_combined_idx]
        
        # Highlight optimal points with distinct markers and labels
        optimal_points = [
            (best_acc_idx, 'Best Accuracy', '^', 'blue', 13),
            (best_mdr_idx, 'Best MDR', 's', 'orange', 13),
            (best_combined_idx, 'Best Combined', '*', 'red', 16)
        ]
        
        for idx, label, marker, color, size in optimal_points:
            row = df.loc[idx]
            plt.scatter(
                row['accuracy'], 
                row['multiple_detection_rate'],
                s=200, 
                marker=marker, 
                color=color,
                edgecolors='black', 
                linewidths=1.5,
                zorder=5,
                label=f"{label}: W={int(row['window_size'])}, T={row['threshold']:.2f}"
            )
        
        # Add annotations for optimal points with offset to avoid overlapping
        for idx, label, _, _, _ in optimal_points:
            row = df.loc[idx]
            plt.annotate(
                f"W={int(row['window_size'])}, T={row['threshold']:.2f}\nAcc={row['accuracy']:.4f}, MDR={row['multiple_detection_rate']:.4f}",
                xy=(row['accuracy'], row['multiple_detection_rate']),
                xytext=(10, 10),  # Offset
                textcoords='offset points',
                backgroundcolor='white',
                alpha=0.8,
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6)
            )
        
        # Identify and plot approximate Pareto frontier
        # Sort by accuracy and find points that have lower MDR than all points with higher accuracy
        sorted_df = df.sort_values('accuracy')
        pareto_points = []
        min_mdr = float('inf')
        
        for _, row in sorted_df.iterrows():
            if row['multiple_detection_rate'] < min_mdr:
                pareto_points.append((row['accuracy'], row['multiple_detection_rate']))
                min_mdr = row['multiple_detection_rate']
        
        # Add Pareto frontier line
        if pareto_points:
            pareto_x, pareto_y = zip(*pareto_points)
            plt.plot(pareto_x, pareto_y, 'k--', alpha=0.5, label='Pareto Frontier')
        
        # Add legend at the bottom to avoid overlap with data points
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
        
        # Add summary in the top right (outside the plot area)
        summary_text = (
            f"Best Accuracy: {best_acc['accuracy']:.4f} (W={int(best_acc['window_size'])}, T={best_acc['threshold']:.2f})\n"
            f"Best MDR: {best_mdr['multiple_detection_rate']:.4f} (W={int(best_mdr['window_size'])}, T={best_mdr['threshold']:.2f})\n"
            f"Best Combined ({acc_weight:.1f}×Acc + {mdr_weight:.1f}×MDR): W={int(best_combined['window_size'])}, T={best_combined['threshold']:.2f}"
        )
        
        # Add explanatory text for threshold sizing
        plt.figtext(0.72, 0.22, "Marker size indicates threshold value\n(larger = higher threshold)",
                fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Add labels and title
        plt.xlabel('Accuracy (Higher is Better)', fontsize=12)
        plt.ylabel('Multiple Detection Rate (Lower is Better)', fontsize=12)
        plt.title('Accuracy vs Multiple Detection Rate Tradeoff Analysis', fontsize=14)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Tight layout with space for legend at bottom
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved improved scatter plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print optimal configurations for reference
        print("\nOptimal Configurations:")
        print(f"Best Accuracy: Window={int(best_acc['window_size'])}, Threshold={best_acc['threshold']:.2f}, " +
            f"Acc={best_acc['accuracy']:.4f}, MDR={best_acc['multiple_detection_rate']:.4f}")
        print(f"Best MDR: Window={int(best_mdr['window_size'])}, Threshold={best_mdr['threshold']:.2f}, " +
            f"Acc={best_mdr['accuracy']:.4f}, MDR={best_mdr['multiple_detection_rate']:.4f}")
        print(f"Best Combined: Window={int(best_combined['window_size'])}, Threshold={best_combined['threshold']:.2f}, " +
            f"Acc={best_combined['accuracy']:.4f}, MDR={best_combined['multiple_detection_rate']:.4f}")
        
        return self
    
    def create_enhanced_table(self, top_n=15, save_path=None):
        """
        Create an enhanced table with visual indicators of performance metrics.
        """
        df = self.metrics_df.copy()
        
        # Calculate combined score
        df['norm_accuracy'] = (df['accuracy'] - df['accuracy'].min()) / (df['accuracy'].max() - df['accuracy'].min())
        df['norm_mdr'] = 1 - (df['multiple_detection_rate'] - df['multiple_detection_rate'].min()) / (df['multiple_detection_rate'].max() - df['multiple_detection_rate'].min())
        df['combined_score'] = 0.7 * df['norm_accuracy'] + 0.3 * df['norm_mdr']
        
        # Sort and select top configurations
        top_df = df.sort_values('combined_score', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, top_n * 0.5 + 2))
        ax.axis('off')
        
        # Prepare data for table
        data = []
        for _, row in top_df.iterrows():
            data.append([
                int(row['window_size']),
                f"{row['threshold']:.2f}",
                f"{row['accuracy']:.4f}",
                f"{row['multiple_detection_rate']:.4f}",
                f"{row['false_positive_rate']:.4f}",
                f"{row['combined_score']:.4f}"
            ])
        
        # Create table
        table = ax.table(
            cellText=data,
            colLabels=['Window\nSize', 'Threshold', 'Accuracy\n(↑)', 'MDR\n(↓)', 'FPR\n(↓)', 'Combined\nScore'],
            loc='center',
            cellLoc='center',
            colWidths=[0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add color coding based on performance
        norm_acc = plt.Normalize(top_df['accuracy'].min(), top_df['accuracy'].max())
        norm_mdr = plt.Normalize(top_df['multiple_detection_rate'].max(), top_df['multiple_detection_rate'].min())
        norm_fpr = plt.Normalize(top_df['false_positive_rate'].max(), top_df['false_positive_rate'].min())
        
        for i in range(len(data)):
            # Color accuracy (green gradient)
            acc_val = float(data[i][2])
            acc_color = plt.cm.Greens(norm_acc(acc_val))
            table[(i+1, 2)].set_facecolor(acc_color)
            
            # Color MDR (red gradient - reversed so lower is better)
            mdr_val = float(data[i][3])
            mdr_color = plt.cm.Reds(1 - norm_mdr(mdr_val))
            table[(i+1, 3)].set_facecolor(mdr_color)
            
            # Color FPR (purple gradient - reversed so lower is better)
            fpr_val = float(data[i][4])
            fpr_color = plt.cm.Purples(1 - norm_fpr(fpr_val))
            table[(i+1, 4)].set_facecolor(fpr_color)
        
        # Add title
        plt.title('Top Performance Configurations with Visual Indicators', fontsize=14, pad=20)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()