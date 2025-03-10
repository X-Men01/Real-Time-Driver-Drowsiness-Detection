from evaluation.drowsiness_system_analyzer import DrowsinessSystemAnalyzer

def test_initialization():
    try:
        # Test with valid directory
        analyzer = DrowsinessSystemAnalyzer("logs/video_evaluation_window_theshold_search")
        print("✅ Successfully initialized with valid directory")
        
        # Test with invalid directory (should raise an error)
        try:
            invalid_analyzer = DrowsinessSystemAnalyzer("nonexistent_directory")
            print("❌ Failed: Should have raised an error for invalid directory")
        except ValueError as e:
            print(f"✅ Correctly raised error for invalid directory: {e}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

def test_collect_data():
    analyzer = DrowsinessSystemAnalyzer("logs/video_evaluation_window_theshold_search")
    analyzer.collect_data()
    
    # Check if data was collected
    if len(analyzer.all_results) > 0:
        print(f"✅ Successfully collected data from {len(analyzer.all_results)} directories")
        
        # Print first result as a sample
        if analyzer.all_results:
            print("\nSample result:")
            sample = analyzer.all_results[0]
            for key, value in sample.items():
                if key != 'config':  # Skip printing full config for brevity
                    print(f"  {key}: {value}")
            if 'config' in sample:
                print("  config: {...}")
    else:
        print("❌ Failed to collect any data")

def test_create_dataframe():
    analyzer = DrowsinessSystemAnalyzer("logs/video_evaluation_window_theshold_search")
    analyzer.collect_data()
    analyzer.create_metrics_dataframe()
    
    # Check if DataFrame was created
    if analyzer.metrics_df is not None:
        print(f"✅ Successfully created DataFrame with shape: {analyzer.metrics_df.shape}")
        
        # Print DataFrame info and sample
        print("\nDataFrame columns:")
        for col in analyzer.metrics_df.columns:
            print(f"  - {col}")
            
        print("\nSample of the DataFrame (first 3 rows):")
        print(analyzer.metrics_df[['window_size', 'threshold', 'accuracy', 
                                  'drowsy_detection_rate', 'false_positive_rate', 
                                  'average_detection_latency']].head(3))
            
    else:
        print("❌ Failed to create DataFrame")

def test_generate_heatmaps():
    analyzer = DrowsinessSystemAnalyzer("logs/video_evaluation_window_theshold_search")
    analyzer.collect_data()
    analyzer.create_metrics_dataframe()
    
    # Test generating heatmaps for a few key metrics
    test_metrics = ['accuracy', 'drowsy_detection_rate', 'false_positive_rate', 'average_detection_latency']
    analyzer.generate_heatmaps(metrics=test_metrics, save_dir="analysis_output/heatmaps")
    
    print("✅ Heatmap generation test complete")

def test_enhanced_heatmaps():
    analyzer = DrowsinessSystemAnalyzer("../../logs/video_evaluation_window_theshold_search")
    analyzer.collect_data()
    analyzer.create_metrics_dataframe()
    
    # Test generating enhanced heatmaps with optimization direction indicators
    test_metrics = ['accuracy', 'drowsy_detection_rate', 'false_positive_rate',]
    analyzer.generate_heatmaps(metrics=test_metrics, save_dir="/Users/ahmedalkhulayfi/Desktop/Real-Time-Driver-Drowsiness-Detection/analysis_output")
    
    print("✅ Enhanced heatmap generation complete")




if __name__ == "__main__":
   
    test_enhanced_heatmaps()
   