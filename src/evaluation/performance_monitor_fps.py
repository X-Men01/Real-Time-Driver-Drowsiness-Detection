from dataclasses import dataclass, field
from typing import Dict, List
from time import perf_counter
from statistics import mean
from collections import deque



from detection.camera_module import CameraModule
from detection.face_detection import FaceDetection
from detection.feature_extraction import FeatureExtraction
from detection.state_classification import StateClassification
from detection.decision_logic import DecisionLogic
from detection.alarm_system import AlarmSystem
from detection.config import Config

@dataclass
class ComponentTiming:
    """Timing information for a specific component."""
    name: str
    times: deque = field(default_factory=lambda: deque(maxlen=30))
    
    def add_time(self, time_ms: float) -> None:
        """Add a new timing measurement."""
        self.times.append(time_ms)
    
    @property
    def average_time(self) -> float:
        """Get average processing time in milliseconds."""
        return mean(self.times) if self.times else 0.0

class PerformanceMonitor:
    """Monitor and track system performance metrics."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.components: Dict[str, ComponentTiming] = {}
        self.frame_times = deque(maxlen=window_size)
        self._last_frame_time = perf_counter()
    
    def start_frame(self) -> None:
        """Mark the start of a new frame."""
        current_time = perf_counter()
        frame_time = (current_time - self._last_frame_time) * 1000  # Convert to ms
        self.frame_times.append(frame_time)
        self._last_frame_time = current_time
    
    @property
    def fps(self) -> float:
        """Calculate current FPS."""
        if not self.frame_times:
            return 0.0
        return 1000 / mean(self.frame_times)  # Convert ms to FPS
    
    def add_component(self, name: str) -> None:
        """Add a new component to monitor."""
        self.components[name] = ComponentTiming(name, deque(maxlen=self.window_size))
    
    def start_component(self, name: str) -> float:
        """Start timing a component."""
        if name not in self.components:
            self.add_component(name)
        return perf_counter()
    
    def stop_component(self, name: str, start_time: float) -> None:
        """Stop timing a component and record its duration."""
        duration = (perf_counter() - start_time) * 1000  # Convert to ms
        self.components[name].add_time(duration)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        metrics = {
            'fps': self.fps,
            'frame_time': mean(self.frame_times) if self.frame_times else 0.0
        }
        
        # Add component metrics
        for name, component in self.components.items():
            metrics[f'{name}_time'] = component.average_time
            
        return metrics
    
    def print_metrics(self) -> None:
        """Print current performance metrics."""
        metrics = self.get_metrics()
        print("\nPerformance Metrics:")
        print(f"FPS: {metrics['fps']:.1f}")
        print(f"Frame Time: {metrics['frame_time']:.1f}ms")
        for name, component in self.components.items():
            print(f"{name}: {component.average_time:.1f}ms")


def main():
    config = Config()
    config.validate()
    
    
    camera = CameraModule(config)
    face_detector = FaceDetection(config)
    feature_extractor = FeatureExtraction(config)
    
    state_classifier = StateClassification(config)
    decision_logic = DecisionLogic(config)
    alarm_system = AlarmSystem(config)
    
    # tracker = DrowsinessTracker(config)
    monitor = PerformanceMonitor(100)
    
    
    try:
        while True:
            # Start frame timing
            monitor.start_frame()
            
            # Capture frame
            start_time = monitor.start_component('camera')
            frame_result = camera.capture_frame()
            monitor.stop_component('camera', start_time)
            
            if frame_result is None:
                continue
            
            # Detect face
            start_time = monitor.start_component('face_detection')
            face_result = face_detector.detect_face(frame_result)
            monitor.stop_component('face_detection', start_time)
            
            if face_result.success:
                # Process features
                start_time = monitor.start_component('feature_extraction')
                features = feature_extractor.process_face(face_result.face)
                monitor.stop_component('feature_extraction', start_time)
                
                if features.success:
                    # Classify states
                    start_time = monitor.start_component('classification')
                    states = state_classifier.process_features(features)
                    monitor.stop_component('classification', start_time)
                    
                    if states.success:
                        # Make decision
                        start_time = monitor.start_component('decision')
                        decision = decision_logic.determine_drowsiness(states)
                        monitor.stop_component('decision', start_time)
            
            # Print performance metrics every 30 frames
            if len(monitor.frame_times) % 30 == 0:
                monitor.print_metrics()
                
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        camera.release()
        

if __name__ == "__main__":
    main()
