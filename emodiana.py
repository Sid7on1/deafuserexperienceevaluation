import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionEvaluation:
    """
    Class for emotion evaluation using EMODIANA methodology.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize EmotionEvaluation instance.
        
        Args:
        config (Dict): Configuration dictionary.
        """
        self.config = config
        self.emotions = ['happiness', 'sadness', 'fear', 'anger', 'surprise']
        self.intensity_thresholds = {
            'happiness': 0.5,
            'sadness': 0.3,
            'fear': 0.4,
            'anger': 0.6,
            'surprise': 0.7
        }
        
    def evaluate_emotion(self, data: Dict) -> Tuple:
        """
        Evaluate emotion based on intensity and velocity.
        
        Args:
        data (Dict): Data dictionary containing intensity and velocity values.
        
        Returns:
        Tuple: Emotion and intensity values.
        """
        intensity = data['intensity']
        velocity = data['velocity']
        
        # Calculate velocity-threshold
        velocity_threshold = self.config['velocity_threshold']
        
        # Calculate Flow Theory score
        flow_score = self.calculate_flow_score(intensity, velocity, velocity_threshold)
        
        # Determine emotion based on intensity and flow score
        emotion = self.determine_emotion(intensity, flow_score)
        
        return emotion, intensity
    
    def calculate_flow_score(self, intensity: float, velocity: float, velocity_threshold: float) -> float:
        """
        Calculate Flow Theory score.
        
        Args:
        intensity (float): Intensity value.
        velocity (float): Velocity value.
        velocity_threshold (float): Velocity threshold value.
        
        Returns:
        float: Flow Theory score.
        """
        if velocity >= velocity_threshold:
            flow_score = intensity / (1 + np.exp(-velocity + velocity_threshold))
        else:
            flow_score = 0
        
        return flow_score
    
    def determine_emotion(self, intensity: float, flow_score: float) -> str:
        """
        Determine emotion based on intensity and flow score.
        
        Args:
        intensity (float): Intensity value.
        flow_score (float): Flow Theory score.
        
        Returns:
        str: Emotion.
        """
        if intensity >= self.intensity_thresholds['happiness']:
            return 'happiness'
        elif intensity >= self.intensity_thresholds['sadness']:
            return 'sadness'
        elif intensity >= self.intensity_thresholds['fear']:
            return 'fear'
        elif intensity >= self.intensity_thresholds['anger']:
            return 'anger'
        elif intensity >= self.intensity_thresholds['surprise']:
            return 'surprise'
        else:
            return 'neutral'
    
    def plot_emotion(self, data: Dict):
        """
        Plot emotion over time.
        
        Args:
        data (Dict): Data dictionary containing emotion and time values.
        """
        emotions = data['emotions']
        times = data['times']
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=times, y=emotions)
        plt.xlabel('Time')
        plt.ylabel('Emotion')
        plt.title('Emotion Over Time')
        plt.show()


class IntensityAssessment:
    """
    Class for intensity assessment using EMODIANA methodology.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize IntensityAssessment instance.
        
        Args:
        config (Dict): Configuration dictionary.
        """
        self.config = config
        self.intensity_thresholds = {
            'happiness': 0.5,
            'sadness': 0.3,
            'fear': 0.4,
            'anger': 0.6,
            'surprise': 0.7
        }
        
    def assess_intensity(self, data: Dict) -> Tuple:
        """
        Assess intensity based on emotion and velocity.
        
        Args:
        data (Dict): Data dictionary containing emotion and velocity values.
        
        Returns:
        Tuple: Intensity and velocity values.
        """
        emotion = data['emotion']
        velocity = data['velocity']
        
        # Determine intensity based on emotion and velocity
        intensity = self.determine_intensity(emotion, velocity)
        
        return intensity, velocity
    
    def determine_intensity(self, emotion: str, velocity: float) -> float:
        """
        Determine intensity based on emotion and velocity.
        
        Args:
        emotion (str): Emotion.
        velocity (float): Velocity value.
        
        Returns:
        float: Intensity value.
        """
        if emotion == 'happiness':
            intensity = 1 - np.exp(-velocity)
        elif emotion == 'sadness':
            intensity = np.exp(-velocity)
        elif emotion == 'fear':
            intensity = 1 - np.exp(-velocity)
        elif emotion == 'anger':
            intensity = np.exp(-velocity)
        elif emotion == 'surprise':
            intensity = 1 - np.exp(-velocity)
        else:
            intensity = 0
        
        return intensity
    
    def plot_intensity(self, data: Dict):
        """
        Plot intensity over time.
        
        Args:
        data (Dict): Data dictionary containing intensity and time values.
        """
        intensities = data['intensities']
        times = data['times']
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=times, y=intensities)
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.title('Intensity Over Time')
        plt.show()


class DataAnalysis:
    """
    Class for data analysis using EMODIANA methodology.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataAnalysis instance.
        
        Args:
        config (Dict): Configuration dictionary.
        """
        self.config = config
        
    def analyze_data(self, data: Dict) -> Dict:
        """
        Analyze data based on emotion and intensity.
        
        Args:
        data (Dict): Data dictionary containing emotion and intensity values.
        
        Returns:
        Dict: Analyzed data dictionary.
        """
        emotions = data['emotions']
        intensities = data['intensities']
        
        # Calculate mean and standard deviation of emotions and intensities
        mean_emotions = np.mean(emotions)
        std_emotions = np.std(emotions)
        mean_intensities = np.mean(intensities)
        std_intensities = np.std(intensities)
        
        # Calculate correlation between emotions and intensities
        correlation = np.corrcoef(emotions, intensities)[0, 1]
        
        # Calculate mean squared error between emotions and intensities
        mse = mean_squared_error(emotions, intensities)
        
        # Create analyzed data dictionary
        analyzed_data = {
            'mean_emotions': mean_emotions,
            'std_emotions': std_emotions,
            'mean_intensities': mean_intensities,
            'std_intensities': std_intensities,
            'correlation': correlation,
            'mse': mse
        }
        
        return analyzed_data
    
    def plot_data(self, data: Dict):
        """
        Plot data based on emotion and intensity.
        
        Args:
        data (Dict): Data dictionary containing emotion and intensity values.
        """
        emotions = data['emotions']
        intensities = data['intensities']
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=emotions, y=intensities)
        plt.xlabel('Emotion')
        plt.ylabel('Intensity')
        plt.title('Emotion vs. Intensity')
        plt.show()


def main():
    # Load configuration
    config = {
        'velocity_threshold': 0.5
    }
    
    # Create EmotionEvaluation instance
    emotion_evaluation = EmotionEvaluation(config)
    
    # Create IntensityAssessment instance
    intensity_assessment = IntensityAssessment(config)
    
    # Create DataAnalysis instance
    data_analysis = DataAnalysis(config)
    
    # Simulate data
    data = {
        'emotions': [0.5, 0.3, 0.4, 0.6, 0.7],
        'intensities': [0.8, 0.9, 0.7, 0.6, 0.5],
        'velocity': [0.2, 0.3, 0.4, 0.5, 0.6]
    }
    
    # Evaluate emotion
    emotion, intensity = emotion_evaluation.evaluate_emotion(data)
    logger.info(f'Emotion: {emotion}, Intensity: {intensity}')
    
    # Assess intensity
    intensity, velocity = intensity_assessment.assess_intensity(data)
    logger.info(f'Intensity: {intensity}, Velocity: {velocity}')
    
    # Analyze data
    analyzed_data = data_analysis.analyze_data(data)
    logger.info(f'Mean Emotions: {analyzed_data["mean_emotions"]}, Standard Deviation Emotions: {analyzed_data["std_emotions"]}')
    logger.info(f'Mean Intensities: {analyzed_data["mean_intensities"]}, Standard Deviation Intensities: {analyzed_data["std_intensities"]}')
    logger.info(f'Correlation: {analyzed_data["correlation"]}, MSE: {analyzed_data["mse"]}')
    
    # Plot emotion
    emotion_evaluation.plot_emotion(data)
    
    # Plot intensity
    intensity_assessment.plot_intensity(data)
    
    # Plot data
    data_analysis.plot_data(data)


if __name__ == '__main__':
    main()