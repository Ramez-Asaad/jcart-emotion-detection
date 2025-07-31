# JCart Emotion Detection System

## Project Overview
A real-time emotion detection and driver monitoring system for automotive applications, developed over 4 weeks of research. This system uses computer vision and machine learning to analyze facial expressions, detect drowsiness indicators, and monitor passenger states in vehicles.

## 🎯 Key Features
- **Multi-face Detection**: Supports up to 5 faces simultaneously using MediaPipe
- **Emotion Recognition**: Detects 6 emotions (happy, sad, angry, neutral, fear, surprised) using DeepFace
- **Drowsiness Detection**: Eye aspect ratio (EAR) monitoring for blink detection
- **Yawning Detection**: Mouth aspect ratio (MAR) analysis
- **Head Movement Tracking**: Nose tip tracking for nodding detection
- **Real-time Processing**: Live camera feed analysis with logging capabilities
- **Performance Analytics**: Comprehensive analysis tools with accuracy metrics

## 📊 System Performance
- **Person 1 Detection**: 85% accuracy (142 test samples)
- **Person 2 Detection**: 84% accuracy (142 test samples)
- **Multi-modal Detection**: Combines emotion, drowsiness, and movement analysis

## 🏗️ Project Structure

### `core_system/` - Main Detection Engine
Contains the primary face detection and emotion analysis modules.

### `data_collection/` - Recording & Logging Tools
Scripts for collecting training/testing data with synchronized video and emotion logs.

### `Datasets/` - Experimental Data
Organized test sessions with video recordings and corresponding emotion logs.

### `model_analysis/` - Generated Reports
Performance metrics, confusion matrices, and visualization outputs.

### `Documentation/` - Project Documentation
Academic reports, research papers, and technical documentation.

### `testing_utilities/` - Archive
Contains legacy scripts and utilities for testing and validation.

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dependencies
- OpenCV (computer vision)
- MediaPipe (face landmark detection)
- DeepFace (emotion analysis)
- NumPy (numerical operations)
- Scipy (scientific computing)
- Imutils (image processing utilities)

### Basic Usage
1. **Live Detection**: Run `core_system/face_detection.py`
2. **Data Collection**: Use `data_collection/side_by_side_with_log.py`
3. **Analysis**: Execute `core_system/advanced_log_analysis.py`

## 📈 Research Timeline
- **week 1-2**: Initial face detection and basic emotion recognition
- **week 3**: Advanced drowsiness detection and multi-face support
- **week 4**: Performance optimization and comprehensive analysis tools

## 🔬 Research Contributions
1. **Custom Emotion Filtering**: Implemented rules to reduce false positives in emotion detection
2. **Multi-modal Monitoring**: Combined emotion, drowsiness, and movement analysis
3. **Automotive Optimization**: Specialized preprocessing for in-vehicle camera conditions
4. **Performance Benchmarking**: Comprehensive evaluation framework with multiple metrics

## 🤝 For Future Researchers
- Start with `Documentation/GETTING_STARTED.md`
- Review `model_analysis/` for baseline performance
- Check `Datasets/` for example datasets
- Use `testing_utilities/` for evaluating improvements

## 📝 License
Academic research project - Please cite if used in publications.

## 👥 Contributors
- Primary Research: Ramez Asaad, Rana Houssam (2021-2025)
- Institution: Alamein International University & JMU

---
*Last updated: July 31, 2025*