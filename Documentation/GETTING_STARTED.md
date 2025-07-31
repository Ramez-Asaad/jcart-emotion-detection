# Getting Started with JCart Emotion Detection System

Welcome to the JCart Emotion Detection System! This guide will help new students understand the project structure, set up the development environment, and start contributing to the research.

## 📋 Prerequisites

### Required Knowledge
- **Python Programming**: Intermediate level (functions, classes, libraries)
- **Computer Vision Basics**: Understanding of image processing concepts
- **Machine Learning Fundamentals**: Basic knowledge of ML workflows
- **Git Version Control**: Basic git commands and workflow

### Hardware Requirements
- **Camera**: Webcam or ZED stereo camera for testing
- **Computer**: Windows/Linux/macOS with at least 8GB RAM
- **GPU**: Optional but recommended for faster processing

## 🚀 Environment Setup

### 1. Clone the Repository
```bash
git clone [repository-url]
cd jcart-emotion-detection
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. ZED Camera Setup (Optional)
If using ZED stereo camera:
1. Download [ZED SDK](https://www.stereolabs.com/developers/release/)
2. Install for your platform
3. Verify camera recognition

### 4. Verify Installation
```bash
python -c "import cv2, mediapipe, deepface; print('All dependencies installed successfully!')"
```

## 🏗️ Understanding the Project Structure

### Core Components
```
jcart-emotion-detection/
├── core_system/           # Main detection algorithms
├── data_collection/       # Data recording scripts
├── Datasets/             # Test data and recordings
├── model_analysis/       # Performance metrics
├── Documentation/        # Project docs (you are here!)
└── testing_utilities/    # Development tools
```

### Key Scripts Overview
1. **`core_system/face_detection_passenger.py`** - Main detection engine (most important)
2. **`core_system/compare_crop_enhance.py`** - Image enhancement core
3. **`data_collection/side_by_side_with_log.py`** - Data collection tool
4. **`core_system/advanced_log_analysis.py`** - Performance analysis

## 🔄 Development Workflow

### Phase 1: Understanding (Week 1)
1. **Read the Documentation**
   - Review this file completely
   - Study the main README.md
   - Examine script documentation headers

2. **Run Basic Tests**
   ```bash
   # Test real-time detection
   python core_system/face_detection_passenger.py
   
   # Test enhancement pipeline
   python testing_utilities/run_compare_on_camera.py
   ```

3. **Explore the Code**
   - Start with `face_detection_passenger.py` (main detection logic)
   - Study `compare_crop_enhance.py` (image processing)
   - Review existing datasets in `Datasets/`

### Phase 2: Data Collection (Week 2)
1. **Generate New Datasets**
   ```bash
   python data_collection/side_by_side_with_log.py
   ```

2. **Manual Annotation**
   - Review recorded videos
   - Add ground truth labels to log files
   - Follow existing annotation format

3. **Data Organization**
   - Save files with consistent naming
   - Document session conditions (lighting, participants, etc.)

### Phase 3: Analysis & Improvement (Week 3-4)
1. **Performance Analysis**
   ```bash
   python core_system/advanced_log_analysis.py
   ```

2. **Algorithm Improvements**
   - Modify detection parameters
   - Implement new emotion correction rules
   - Test enhancement techniques

3. **Validation**
   - Compare before/after performance
   - Generate new performance visualizations

## 🎯 Common Tasks for New Students

### Task 1: Data Collection Session
1. Set up camera (index 4 for ZED, 0 for webcam)
2. Run `side_by_side_with_log.py` with different participants
3. Record 5-10 minute sessions with varied emotions
4. Manually annotate the log files post-recording

### Task 2: Algorithm Testing
1. Modify parameters in `face_detection_passenger.py`:
   - `EAR_THRESHOLD` (drowsiness sensitivity)
   - `MAR_THRESHOLD` (yawning detection)
   - Emotion correction rules
2. Test changes with existing datasets
3. Measure performance impact

### Task 3: Enhancement Validation
1. Use `run_compare_on_camera.py` to test image processing
2. Adjust `smart_enhance()` intensity parameters
3. Compare detection accuracy with/without enhancement

## 📊 Performance Evaluation

### Current Baseline Metrics (based on just 3 tests)
- **Overall Accuracy**: 84-85%
- **Happy Detection**: 91% accuracy
- **Sad Detection**: 96% accuracy
- **Angry Detection**: 100% accuracy
- **Neutral Detection**: 76% accuracy

### Improvement Goals
- Increase neutral emotion accuracy (currently lowest)
- Reduce false positives in challenging lighting
- Improve multi-face detection consistency

## 🔧 Troubleshooting

### Common Issues
1. **Camera not detected**: Check camera index in scripts
2. **Import errors**: Verify all dependencies are installed
3. **Performance issues**: Check available RAM and close other applications
4. **ZED camera issues**: Ensure ZED SDK is properly installed

### Getting Help
1. **Check script documentation**: Each file has detailed headers
2. **Review existing datasets**: Learn from successful data collection
3. **Study performance visualizations**: Understand current strengths/weaknesses
4. **Ask senior team members**: Document questions for discussion

## 📈 Research Methodology

### Data Collection Best Practices
- **Consistent Environment**: Same lighting, camera position
- **Diverse Participants**: Various ages, ethnicities, glasses/no glasses
- **Emotion Variety**: Ensure all emotions are represented
- **Quality Control**: Review recordings before annotation

### Code Development Guidelines
- **Document Changes**: Add comments explaining modifications
- **Test Thoroughly**: Validate on multiple datasets
- **Preserve Originals**: Keep backup copies of working code
- **Performance Tracking**: Measure impact of all changes

## 🎓 Learning Resources

### Computer Vision
- OpenCV documentation and tutorials
- MediaPipe face detection guides
- Image processing fundamentals

### Machine Learning
- DeepFace library documentation
- Emotion recognition research papers
- Performance evaluation metrics

### Automotive Applications
- Driver monitoring system standards
- In-vehicle camera placement guidelines
- Safety system integration requirements

## 📝 Next Steps

### After Setup Completion
1. **Run the entire pipeline** end-to-end
2. **Collect your first dataset** (start small - 2-3 minutes)
3. **Analyze the results** using existing tools
4. **Identify improvement opportunities**
5. **Propose specific research directions**

### Research Ideas for New Students
- Lighting condition optimization
- Real-time performance improvements
- Additional emotion categories
- Integration with vehicle systems
- Mobile/embedded deployment

## 🤝 Contributing Guidelines

### Before Making Changes
1. Understand the existing codebase
2. Document your planned changes
3. Test on small datasets first
4. Measure performance impact

### Code Quality Standards
- Follow existing naming conventions
- Add comprehensive comments
- Include error handling
- Document new functions

---

**Welcome to the team! This research has significant potential for improving automotive safety and passenger experience. Your contributions will build upon 4 weeks of intensive development and help push the boundaries of real-time emotion detection technology.**

*For questions or clarifications, refer to the main README.md or consult with project supervisors.*
