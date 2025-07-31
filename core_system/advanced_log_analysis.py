"""
=================================================================
ADVANCED LOG ANALYSIS MODULE
=================================================================

ABOUT:
This module provides comprehensive analysis and visualization of emotion detection
performance data collected from the JCart Driver/Passenger Monitoring System.

CORE FUNCTIONALITY:
1. Multi-file Log Processing: Automatically loads and consolidates multiple CSV log files
2. Performance Metrics: Generates confusion matrices, classification reports, and accuracy metrics
3. Error Analysis: Identifies misclassification patterns and temporal error distribution
4. Visualization Suite: Creates comprehensive charts for research documentation and debugging
5. Cross-person Comparison: Analyzes agreement between multiple subjects in the same session

INPUT DATA FORMAT:
Expected CSV columns: ['timestamp', 'emotion_1', 'emotion_2', 'true_emotion1', 'true_emotion2']
- emotion_1/2: Detected emotions from the face detection system
- true_emotion1/2: Ground truth labels (manually annotated or predetermined)
- timestamp: Recording time for temporal analysis

SUPPORTED EMOTIONS:
- Standard emotions: ['happy', 'sad', 'angry', 'neutral', 'fear']
- Special states: ['Head Nodding', 'none'] - converted to 'special' category

OUTPUT ARTIFACTS:
- Console reports with precision/recall/F1 scores
- Confusion matrices for each person
- Temporal analysis charts (emotion timelines, error distributions)
- Performance visualization (precision/recall/F1 bar charts)
- Cross-subject agreement heatmaps
- All visualizations saved to 'model_analysis/' directory

USAGE IN SYSTEM:
1. Data Collection Phase: Run after recording sessions with side_by_side_with_log.py
2. Model Evaluation: Assess detection accuracy across different subjects and conditions
3. Research Documentation: Generate publication-ready performance visualizations
4. System Debugging: Identify systematic errors and model weaknesses
5. Longitudinal Analysis: Compare performance across multiple recording sessions

RESEARCH APPLICATION:
This analysis framework was developed to evaluate the effectiveness of the emotion
detection pipeline in automotive monitoring scenarios, providing quantitative metrics
for research publications and system optimization.

Author: Ramez Asaad - summer 25
Last Updated: July 2025
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import groupby
from sklearn.metrics import confusion_matrix, classification_report
import glob
import os

# --- LOAD MULTIPLE LOG FILES ---
log_files = glob.glob("wed 1*.log") + glob.glob("wed 2*.log")
if not log_files:
    print("No log files found starting with 'wed 1' or 'wed 2'.")
    exit(1)
print(f"Found log files: {log_files}")

df_list = []
for f in log_files:
    try:
        df_file = pd.read_csv(f, on_bad_lines='skip')  # skips bad lines
        # Standardize columns: pad missing columns with None
        expected_cols = ['timestamp','emotion_1','emotion_2','true_emotion1','true_emotion2']
        for col in expected_cols:
            if col not in df_file.columns:
                df_file[col] = None
        df_file = df_file[expected_cols]
        df_list.append(df_file)
    except Exception as e:
        print(f"Warning: Could not load {f}: {e}")

if not df_list:
    print("No valid log data loaded.")
    exit(1)

df = pd.concat(df_list, ignore_index=True)

# Clean up 'none' and special states
emotions = ['happy', 'sad', 'angry', 'neutral', 'fear']
special_states = ['Head Nodding', 'none']

def clean_emotion(e):
    return e if e in emotions else 'special'

for col in ['emotion_1', 'emotion_2', 'true_emotion1', 'true_emotion2']:
    df[col+'_clean'] = df[col].apply(clean_emotion)

# Confusion matrix (per person)
def plot_confusion(true, pred, title):
    cm = confusion_matrix(true, pred, labels=emotions+['special'])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotions+['special'], yticklabels=emotions+['special'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion(df['true_emotion1_clean'], df['emotion_1_clean'], "Person 1 Confusion Matrix")
plot_confusion(df['true_emotion2_clean'], df['emotion_2_clean'], "Person 2 Confusion Matrix")

# Classification report
print("Person 1 Classification Report:")
print(classification_report(df['true_emotion1_clean'], df['emotion_1_clean'], labels=emotions+['special']))
print("Person 2 Classification Report:")
print(classification_report(df['true_emotion2_clean'], df['emotion_2_clean'], labels=emotions+['special']))

# Head Nodding and special state analysis
for person, col in [('Person 1', 'emotion_1'), ('Person 2', 'emotion_2')]:
    nods = df[df[col] == 'Head Nodding']
    print(f"{person} Head Nodding count: {len(nods)}")
    print(f"Timestamps: {nods['timestamp'].tolist()}")

# Optional: Save all plots to files
# plt.savefig("plotname.png")

# --- MODEL ACCURACY AND METRICS ---
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def print_metrics(true, pred, person):
    acc = accuracy_score(true, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, labels=emotions, average=None)
    print(f"\n{person} Overall Accuracy: {acc:.3f}")
    print(f"Per-emotion metrics:")
    for i, emo in enumerate(emotions):
        print(f"  {emo:8} | Precision: {precision[i]:.2f} | Recall: {recall[i]:.2f} | F1: {f1[i]:.2f}")
    # Bar chart of per-emotion accuracy
    correct = [sum((np.array(true)==emo) & (np.array(pred)==emo)) for emo in emotions]
    total = [sum(np.array(true)==emo) for emo in emotions]
    accs = [c/t if t>0 else 0 for c,t in zip(correct, total)]
    plt.figure(figsize=(7,4))
    plt.bar(emotions, accs)
    plt.title(f"{person} Per-Emotion Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.show()
    # Summary table
    print("\nSummary Table:")
    print(pd.DataFrame({
        'Emotion': emotions,
        'Correct': correct,
        'Total': total,
        'Accuracy': accs
    }))

print_metrics(df['true_emotion1_clean'], df['emotion_1_clean'], "Person 1")
print_metrics(df['true_emotion2_clean'], df['emotion_2_clean'], "Person 2")

# Combined accuracy (if you want to see overall performance)
all_true = pd.concat([df['true_emotion1_clean'], df['true_emotion2_clean']])
all_pred = pd.concat([df['emotion_1_clean'], df['emotion_2_clean']])
print_metrics(all_true, all_pred, "Combined (Person 1 + 2)")

# --- ERROR ANALYSIS AND VISUALIZATIONS ---
analysis_dir = "model_analysis"
os.makedirs(analysis_dir, exist_ok=True)

# Error breakdown by emotion (bar chart)
def plot_error_breakdown(true, pred, person):
    errors = [t for t, p in zip(true, pred) if t != p and t in emotions]
    error_counts = pd.Series(errors).value_counts().reindex(emotions, fill_value=0)
    plt.figure(figsize=(7,4))
    error_counts.plot(kind='bar', color='red')
    plt.title(f"{person} Error Breakdown by Emotion")
    plt.ylabel("Misclassifications")
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/{person}_error_breakdown.png")
    plt.close()

plot_error_breakdown(df['true_emotion1_clean'], df['emotion_1_clean'], "Person 1")
plot_error_breakdown(df['true_emotion2_clean'], df['emotion_2_clean'], "Person 2")

# True vs. predicted emotion over time (line plot)
def plot_emotion_timeline(true, pred, person):
    plt.figure(figsize=(12,4))
    plt.plot(true.values, label='True', marker='o', linestyle='-', alpha=0.7)
    plt.plot(pred.values, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.title(f"{person} True vs. Predicted Emotion Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Emotion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/{person}_emotion_timeline.png")
    plt.close()

plot_emotion_timeline(df['true_emotion1_clean'], df['emotion_1_clean'], "Person 1")
plot_emotion_timeline(df['true_emotion2_clean'], df['emotion_2_clean'], "Person 2")

# Misclassification timeline (scatter plot)
def plot_misclassification_timeline(true, pred, person):
    mis_idx = [i for i, (t, p) in enumerate(zip(true, pred)) if t != p]
    plt.figure(figsize=(12,2))
    plt.scatter(mis_idx, [1]*len(mis_idx), color='red', marker='x')
    plt.title(f"{person} Misclassification Timeline")
    plt.xlabel("Frame Index")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/{person}_misclassification_timeline.png")
    plt.close()

plot_misclassification_timeline(df['true_emotion1_clean'], df['emotion_1_clean'], "Person 1")
plot_misclassification_timeline(df['true_emotion2_clean'], df['emotion_2_clean'], "Person 2")

# Per-emotion precision/recall/F1 (bar chart)
def plot_prf_bars(true, pred, person):
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, labels=emotions, average=None)
    x = np.arange(len(emotions))
    plt.figure(figsize=(8,5))
    plt.bar(x-0.2, precision, width=0.2, label='Precision')
    plt.bar(x, recall, width=0.2, label='Recall')
    plt.bar(x+0.2, f1, width=0.2, label='F1')
    plt.xticks(x, emotions)
    plt.title(f"{person} Per-Emotion Precision/Recall/F1")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/{person}_prf_bars.png")
    plt.close()

plot_prf_bars(df['true_emotion1_clean'], df['emotion_1_clean'], "Person 1")
plot_prf_bars(df['true_emotion2_clean'], df['emotion_2_clean'], "Person 2")

# Distribution of detected emotions (bar chart)
def plot_detected_distribution(pred, person):
    counts = pd.Series(pred).value_counts().reindex(emotions, fill_value=0)
    plt.figure(figsize=(7,4))
    counts.plot(kind='bar', color='blue')
    plt.title(f"{person} Distribution of Detected Emotions")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/{person}_detected_distribution.png")
    plt.close()

plot_detected_distribution(df['emotion_1_clean'], "Person 1")
plot_detected_distribution(df['emotion_2_clean'], "Person 2")

# Special state analysis (bar chart)
def plot_special_state_analysis(df, person, col):
    specials = df[col].value_counts().reindex(special_states, fill_value=0)
    plt.figure(figsize=(5,4))
    specials.plot(kind='bar', color='purple')
    plt.title(f"{person} Special State Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/{person}_special_state_frequency.png")
    plt.close()

plot_special_state_analysis(df, "Person 1", 'emotion_1')
plot_special_state_analysis(df, "Person 2", 'emotion_2')

# Heatmap of prediction agreement between persons
def plot_agreement_heatmap(df):
    agreement = pd.crosstab(df['emotion_1_clean'], df['emotion_2_clean'])
    plt.figure(figsize=(7,6))
    sns.heatmap(agreement, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("Prediction Agreement Between Persons")
    plt.xlabel("Person 2 Emotion")
    plt.ylabel("Person 1 Emotion")
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/prediction_agreement_heatmap.png")
    plt.close()

plot_agreement_heatmap(df)

print("Advanced log analysis complete.")