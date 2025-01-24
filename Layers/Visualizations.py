import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# Function to create a timestamped folder
def create_timestamped_folder(base_dir="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# Function to plot and save the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()

# Function to plot and save class-wise accuracy
def plot_class_accuracy(y_true, y_pred, classes, save_path):
    class_accuracies = []
    for cls in classes:
        cls_indices = y_true == cls
        cls_correct = np.sum(y_true[cls_indices] == y_pred[cls_indices])
        cls_total = np.sum(cls_indices)
        accuracy = cls_correct / cls_total if cls_total > 0 else 0
        class_accuracies.append(accuracy)

    plt.figure(figsize=(8, 6))
    plt.bar(classes, class_accuracies, color="skyblue")
    plt.title("Class-wise Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.xticks(classes)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "class_accuracy.png"))
    plt.close()

# Function to save classification report as text
def save_classification_report(y_true, y_pred, classes, save_path):
    report = classification_report(y_true, y_pred, labels=classes, target_names=[str(cls) for cls in classes])
    report_path = os.path.join(save_path, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
        
# Function to plot and save classification report metrics
def plot_classification_report(y_true, y_pred, classes, save_path):
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)

    metrics = ['precision', 'recall', 'f1-score']
    values = {metric: [report[str(cls)][metric] for cls in classes] for metric in metrics}

    x = np.arange(len(classes))
    width = 0.2

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, values[metric], width, label=metric)

    plt.title("Classification Report Metrics")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.xticks(x + width, classes)
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "classification_report_metrics.png"))
    plt.close()

# Main pipeline function
def generate_visualizations(y_true, y_pred, classes = np.arange(1, 11)):
    # Create a timestamped folder
    save_path = create_timestamped_folder()

    # Generate visualizations and save them
    plot_confusion_matrix(y_true, y_pred, classes, save_path)
    plot_class_accuracy(y_true, y_pred, classes, save_path)
    save_classification_report(y_true, y_pred, classes, save_path)
    plot_classification_report(y_true, y_pred, classes, save_path)

    print(f"Visualizations and reports saved to: {save_path}")

# Example usage
if __name__ == "__main__":
    # Example classifications (replace with real data)
    y_true = np.random.randint(1, 11, 1000)
    y_pred = np.random.randint(1, 11, 1000)
    classes = np.arange(1, 11)  # Assuming classes are 1 to 5

    generate_visualizations(y_true, y_pred, classes)