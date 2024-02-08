import keras
import data_pca
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""x_train_augs = data.x_train_augs
y_train_augs = data.y_train_augs
x_val_closed = data.x_val_closed
y_val_closed = data.y_val_closed
x_validation = data.val_data_imageDirectory
y_validation = data.val_data_answerDirectory
x_val_4 = data.rec_4_val_imageDirectory
y_val_4 = data.rec_4_valanswerDirectory"""

x_train = data_pca.x_train_ica #x_train
y_train = data_pca.y_train 
x_test = data_pca.x_test_ica #x_test
y_test = data_pca.y_test
x_val = data_pca.x_val_ica #val_data_imageDirectory
y_val = data_pca.val_data_answerDirectory
"""x_1 = data.data_2_1_imageDirectory
y_1 = data.data_2_1_answerDirectory
x_2 = data.data_2_2_imageDirectory
y_2 = data.data_2_2_answerDirectory
x_3 = data.data_2_3_imageDirectory
y_3 = data.data_2_3_answerDirectory
x_4 = data.data_2_4_imageDirectory
y_4 = data.data_2_4_answerDirectory
x_5 = data.data_2_5_imageDirectory
y_5 = data.data_2_5_answerDirectory"""

"""x_3 = data.data_3_imageDirectory
y_3 = data.data_3_answerDirectory
x_4 = data.data_4_imageDirectory
y_4 = data.data_4_answerDirectory
"""
model_path = 'checkpoints'
model = keras.models.load_model(model_path)

evaluation = model.evaluate(x_test, y_test)

# Print the overall accuracy of the model
accuracy = evaluation[1]  # Assuming accuracy is the second metric
print("Overall Accuracy:", accuracy)

expected_accuracy = 1/3
total_samples = 119
accurate_predictions = (round(accuracy*total_samples))  # or any number between 45 and 90

# Calculate the p-value using the binomial test
p_value = 1 - stats.binom.cdf(accurate_predictions - 1, total_samples, expected_accuracy)

print("testing p-value:", p_value)

#############

val_evaluation = model.evaluate(x_val, y_val)

# Print the overall accuracy of the model
accuracy = val_evaluation[1]  # Assuming accuracy is the second metric
print("Overall Validation Accuracy:", accuracy)

expected_accuracy = 1/3
total_samples = 119
accurate_predictions = (round(accuracy*total_samples))  # or any number between 45 and 90

# Calculate the p-value using the binomial test
p_value = 1 - stats.binom.cdf(accurate_predictions - 1, total_samples, expected_accuracy)

print("validation p-value:", p_value)


"""#############

val_evaluation = model.evaluate(x_1, y_1)

# Print the overall accuracy of the model
accuracy = val_evaluation[1]  # Assuming accuracy is the second metric
print("Overall Validation Accuracy:", accuracy)

expected_accuracy = 1/3
total_samples = 174
accurate_predictions = (round(accuracy*total_samples))  # or any number between 45 and 90

# Calculate the p-value using the binomial test
p_value = 1 - stats.binom.cdf(accurate_predictions - 1, total_samples, expected_accuracy)

print("validation p-value:", p_value)

#############

val_evaluation = model.evaluate(x_2, y_2)

# Print the overall accuracy of the model
accuracy = val_evaluation[1]  # Assuming accuracy is the second metric
print("Overall Validation Accuracy:", accuracy)

expected_accuracy = 1/3
total_samples = 174
accurate_predictions = (round(accuracy*total_samples))  # or any number between 45 and 90

# Calculate the p-value using the binomial test
p_value = 1 - stats.binom.cdf(accurate_predictions - 1, total_samples, expected_accuracy)

print("validation p-value:", p_value)
#############

val_evaluation = model.evaluate(x_3, y_3)

# Print the overall accuracy of the model
accuracy = val_evaluation[1]  # Assuming accuracy is the second metric
print("Overall Validation Accuracy:", accuracy)

expected_accuracy = 1/3
total_samples = 174
accurate_predictions = (round(accuracy*total_samples))  # or any number between 45 and 90

# Calculate the p-value using the binomial test
p_value = 1 - stats.binom.cdf(accurate_predictions - 1, total_samples, expected_accuracy)

print("validation p-value:", p_value)

#############

val_evaluation = model.evaluate(x_4, y_4)

# Print the overall accuracy of the model
accuracy = val_evaluation[1]  # Assuming accuracy is the second metric
print("Overall Validation Accuracy:", accuracy)

expected_accuracy = 1/3
total_samples = 174
accurate_predictions = (round(accuracy*total_samples))  # or any number between 45 and 90

# Calculate the p-value using the binomial test
p_value = 1 - stats.binom.cdf(accurate_predictions - 1, total_samples, expected_accuracy)

print("validation p-value:", p_value)

#############

val_evaluation = model.evaluate(x_5, y_5)

# Print the overall accuracy of the model
accuracy = val_evaluation[1]  # Assuming accuracy is the second metric
print("Overall Validation Accuracy:", accuracy)

expected_accuracy = 1/3
total_samples = 174
accurate_predictions = (round(accuracy*total_samples))  # or any number between 45 and 90

# Calculate the p-value using the binomial test
p_value = 1 - stats.binom.cdf(accurate_predictions - 1, total_samples, expected_accuracy)

print("validation p-value:", p_value)"""

"""#############

val_3_evaluation = model.evaluate(x_3, y_3)

# Print the overall accuracy of the model
accuracy = val_3_evaluation[1]  # Assuming accuracy is the second metric
print("Overall Validation 3 Accuracy:", accuracy)

expected_accuracy = 1/3
total_samples = 119
accurate_predictions = (round(accuracy*total_samples))  # or any number between 45 and 90

# Calculate the p-value using the binomial test
p_value = 1 - stats.binom.cdf(accurate_predictions - 1, total_samples, expected_accuracy)

print("validation 3 p-value:", p_value)

#############

val_4_evaluation = model.evaluate(x_4, y_4)

# Print the overall accuracy of the model
accuracy = val_4_evaluation[1]  # Assuming accuracy is the second metric
print("Overall Validation 4 Accuracy:", accuracy)

expected_accuracy = 1/3
total_samples = 119
accurate_predictions = (round(accuracy*total_samples))  # or any number between 45 and 90

# Calculate the p-value using the binomial test
p_value = 1 - stats.binom.cdf(accurate_predictions - 1, total_samples, expected_accuracy)

print("validation 4 p-value:", p_value)

print(model.summary())"""

##########

y_pred = model.predict(x_test)

# Convert predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert true labels to class labels
y_true_labels = np.argmax(y_test, axis=1)

# Calculate the number of correct predictions for each class
class_counts = np.bincount(y_pred_labels[y_pred_labels == y_true_labels])

# Display the number of correct predictions for each class
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count}/{np.sum(y_true_labels == i)}")

confusion_mat = confusion_matrix(y_true_labels, y_pred_labels)

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_mat)
class_labels = ['A', 'B', 'C']

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix from Testing Dataset")
plt.show()

##############

y_pred_val = model.predict(x_val)

# Convert predicted probabilities to class labels
y_pred_val_labels = np.argmax(y_pred_val, axis=1)

# Convert true labels to class labels
y_true_val_labels = np.argmax(y_val, axis=1)

# Calculate the number of correct predictions for each class
class_counts = np.bincount(y_pred_val_labels[y_pred_val_labels == y_true_val_labels])

# Display the number of correct predictions for each class
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count}/{np.sum(y_true_val_labels == i)}")

confusion_mat = confusion_matrix(y_true_val_labels, y_pred_val_labels)

# Display the confusion matrix
print("Validation Confusion Matrix:")
print(confusion_mat)

# Define class labels
class_labels = ['A', 'B', 'C']

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix from Validation Dataset")
plt.show()

"""############

y_pred_val = model.predict(x_3)

# Convert predicted probabilities to class labels
y_pred_val_labels = np.argmax(y_pred_val, axis=1)

# Convert true labels to class labels
y_true_val_labels = np.argmax(y_3, axis=1)

# Calculate the number of correct predictions for each class
class_counts = np.bincount(y_pred_val_labels[y_pred_val_labels == y_true_val_labels])

# Display the number of correct predictions for each class
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count}/{np.sum(y_true_val_labels == i)}")

confusion_mat = confusion_matrix(y_true_val_labels, y_pred_val_labels)

# Display the confusion matrix
print("Validation Confusion Matrix:")
print(confusion_mat)

# Define class labels
class_labels = ['A', 'B', 'C']

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix from Validation Dataset 3")
plt.show()

############

y_pred_val = model.predict(x_4)

# Convert predicted probabilities to class labels
y_pred_val_labels = np.argmax(y_pred_val, axis=1)

# Convert true labels to class labels
y_true_val_labels = np.argmax(y_4, axis=1)

# Calculate the number of correct predictions for each class
class_counts = np.bincount(y_pred_val_labels[y_pred_val_labels == y_true_val_labels])

# Display the number of correct predictions for each class
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count}/{np.sum(y_true_val_labels == i)}")

confusion_mat = confusion_matrix(y_true_val_labels, y_pred_val_labels)

# Display the confusion matrix
print("Validation Confusion Matrix:")
print(confusion_mat)

# Define class labels
class_labels = ['A', 'B', 'C']

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix from Validation Dataset 4")
plt.show()"""