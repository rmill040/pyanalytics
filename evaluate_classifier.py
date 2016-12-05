from __future__ import division, print_function

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

__all__ = ["classifier_metrics"]


def classifier_metrics(true = None, predicted = None, predicted_proba = None, class_names = None):
	"""Prints a classification report that calculates class-specific metrics and class-general metrics

	Parameters
	----------
	true : 1d array-like,
	   Ground truth class labels
	
	predicted : 1d array-like,
		Predicted class labels returned by a classifier's

	predicted_proba : 2d array-like,
		Predicted probabilities as returned by a classifier's predict_proba method

	class_names : array, shape = [n_labels]
		Optional list of class names to include in the classifier metrics report
	
	Returns
	-------
	report : string
		Text summary of class-specific and class-general metrics
	"""
	# CONFUSION MATRIX
	print('\n --- Confusion Matrix --- \n')
	cm = confusion_matrix(y_true = true, y_pred = predicted) # Store confusion matrix; used for binary problems
	print(cm, '\n')

	# Determine if binary vs multi-class problem
	unique_labels = np.unique(true)

	# Binary problem
	if len(unique_labels) == 2:
	
		# Get counts
		tp = cm[1, 1]	# True positives
		tn = cm[0, 0]	# True negatives
		fp = cm[0, 1]	# False positives
		fn = cm[1, 0]	# False negatives

		# Accuracy metrics
		accuracy = np.mean(true == predicted)
		specificity = tn / (tn + fp)	# class 0 accuracy
		sensitivity = tp / (tp + fn)	# class 1 accuracy
		AUC = roc_auc_score(true, predicted, average = 'weighted')

		# Log loss (cross-entropy) if predicted probabilities are provided
		if predicted_proba != None:
			logloss = log_loss(y_true = true, y_pred = predicted_proba)	
		else:
			logloss = None

		# Hamming loss (fraction of labels incorrectly predicted)
		hammingloss = hamming_loss(y_true = true, y_pred = predicted)

		# Print summary of overall metrics
		print('\n --- Metrics --- \n')
		print('\tAccuracy: %.3f\n' % accuracy)
		if class_names != None:
			print('\tSpecificity (Class %s == 0 Accuracy): %.3f\n' % (class_names[0], specificity))
			print('\tSensitivity (Class %s == 1 Accuracy): %.3f\n' % (class_names[1], sensitivity))
		else:
			print('\tSpecificity (Class == 0 Accuracy): %.3f\n' % (specificity))
			print('\tSensitivity (Class == 1 Accuracy): %.3f\n' % (sensitivity))
		print('\tAUC: %.3f\n' % AUC)
		if logloss:		
			print('\tLog loss: %.3f\n' % logloss)
		print('\tHamming loss: %.3f\n' % hammingloss)

	# Multi-class problem
	else:
		# SPECIFIC METRICS: MULTI-CLASS 
		print('\n --- Class-Specific Metrics --- \n')
		print(classification_report(y_true = true, y_pred = predicted, target_names = class_names, digits = 3))

		# OVERALL METRICS: MULTI-CLASS
		# Accuracy
		accuracy = np.mean(true == predicted)	

		# Area under the curve (weighted by class balance). Need to one-hot encode multi-class problems using label_binarize
		true_binarized = label_binarize(true, unique_labels)
		predicted_binarized = label_binarize(predicted, unique_labels)
		AUC = roc_auc_score(true_binarized, predicted_binarized, average = 'weighted')

		# Log loss (cross-entropy) if predicted probabilities are provided
		if predicted_proba != None:
			logloss = log_loss(y_true = true, y_pred = predicted_proba)	
		else:
			logloss = None

		# Hamming loss (fraction of labels incorrectly predicted)
		hammingloss = hamming_loss(y_true = true, y_pred = predicted)

		# Print summary of overall metrics
		if logloss:
			print('\n --- Overall Metrics --- \n')
			header_names = '{acc:<10s}{auc:<10s}{ll:<10s}{hl:<10s}\n'.format(acc = 'Accuracy', auc = 'AUC', ll = 'Log Loss', hl = 'Hamming Loss')
			print(header_names)
			print('{acc:<10.3f}{auc:<10.3f}{ll:<10.3f}{hl:<10.3f}\n'.format(acc = accuracy, auc = AUC, ll = logloss, hl = hammingloss))
		else:
			print('\n --- Overall Metrics --- \n')
			header_names = '{acc:<10s}{auc:<10s}{hl:<10s}\n'.format(acc = 'Accuracy', auc = 'AUC', hl = 'Hamming Loss')
			print(header_names)
			print('{acc:<10.3f}{auc:<10.3f}{hl:<10.3f}\n'.format(acc = accuracy, auc = AUC, hl = hammingloss))