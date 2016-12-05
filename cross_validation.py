from __future__ import division, print_function

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

__all__ = ["temporal_skf_cv"]


def temporal_skf_cv(model = None, X = None, y = None, folds = 5, metric = 'acc', standardize = True, verbose = True):
	"""Temporal stratified k-fold cross-validation for evaluating classifier performance. 

	Parameters
	----------
	model : instantiated machine learning classifier with .fit() method
		The model that will be evaluated with cross-validation

	X : 2d numpy array with dimensions = [n_samples, n_features]
		Feature array

	y : 1d numpy array with dimensions = [n_samples]
		Labels array

	folds : int
		Number of equal folds to split data into

	metric : str
		Metric used in the cross-validation procedure

	standardize : boolean
		Whether to standardize data (important for models that are not scale invariant for the features)

	verbose: boolean
		Whether to print metrics at each round of testing
	
	Returns
	-------
	cv_mean : float
		Average metric across folds

	cv_std : float
		Standard deviation metric across folds
	"""
	# Number of samples
	n = X.shape[0]
	unique_classes = np.unique(y)
	n_classes = len(unique_classes)

	# Get frequency of occurrence
	freqs = np.bincount(y.astype('int32'))

	# Check to make sure there is at least one data point to be used in each fold
	assert(np.all(freqs/folds >= 1)), "At least one sample should be present for each class in each fold. Reduce the number of folds"

	# Split indices into k stratified folds (class proportions in each fold should be approximately the same as the full sample)
	idx_split = []
	for i, label in enumerate(unique_classes):
		tmp = np.where(y == label)[0] # Find indices for each class
		idx_split.append(np.array_split(tmp, indices_or_sections = folds))

	# Combine indices from each class into k folds
	idx_folds = []
	for k in xrange(folds):
		tmp = []
		for i in xrange(n_classes):
			tmp.append(idx_split[i][k])
		idx_folds.append(np.concatenate(tmp))

	# Cross validation
	estimates = np.zeros((folds-1,))
	training_folds = []
	for k in xrange(folds-1):
		
		# Training is based on following scheme: Testing can only be done in the future with time series data. 
		# For example, with 4 folds there will be 3 rounds of testing:
		#	 Round 1: train(1) | test(2)
		#	 Round 2: train(1, 2) | test(3)
		# 	 Round 3: train(1, 2, 3) | test(4)
		
		train_idx = np.concatenate(idx_folds[:k+1]) # When k = 0, this will only use the indices for the first fold
		test_idx = idx_folds[k+1]

		# Split data into features and labels
		X_train, X_test = X[train_idx], X[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]

		# Standardize if needed
		if standardize:
			scaler = StandardScaler().fit(X_train)
			X_train = scaler.transform(X_train)
			X_test = scaler.transform(X_test)

		# Train model and predict labels on test set
		model.fit(X_train, y_train.ravel())
		y_hat = model.predict(X_test)

		# Calculate metric of interest
		if metric == 'acc':
			estimates[k] = np.mean(y_test.ravel() == y_hat.ravel())
			if verbose:
				training_folds.append(k+1)
				print('\n\tRound %d: %s = %.3f - Train %s | Test %d\n' % (k+1, metric, estimates[k], str(training_folds), k+2))
		elif metric == 'auc':
			estimates[k] = roc_auc_score(y_true = y_test, y_score = y_hat, average = 'weighted')
			if verbose:
				training_folds.append(k+1)
				print('\n\tRound %d: %s = %.3f - Train %s | Test %d\n' % (k+1, metric, estimates[k], str(training_folds), k+2))
		else:
			raise ValueError('%s not a valid metric' % metric)

	# Print summary and return estimates
	if verbose: 
		print('\nOverall %s = %.3f, SD = %.3f\n' % (metric, np.mean(estimates), np.std(estimates)))

	return np.mean(estimates), np.std(estimates)