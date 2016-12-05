from __future__ import division, print_function

import numpy as np

__all__ = ["EasyEnsemble"]


class EasyEnsemble(object):
	"""Class to implement EasyEnsemble algorithm

	# Note: Model only applicable to binary classification problems where the labels are 0/1. 

	Parameters
	----------
	resamples : int
		Number of resamples to use when building classifiers

	classifiers : list
		A list of classifiers to use for each resample

	verbose : boolean
		Whether to print status of algorithm training and testing
	
	Returns
	-------
	self : instance of EasyEnsemble
		Note: The number of training models is calculated as resamples*len(classifiers).
		Example: If 10 resamples with 3 classifiers, then 30 classifiers will be trained and saved for later use
	"""
	def __init__(self, resamples = 100, classifiers = None, verbose = False):
		self.resamples = resamples

		# Force classifiers to be a list
		if isinstance(classifiers, list) == False:
			classifiers = [classifiers]
		self.classifiers = classifiers
		self.verbose = verbose
		self.training_complete = False


	def fit(self, X = None, y = None):
		"""Train EasyEnsemble model

		Parameters
		----------
		X : 2d array-like
			Feature matrix of training data for input into classifiers
		
		y : 1d array-like
			Vector of ground truth labels of training data

		Returns
		-------
		None
			Trains an ensemble of models saved as attribute self.trained_models
		"""
		# Check to make sure labels start at 0
		assert(np.min(y) == 0), "Minimum class label should be 0, not %d. Fix labels before training model" % np.min(y)

		# Find minority and majority classes and get number of samples in each
		class_counts = np.bincount(y.astype('int')) # Convert to integer or else error will be thrown if float32/float64
		min_class = np.argmin(class_counts)
		maj_class = np.argmax(class_counts)

		n_min = class_counts[min_class]
		n_maj = class_counts[maj_class]

		# Split data into minority and majority classes
		min_idx = np.where(y == min_class)[0]
		maj_idx = np.where(y == maj_class)[0]

		X_min, y_min = X[min_idx], y[min_idx]
		X_maj, y_maj = X[maj_idx], y[maj_idx]

		# Undersample majority class using bootstrap samples (size == n_min) without replacement and train models
		self.trained_models = []
		for i in xrange(self.resamples):

			# Print training iteration every 10 samples
			if self.verbose:
				if (i+1) % 10 == 0:
					print('Training sample %d/%d\n' % (i+1, self.resamples))

			# Bootstrap sample b
			idx_boot = np.random.choice(np.arange(n_maj), size = n_min, replace = False)
			X_maj_boot = X_maj[idx_boot]
			y_maj_boot = y_maj[idx_boot]

			# Create balanced training set
			X_balanced = np.vstack((X_min, X_maj_boot))
			y_balanced = np.concatenate((y_min, y_maj_boot))

			# Train classifiers
			for model in self.classifiers:
				self.trained_models.append(model.fit(X_balanced, y_balanced))

		# Set flag to indicate training is finished
		self.training_complete = True


	def predict(self, X = None):
		"""Predict class labels on testing data

		Parameters
		----------
		X : 2d array-like
			Feature matrix of testing data
		
		Returns
		-------
		y_hat : 1d array-like
			Vector of predicted class labels
		"""
		# Make sure models were actually trained
		assert(self.training_complete == True), "Need to train models first using .fit() method"

		# Make predictions on provided test data
		predicted_probs = np.zeros((X.shape[0], 1))
		for model in self.trained_models:
			predicted_probs = predicted_probs + model.predict_proba(X)

		# Normalize probabilites (for level 2 aggregation)
		predicted_probs = predicted_probs/len(self.trained_models)

		return np.argmax(predicted_probs, axis = 1)


	def predict_proba(self, X = None):
		"""Predict class probabilities on testing data

		Note: Method assumes classifiers have built-in predict_proba() method

		Parameters
		----------
		X : 2d array-like
			Feature matrix of testing data
		
		Returns
		-------
		y_hat : 2d array-like
			Vector of predicted class probabilities -> Each row has two dimensions, one dimension for each class probability
		"""
		# Make sure models were actually trained
		assert(self.training_complete == True), "Need to train models first using .fit() method"

		# Make predictions on provided test data
		predicted_probs = np.zeros((X.shape[0], 1))
		for model in self.trained_models:
			predicted_probs = predicted_probs + model.predict_proba(X)

		# Normalize probabilites (for level 2 aggregation)
		return predicted_probs/len(self.trained_models)