from __future__ import division, print_function

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


__all__ = ["MetaModelClassifier", "MetaModelRegressor"]


# TODO: 1. Test one-hot encoding with classifiers in MetaModelClassifier
#       2. Test MetaModelRegressor --> Very basic tests performed


class MetaModelClassifier(object):
	"""
	Builds two level meta model for classification prediction problems

	Parameters
	----------
	level1 : list
		List of instantiated classification models

	level2 : class
		Instantiated classification model
	
	k : int (default = 5)
		Number of folds for cross-validation training to generate meta-features

	shuffle : boolean (default = False)
		Whether to shuffle the samples before cross-validation

	standardize : boolean (default = True)
		Whether to standardize inputs

	level1_needs_one_hot : list
		List of boolean arguments indicating which level 1 models need one-hot-encoding vectors for the labels

	level2_needs_one_hot = boolean
		Whether the level 2 model needs a one-hot encoded vector for the labels

	verbose : boolean (default = False)
		Whether to display output summary during training

	Returns
	-------
	self : instance of MetaModel
	"""
	def __init__(self, level1 = None, level2 = None, k = 5, shuffle = False, standardize = True, 
				 level1_needs_one_hot = None, level2_needs_one_hot = None, verbose = False):

		# Check and define class attributes
		if isinstance(level1, list) == False:
			level1 = [level1]
		for model in level1:
			assert(hasattr(model, 'predict_proba')), "%s does not have method .predict_proba()" % (model)
		self.level1 = level1
		self.level2 = level2
		self.k = k
		self.shuffle = shuffle
		self.standardize = standardize
		self.verbose = verbose

		# Number of level1 models
		self.n_level1 = len(level1)

		# Check for models that need y transformed into one-hot representation (usually neural networks)
		if level1_needs_one_hot == None:
			level1_needs_one_hot = self.n_level1*[0]
		else:
			if isinstance(level1_needs_one_hot, list) == False:
				level1_needs_one_hot = [level1_needs_one_hot]
			assert(len(level1_needs_one_hot) == self.n_level1), "level1_needs_one_hot length (%d) is not the same as the number of level 1 models (%d)" % (len(level1_needs_one_hot), self.n_level1)
		self.level1_needs_one_hot = level1_needs_one_hot

		if level2_needs_one_hot == None:
			level2_needs_one_hot = 0
		self.level2_needs_one_hot = level2_needs_one_hot

		# Flag indicating if models trained
		self.trained = False


	def _get_folds(self, X = None, y = None):
		"""
		Gets train and test indices for cross-validation.

		Parameters
		----------
		X : 2d array-like
			Feature matrix from training data

		y : 1d array-like
			Array of labels from training data

		Returns
		-------
		fold_generator : generator
			A generator that is used to generate training and testing indices for cross-validation
		"""
		return StratifiedKFold(n_splits = self.k, shuffle = self.shuffle)


	@staticmethod
	def _one_hot(y = None):
		"""
		Create one hot encoding representation of y

		Parameters
		----------
		y : 1d array-like
			Array of labels from training data

		Returns
		-------
		y_onehot : 2d array-like
			Array of one-hot encoded form of y
		"""
		enc = OneHotEncoder()
		return enc.fit(y)


	def _standardize_train(self, X = None):
		"""
		Standardize training features to have mean 0 and variance 1
		
		Parameters
		----------
		X : 2d array-like
			Feature matrix from training data

		Returns
		-------
		X_norm : 2d array-like
			Standardized feature matrix of training data
	
		scaler : instance of preprocessing class
			Transformation object that can be used to transform testing data
		"""	
		scaler = StandardScaler().fit(X)
		return scaler.transform(X), scaler


	def _standardize_test(self, X = None, scaler = None):
		"""
		Standardize testing features to have mean 0 and variance 1
		
		Parameters
		----------
		X : 2d array-like
			Feature matrix from testing data

		Returns
		-------
		X_norm : 2d array-like
			Standardized feature matrix of testing data
		"""	
		return scaler.transform(X)


	def fit(self, X = None, y = None):
		"""
		Trains two-level meta-model using k-fold cross-validation to get meta-features from level 1 models

		Parameters
		----------
		X : 2d array-like
			Feature matrix from training data	

		y : 1d array-like
			Array of labels from training data

		Returns
		-------
		None
			Trains MetaModel so that .predict() method is callable
		"""
		if self.verbose:
			print('\nTraining: {0}-Fold CV to Generate Meta-Features from Level 1 Models...'.format(self.k))

		# Reshape y
		y = y.ravel()

		# Generate indices for cross-validation and preallocate data structures
		fold_generator = self._get_folds(X, y)			# Generator for CV folds
		X_meta = []						# Empty list for meta-features
		level1_acc = np.zeros((self.n_level1, self.k)) 		# Numpy array for level1 model accuracies across folds
		n_classes = np.unique(y)				# Unique classes
		fold = 0						# Fold counter

		##########################################################################
		####################### GENERATE META-FEATURES ###########################
		##########################################################################

		# Start cross-validation loop for level 1 models to generate meta-features
		for train_idx, test_idx in fold_generator.split(X, y):
			if self.verbose:
				print('\n\tFold:', fold + 1)
			
			# Temporary storage of meta-features for all level 1 models in current CV split
			meta = []

			# Loop through each model for current fold and make prediction
			for i in xrange(self.n_level1):
				
				# Standardize training data and test data with current CV split if specified
				if self.standardize:
					X_train, scaler = self._standardize_train(X[train_idx])
					X_test = self._standardize_test(X[test_idx], scaler)
				else:
					X_train, X_test = X[train_idx], X[test_idx]

				# Create one-hot encoding of y if specified
				if self.level1_needs_one_hot[i]:
					y_train, y_test = self._one_hot(y[train_idx]), self._one_hot(y[test_idx])
				else:
					y_train, y_test = y[train_idx], y[test_idx]

				# Train ith level 1 model using features of current CV split, get predictions, insert into meta, get accuracy metric at fold
				self.level1[i].fit(X_train, y_train)
				y_probs = self.level1[i].predict_proba(X_test)
				meta.append(y_probs)
				level1_acc[i, fold] = self.score(y_test, y_probs)

			# Horizontally concatenate all meta-features in current CV split then append to X_meta list, and increment fold counter
			meta = np.hstack((meta))
			X_meta.append(meta) 
			fold = fold + 1

		##########################################################################
		################## RETRAIN LEVEL 1 MODELS ON ALL DATA ####################
		##########################################################################

		# Now retrain level 1 models on all data
		if self.verbose:
			print('\nTraining Level 1 Models on all data...')

		# Standardize if needed
		if self.standardize:
			X_all, self.scalers1 = self._standardize_train(X)	# Save scaler for level 1 models
		else:
			X_all = X.copy()

		# Train level 1 models
		for i in xrange(self.n_level1):
			if self.verbose:
				print('\n\tModel {0}'.format(i+1))

			# Copy of y in case one-hot changes across models
			y_all = y.copy()
			if self.level1_needs_one_hot[i]:
				y_all = self._one_hot(y)

			self.level1[i].fit(X_all, y_all)

		##########################################################################
		######################### TRAIN LEVEL 2 MODEL ############################
		##########################################################################
		
		# Stack all meta-features together then create augmented matrix of original features with meta-features
		X_meta = np.vstack(X_meta)
		X_aug = np.hstack((X, X_meta))

		if self.level2_needs_one_hot:
			y_all = self._one_hot(y)

		# Train level 2 model
		if self.verbose:
			print('\nTraining Level 2 Model on all data...')

		# Standardize augmented feature set if specified
		self.level2.fit(X_aug, y_all)				
		self.trained = True

		# Variable to indicate that models were already trained		
		if self.verbose:
			print('{:-^60}'.format(''))
			print('{:^60}'.format('META-MODEL SUMMARY'))
			print('{:-^60}\n'.format(''))

			# Create strings
			if self.standardize:
				standardize_str = 'True'
			else:
				standardize_str = 'False'

			if self.shuffle:
				shuffle_str = 'True'
			else:
				shuffle_str = 'False'

			print('{0:<25}{1:<25}'.format('Samples:', X.shape[0]))
			print('{0:<25}{1:<25}'.format('Classes:', len(n_classes)))
			print('{0:<25}{1:<25}'.format('Level 1 Models:',self.n_level1))
			print('{0:<25}{1:<25}'.format('Features:', X.shape[1]))
			print('{0:<25}{1:<25}'.format('Meta-Features:', X_meta.shape[1]))
			print('{0:<25}{1:<25}'.format('Folds:', self.k))
			print('{0:<25}{1:<25}'.format('Standardize:', standardize_str))
			print('{0:<25}{1:<25}'.format('Shuffle:', shuffle_str))

			print('\n -- Average Classification Accuracy : Level 1 Models --\n')
			for i in xrange(self.n_level1):
				print('\tModel', i+1, ':', np.mean(level1_acc[i, :])) # Average accuracy across folds


	def predict(self, X = None):
		"""
		Predict labels from set of test features

		Parameters
		----------
		X : 2d array-like numpy array
			Feature matrix from testing data

		Returns
		-------
		y_hat : 1d array-like
			Predicted labels from meta-model
		"""
		# Make sure models were trained first
		assert(self.trained == True), "Error: Need to call .fit() method to train models before calling .predict() method"

		# Get size of testing features, preallocate empty array for meta features
		X = np.atleast_2d(X) 	# Ensure that X has 2 dimensions
		n = X.shape[0]
		X_meta = []
		fold = 0

		# Standardize testing data if specified
		if self.standardize:
			X = self._standardize_test(X, self.scalers1) # Use saved scalers for level 1 models from training data

		# Loop through each level 1 model and make prediction
		for i in xrange(self.n_level1):

			# Append meta-features
			X_meta.append(self.level1[i].predict_proba(X))
	
		# Concatenate meta-features and augment feature matrix
		X_meta = np.hstack(X_meta)
		X_aug = np.hstack((X, X_meta))

		# Level 2 Model
		if self.verbose:
			print('\nTesting: Making Predictions For Level 2 Model...\n')
		return self.level2.predict(X_aug)


	@staticmethod
	def score(y_test = None, y_probs = None):
		"""
		Calculate classification accuracy

		Parameters
		----------
		y_true : 1d array-like
			Array of ground truth labels

		y_probs : 2d array-like
			Array of predicted probabilities

		Returns
		-------
		metric: float
			Classification accuracy
		"""
		y_predict = np.argmax(y_probs, axis = 1)
		return np.mean(y_test.ravel() == y_predict.ravel())


class MetaModelRegressor(object):
	"""
	Builds two level meta model for regression prediction problems

	Parameters
	----------
	level1 : list
		List of instantiated regression models

	level2 : class
		Instantiated regression model
	
	k : int (default = 5)
		Number of folds for cross-validation training to generate meta-features

	shuffle : boolean (default = False)
		Whether to shuffle the samples before cross-validation

	standardize : boolean (default = True)
		Whether to standardize inputs

	verbose : boolean (default = False)
		Whether to display output summary during training

	Returns
	-------
	self : instance of MetaModel
	"""
	def __init__(self, level1 = None, level2 = None, k = 5, shuffle = False, standardize = True, 
				verbose = False):

		# Check and define class attributes
		if isinstance(level1, list) == False:
			level1 = [level1]
		for model in level1:
			assert(hasattr(model, 'predict')), "%s does not have method .predict()" % (model)
		self.level1 = level1
		self.level2 = level2
		self.k = k
		self.shuffle = shuffle
		self.standardize = standardize
		self.verbose = verbose

		# Number of level1 models
		self.n_level1 = len(level1)

		# Flag indicating if models trained
		self.trained = False


	def _get_folds(self, X = None):
		"""
		Gets train and test indices for cross-validation.

		Parameters
		----------
		X : 2d array-like
			Feature matrix from training data

		Returns
		-------
		fold_generator : generator
			A generator that is used to generate training and testing indices for cross-validation
		"""
		return KFold(n_splits = self.k, shuffle = self.shuffle)


	def _standardize_train(self, X = None):
		"""
		Standardize training features to have mean 0 and variance 1
		
		Parameters
		----------
		X : 2d array-like
			Feature matrix from training data

		Returns
		-------
		X_norm : 2d array-like
			Standardized feature matrix of training data
	
		scaler : instance of preprocessing class
			Transformation object that can be used to transform testing data
		"""	
		scaler = StandardScaler().fit(X)
		return scaler.transform(X), scaler


	def _standardize_test(self, X = None, scaler = None):
		"""
		Standardize testing features to have mean 0 and variance 1
		
		Parameters
		----------
		X : 2d array-like
			Feature matrix from testing data

		Returns
		-------
		X_norm : 2d array-like
			Standardized feature matrix of testing data
		"""	
		return scaler.transform(X)


	def fit(self, X = None, y = None):
		"""
		Trains two-level meta-model using k-fold cross-validation to get meta-features from level 1 models

		Parameters
		----------
		X : 2d array-like
			Feature matrix from training data	

		y : 1d array-like
			Array of labels from training data

		Returns
		-------
		None
			Trains MetaModel so that .predict() method is callable
		"""
		if self.verbose:
			print('\nTraining: {0}-Fold CV to Generate Meta-Features from Level 1 Models...'.format(self.k))

		# Reshape y
		y = y.ravel()

		# Generate indices for cross-validation and preallocate data structures
		fold_generator = self._get_folds(X)			# Generator for CV folds
		X_meta = []						# Empty list for meta-features
		level1_acc = np.zeros((self.n_level1, self.k)) 		# Numpy array for level1 model accuracies across folds
		fold = 0						# Fold counter

		##########################################################################
		####################### GENERATE META-FEATURES ###########################
		##########################################################################

		# Start cross-validation loop for level 1 models to generate meta-features
		for train_idx, test_idx in fold_generator.split(X):
			if self.verbose:
				print('\n\tFold:', fold + 1)
			
			# Temporary storage of meta-features for all level 1 models in current CV split
			meta = []

			# Loop through each model for current fold and make prediction
			for i in xrange(self.n_level1):
				
				# Standardize training data and test data with current CV split if specified
				if self.standardize:
					X_train, scaler = self._standardize_train(X[train_idx])
					X_test = self._standardize_test(X[test_idx], scaler)
				else:
					X_train, X_test = X[train_idx], X[test_idx]

				# Labels for current CV split
				y_train, y_test = y[train_idx], y[test_idx]

				# Train ith level 1 model using features of current CV split, get predictions, insert into meta, get accuracy metric at fold
				self.level1[i].fit(X_train, y_train)
				y_hat = self.level1[i].predict(X_test)
				meta.append(y_hat.reshape(-1, 1))
				level1_acc[i, fold] = self.score(y_test, y_hat)

			# Horizontally concatenate all meta-features in current CV split then append to X_meta list, and increment fold counter
			meta = np.hstack((meta))
			X_meta.append(meta) 
			fold = fold + 1

		##########################################################################
		################## RETRAIN LEVEL 1 MODELS ON ALL DATA ####################
		##########################################################################

		# Now retrain level 1 models on all data
		if self.verbose:
			print('\nTraining Level 1 Models on all data...')

		# Standardize if needed
		if self.standardize:
			X_all, self.scalers1 = self._standardize_train(X)	# Save scaler for level 1 models
		else:
			X_all = X.copy()

		# Train level 1 models
		for i in xrange(self.n_level1):
			if self.verbose:
				print('\n\tModel {0}'.format(i+1))
			self.level1[i].fit(X_all, y)

		##########################################################################
		######################### TRAIN LEVEL 2 MODEL ############################
		##########################################################################
		
		# Stack all meta-features together then create augmented matrix of original features with meta-features
		X_meta = np.vstack(X_meta)
		X_aug = np.hstack((X, X_meta))

		# Train level 2 model
		if self.verbose:
			print('\nTraining Level 2 Model on all data...')

		# Standardize augmented feature set if specified
		self.level2.fit(X_aug, y)				
		self.trained = True

		# Variable to indicate that models were already trained		
		if self.verbose:
			print('{:-^60}'.format(''))
			print('{:^60}'.format('META-MODEL SUMMARY'))
			print('{:-^60}\n'.format(''))

			# Create strings
			if self.standardize:
				standardize_str = 'True'
			else:
				standardize_str = 'False'

			if self.shuffle:
				shuffle_str = 'True'
			else:
				shuffle_str = 'False'

			print('{0:<25}{1:<25}'.format('Samples:', X.shape[0]))
			print('{0:<25}{1:<25}'.format('Level 1 Models:',self.n_level1))
			print('{0:<25}{1:<25}'.format('Features:', X.shape[1]))
			print('{0:<25}{1:<25}'.format('Meta-Features:', X_meta.shape[1]))
			print('{0:<25}{1:<25}'.format('Folds:', self.k))
			print('{0:<25}{1:<25}'.format('Standardize:', standardize_str))
			print('{0:<25}{1:<25}'.format('Shuffle:', shuffle_str))

			print('\n -- Average Mean Squared Error : Level 1 Models --\n')
			for i in xrange(self.n_level1):
				print('\tModel', i+1, ':', np.mean(level1_acc[i, :])) # Average accuracy across folds


	def predict(self, X = None):
		"""
		Predict labels from set of test features

		Parameters
		----------
		X : 2d array-like numpy array
			Feature matrix from testing data

		Returns
		-------
		y_hat : 1d array-like
			Predicted labels from meta-model
		"""
		# Make sure models were trained first
		assert(self.trained == True), "Error: Need to call .fit() method to train models before calling .predict() method"

		# Get size of testing features, preallocate empty array for meta features
		X = np.atleast_2d(X) 	# Ensure that X has 2 dimensions
		n = X.shape[0]
		X_meta = []
		fold = 0

		# Standardize testing data if specified
		if self.standardize:
			X = self._standardize_test(X, self.scalers1) # Use saved scalers for level 1 models from training data

		# Loop through each level 1 model and make prediction
		for i in xrange(self.n_level1):

			# Append meta-features
			X_meta.append(self.level1[i].predict(X).reshape(-1, 1))
	
		# Concatenate meta-features and augment feature matrix
		X_meta = np.hstack(X_meta)
		X_aug = np.hstack((X, X_meta))

		# Level 2 Model
		if self.verbose:
			print('\nTesting: Making Predictions For Level 2 Model...\n')
		return self.level2.predict(X_aug)


	@staticmethod
	def score(y_test = None, y_predict = None):
		"""
		Calculate regression accuracy using mean squared error

		Parameters
		----------
		y_true : 1d array-like
			Array of ground truth labels

		y_predict : 1d array-like
			Array of predicted labels

		Returns
		-------
		metric: float
			Mean squared error
		"""
		return np.mean(np.sum((y_test.ravel() - y_predict.ravel()))**2)
