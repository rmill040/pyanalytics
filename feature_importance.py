from __future__ import division

import numpy as np

__all__ = ["rank_importances"]

def rank_importances(trained_model = None, features_to_rank = -1, plot = False, return_indices = False):
	"""Calculate feature importance for tree-based models and prints summary and/or plots importances

	Parameters
	----------
	trained_model : class
		Trained tree-based model

	features_to_rank : int
		Number of features to rank

	plot : boolean
		Whether to plot feature importances

	return_indices : boolean
		Whether to return indices for important features
	
	Returns
	-------
	indices : 1d array-like
		Indices of important features if return_indices == True
	"""
	# Check if model has feature_importances_ attribute (most sklearn models do)
	try:
		importances = trained_model.feature_importances_
	except AttributeError:
		print('\nYour trained model does not have a feature_importances_ attribute. If the model is a tree model, it should be trained first\n')

	# Sort indices of importances from highest to lowest
	indices = np.argsort(importances)[::-1]

	# Determine how many features to return (with a lot of features, returning all of them is often challening to look through)
	if features_to_rank == -1:		# Return all features
		n_feats = len(indices)
	elif features_to_rank < len(indices):	# Return top k features
		n_feats = features_to_rank
	else:
		print('\nThe number of features you input (%d) is less than the number of features available (%d) -> Ranking only %d features\n' % (features_to_rank, len(indices), len(indices)))
		n_feats = len(indices)		# Return all features available since features_to_rank < features in data set

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(n_feats):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	if plot:
		# Need matplotlib
		import matplotlib.pyplot as plt

		# Get the standard deviation of the importances for error bars in plots
		std = np.std([tree.feature_importances_ for tree in trained_model.estimators_], axis=0)

		# Create plot
		plt.figure()
		plt.title("Feature Importances for %d Features" % n_feats)
		plt.bar(range(n_feats), importances[indices[:n_feats]], color="r", yerr = std[indices[:n_feats]], align="center")
		plt.xticks(range(n_feats), indices)
		plt.xlim([-1, n_feats])
		plt.xlabel('Feature IDs/Names')
		plt.ylabel('Importances')
		plt.show()

	# Return top rank feature indices in order from most important to least important
	if return_indices:
		return indices[:n_feats]
