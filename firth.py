from __future__ import division, print_function

import numpy as np

__all__ = ["FirthLogisticRegression"]

# TODO: - Add catches for singular matrices, maybe add generalized inverse in some cases
#		- Compare method to R's implementation
#		- Add misc error checking
#		- Find example that tests half-stepping
#		- Add functionality for confidence intervals
#		- Add functionality to plot decision boundary


class FirthLogisticRegression(object):
	"""A naive implementation of Firth's logistic regression

	Notation below: ' = matrix transpose
					* = matrix multiplication or elementwise multiplication (lazy notation here)

	Parameters
	----------
	fit_intercept : bool (default True)
		Whether to fit the intercept in the model. If true, a vector of ones is appended to the covariate matrix

	max_its : int (default 100)
		Maximum iterations for fitting

	tol : float (default 1e-12)
		Tolerance criteria for convergence

	verbose : bool (default False)
		Whether to print status of fitting algorithm. Value of 1 indicates print only convergence status,
		value > 1 print status at every iteration

	half_step : bool (default True)
		Whether to implement half-step method to deal with convergence issues

		Note: This method is not tested since simple tests did not meet criteria. Also, may need to incorporate
		      Firth's score correction in the implementation as opposed to score vector in classific logistic reg
    
    Returns
    -------
    self : object
        Instance of FirthLogisticRegression class
    """
	def __init__(self, fit_intercept = True, max_its = 100, tol = 1e-12, verbose = False, half_step = True):
		_valid_bool = [True, False, 1, 0]
		if fit_intercept in _valid_bool:
			self.fit_intercept = fit_intercept
		else:
			raise ValueError('%s not a valid fit_intercept argument. Valid arguments are %s' % (fit_intercept, _valid_bool))
		
		self.max_its = max_its
		self.tol = tol
		
		if verbose in _valid_bool or verbose > 1:
			self.verbose = verbose
		else:
			raise ValueError('%s not a valid verbose argument' % verbose)

		if half_step in _valid_bool:
			self.half_step = half_step
		else:
			raise ValueError('%s not a valid half_step argument. Valid arguments are %s' % (half_step, _valid_bool))


	@staticmethod
	def _logit(X = None, b = None):
		"""Logit transformation that generates predicted probabilities based on X and b

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates

		b : 1d array-like
			Array of coefficients

		Returns
		-------
		p : 1d array-like
			Predicted probabilities based on X and b
		"""
		exp_Xb = np.exp(np.dot(X, b))
		return (exp_Xb / (1 + exp_Xb)).reshape(-1, 1)


	def _log_likelihood(self, X = None, b = None, y = None):
		"""Calculate log-likelihood (of Bernoulli distribution) for n samples as y*log(p) + (1 - y)*log(1 - p)

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates

		b : 1d array-like
			Array of coefficients
		
		y : 1d array-like
			Array of dependent variable (or labels)

		Returns
		-------
		ll : float
			Log-likelihood value
		"""
		p = self._logit(X = X, b = b)
		return np.sum(y*np.log(p) + (1 - y)*np.log(1 - p))


	@staticmethod
	def _weight_matrix(p = None):
		"""Calculate weight matrix as I * (p*(1 - p)), where I is the j x j identity matrix (j = # covariates)

		Parameters
		----------
		p : 1d array-like
			Array of predicted probabilities

		Returns
		-------
		W : 2d array-like
			Weight matrix
		"""
		# Create identity matrix
		I = np.eye(len(p))
		return I * (p * (1-p))


	@staticmethod
	def _hessian(X = None, W = None):
		"""Hessian matrix calculated as -X'*W*X

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates

		W : 2d array-like
			Weight matrix

		Returns
		-------
		hessian : 2d array-like
			Hessian matrix
		"""
		return -np.dot(X.T, np.dot(W, X))


	def _hat_matrix(self, X = None, W = None):
		"""Calculate hat matrix hat = W^(1/2) * X * (X'*W*X)^(-1) * X'*W^(1/2)

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates

		W : 2d array-like
			Diagonal weight matrix

		Returns
		-------
		hat : 2d array-like
			Hat matrix

			Note: If half-step method not implemented, this matrix is simply not used
		"""
		# W^(1/2)
		Wsqrt = W**2

		# (X'*W*X)^(-1)
		XtWX = -self._hessian(X = X, W = W)
		XtWX_inv = np.linalg.inv(XtWX)

		# W^(1/2)*X
		WsqrtX = np.dot(Wsqrt, X)

		# X'*W^(1/2)
		XtWsqrt = np.dot(X.T, Wsqrt)

		return np.dot(WsqrtX, np.dot(XtWX_inv, XtWsqrt))


	@staticmethod
	def _firth_score(X = None, y = None, p = None, hat = None, W = None):
		"""Score (gradient) vector with Firth's correction as X' * (y - p + h * (.5 - p))

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates

		y : 1d array-like
			Array of dependent variable (or labels)

		p : 1d array-like
			Array of predicted probabilities

		hat : 2d array-like
			Hat matrix

		W : 2d array-like
			Weight matrix

		Returns
		-------
		score : 1d array-like
			Score vector with Firth's correction
		"""
		# Calculate 'residuals'
		resid = y - p

		# Calculate bias for gradient vector
		h = np.diag(hat).reshape(-1, 1)		# Diagonal of hat matrix
		adj = (.5 - p).reshape(-1, 1)		# Adjustment term
		bias = (h*adj).reshape(-1, 1)		# Bias term

		return np.dot(X.T, (resid + bias))


	@staticmethod
	def _update(b_old = None, hessian = None, score = None):
		"""Newton update for coefficients as b_old - hessian^(-1)*score

		Parameters
		----------
		b_old : 1d array-like
			Array of coefficients for (i - 1)th iteration

		hessian : 2d array-like
			Hessian matrix

		score : 1d array-like
			Score vector

		Returns
		-------
		b_new : float
			Updated coefficient based on Newton's method
		"""
		# Newton update
		return b_old.reshape(-1, 1) - np.dot(np.linalg.inv(hessian), score)


	def _check_convergence(self, ll_old = None, ll_new = None):
		"""Check convergence of current iteration

		Parameters
		----------
		ll_old : float
			Log-likelihood at (i - 1)th iteration

		ll_new : float
			Log-likelihood at ith iteration

		Returns
		-------
		status : bool
			Whether algorithm converges (1) or not (0)
		"""
		if np.abs(ll_old - ll_new) < self.tol:
			return 1
		else:
			return 0


	@staticmethod
	def _halfstep_adjust(X = None, y = None, p = None, XtWX = None, b = None, step = None):
		"""Half step adjustment method

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates

		y : 1d array-like
			Matrix of dependent variable (or labels)

		p : 1d array-like
			Array of predicted probabilities

		XtWX : 2d array-like
			Negative hessian matrix

		b : 1d array-like
			Array of coefficients

		step : float
			Step at the ith iteration

		Returns
		-------
		b_new : 1d array-like
			Updated coefficients using half-step method
		"""
		# Calculate 'residual' as y - p
		resid = y.reshape(-1, 1) - p.reshape(-1, 1)

		# X'*(y - p)
		Xt_resid = np.dot(X.T, resid)

		# Solutions to equation X'*W*X = X'*(y - p)
		delta = np.linalg.solve(XtWX, Xt_resid)

		# Check for step criteria
		if np.abs(step - 1.0) > .001:
			delta *= step

		# Update coefficients
		b += delta
		return b


	def fit(self, X = None, y = None):
		"""Main method to fit the logistic regression model and obtain coefficients

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates

		y : 1d array-like
			Array of dependent variable (or labels)

		Returns
		-------
		self : object
			Instance of self
		"""
		# Error checking
		if set(y.ravel()) != set([0, 1]):
			# Try to adjust y to force labels to be 0 and 1
			assert(len(set(y.ravel())) == 2), 'y can only have 2 classes. Current array contains %d classes' % (len(set(y.ravel())))
			
			# Find minimum and maximum values along with indices for each
			min_y, max_y = np.min(y), np.max(y)
			id_min, id_max = np.where(y == min_y)[0], id_max = np.where(y == max_y)[0]

			# Make adjustments now --> class with smallest numeric ID number is set to 0, else set to 1
			y[id_min] = y[id_min] - min_y
			y[id_max] = y[id_max] - max_y + 1

		# Add intercept if needed
		if self.fit_intercept:
			ones = np.ones((X.shape[0], 1))
			X = np.hstack((ones, X))

		# Dimension of feature matrix
		_, p = X.shape

		# Preallocate beta vector
		b_old = np.zeros(p)

		# Starting likelihood
		ll_old = self._log_likelihood(X = X, b = b_old, y = y)

		# Starting step size
		step = 1.0
		i = 1

		# Start algorithm
		while i < self.max_its:
		
			# Predicted probabilities
			p = self._logit(X = X, b = b_old)

			# Weight matrix
			W = self._weight_matrix(p = p)

			# Hat matrix
			hat = self._hat_matrix(W = W, X = X)

			# Score vector and hessian matrix (save as attributes)
			self.score, self.hessian = self._firth_score(X = X, y = y, p = p, hat = hat, W = W), self._hessian(X = X, W = W)

			# Update coefficients and calculate likelihood
			b_new = self._update(b_old = b_old, hessian = self.hessian, score = self.score)
			ll_new = self._log_likelihood(X = X, b = b_new, y = y)

			# Check covergence
			if self._check_convergence(ll_old = ll_old, ll_new = ll_new):
				self.coef_ = b_new		# Save final coefficients as attribute

				# Print if necessary
				if self.verbose == 1:
					print('Algorithm converged after %d iterations\n' % i)
				elif self.verbose > 1:
					print('Iteration %d | log-likelihood = %f\n' % (i, ll_new))
					print('Algorithm converged after %d iterations\n' % i)
				else:
					pass
				# Break after printing option
				break
			
			else:

				# Implement half-step method if specified to try and find better b_new			
				if self.half_step:
					if ll_new < ll_old:
						step /= 2.0
						b_new = self._halfstep_adjust(X = X, y = y, p = p, XtWX = -self.hessian, b = b_old, step = step)
						ll_new = self._log_likelihood(X = X, b = b_new, y = y)

				# Print if necessary
				if self.verbose > 1:
					print('Iteration %d | log-likelihood = %f' % (i, ll_new))

				# Copy old iterations and increase counter
				b_old, ll_old = b_new, ll_new
				i += 1

		else:
			raise RuntimeError('Algorithm did not converge after %d iterations\n' % self.max_its)


	def predict_proba(self, X = None):
		"""Calculate predicted probabilities

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates (usually from testing set)

		y : 1d array-like
			Array of dependent variable (or labels)

		Returns
		-------
		y_probs : 1d array-like
			Array of predicted class probabilities (dimension is [n, 2] since each class gets a probability)
		"""
		return np._logit(X = X, b = self.coef_)


	def predict(self, X = None):
		"""Calculate class labels

		TODO: Implement a vectorized version of converting to 0/1 labels

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates (usually from testing set)

		Returns
		-------
		y_classes : 1d array-like
			Array of predicted class labels
		"""
		y_probs = self._predict_proba(X = X)
		y_classes = np.zeros(y_probs.shape)

		# Threshold probabilities
		for i in xrange(y_probs.shape[0]):
			if y_probs >= .5:
				y_classes[i] = 1
		return y_classes


	def score(self, y_true = None, y_hat = None):
		"""Calculate classification accuracy

		Parameters
		----------
		y_true : 1d array-like
			Ground truth of dependent variable (or labels)

		y_hat : 1d array-like
			Predicted dependent variable (or labels)

		Returns
		-------
		acc : float
			Classification accuracy
		"""
		return np.mean(y_true == y_hat)


	def estimate(self, X = None, y = None):
		"""Main method to estimate coefficients and standard errors

		Parameters
		----------
		X : 2d array-like
			Matrix of covariates

		y : 1d array-like
			Array of dependent variable (or labels)

		Returns
		-------
		estimates : 2d array-like
			Matrix of dimension [j, 2] that contains point estimates in column 1 and standard errors in column 2,
			where j is the number of covariates
		"""
		# Estimate model and get coefficients and hessian matrix
		self.fit(X = X, y = y)

		# Get information matrix (calculated as negative expected hessian matrix evaluated at self.coef_)
		info = -self.hessian

		# Invert information matrix to get variance/covariance matrix and extract diagonal for variances of self.coef_
		# then take square root for standard errors
		var_cov = np.linalg.inv(info)		
		se = np.sqrt(np.diag(var_cov))

		return np.hstack((self.coef_.reshape(-1, 1), se.reshape(-1, 1)))


def main():

	# TESTS
	from sklearn.linear_model import LogisticRegression
	import statsmodels.api as sm

	# Simulate simple logistic regression model
	N = 500
	b0 = 1
	b1 = .59

	# Covariate(s)
	x = np.random.normal(0, 1, (N, 1))

	# Dependent variable
	y_linear = b0 + b1*x
	y_probs = 1 / (1 + np.exp(-y_linear))
	y = np.random.binomial(1, y_probs, (N, 1))

	# Firth method with half-step
	firth_half = FirthLogisticRegression(verbose = 2, half_step = True)
	firth_half_estimates = firth_half.estimate(X = x, y = y)
	
	# Firth method w/o half-step
	firth_reg = FirthLogisticRegression(verbose = 2, half_step = False)
	firth_reg_estimates = firth_reg.estimate(X = x, y = y)

	# Sklearn's logistic regression
	# Append intercept term to x for sklearn and statsmodel
	x = np.hstack((np.ones(x.shape), x))
	sklearn = LogisticRegression(fit_intercept = False)
	sklearn.fit(x, y.ravel())
	sklearn_coef = sklearn.coef_

	# Statsmodels logistic regression
	statsmodel = sm.GLM(y, x, family = sm.families.Binomial())
	results = statsmodel.fit()
	statsmodel_coef = results.params
	statsmodel_se = np.sqrt(np.diag(-np.linalg.inv(statsmodel.information(statsmodel_coef))))

	print('\n---- POINT ESTIMATES\n')
	print('Firth logistic regression w/ half-stepping', firth_half_estimates[:, 0].reshape(1, -1))
	print('Firth logistic regression w/o half-stepping', firth_reg_estimates[:, 0].reshape(1, -1))
	print('Sklearn logistic regression', sklearn_coef.reshape(1, -1))
	print('Statsmodels logistic regression', statsmodel_coef.reshape(1, -1))

	print('\n---- STANDARD ERRORS\n')
	print('Firth logistic regression w/ half-stepping', firth_half_estimates[:, 1].reshape(1, -1))
	print('Firth logistic regression w/o half-stepping', firth_reg_estimates[:, 1].reshape(1, -1))
	print('Statsmodels logistic regression', statsmodel_se.reshape(1, -1))

if __name__ == "__main__":
	main()