
# This code was written by Marco de Innocentis in the fall of 2018 for a short investigation on
# ML for option pricing.
#
# The program below, together with the associated dataset (not included) is used to fit 30 
# neural networks to the prices or implied volatilities for a set of vanilla option prices
# of varying moneyness and times to maturity under the 5-parameter Heston stochastic volatility 
# model, using the pricing algorithm described in https://www.risk.net/media/download/957931/download
#
# The code runs under Python 3.6.5 and TensorFlow 1.10.1. It may not work, or may generate several 
# warnings, on different configurations.
#
# The code: 
# 1. Partitions the dataset into a) training sets, one for each NN, b) validation sets, used
# 	 to assess each calibration's performance, c) testing and regression sets (more about these later).
# 2. Each NN is calibrated. Up to 200 epochs are used, however calibration terminates after 15 epochs 
#    of non-improving performance. The "best" model is then taken to be the one with best out of 
#    sample (i.e. validation) performance before stopping. Both the "final" models and the "best" 
#    models are saved.
# 3. While the program runs, calibration can be tracked in TensorBoard. 
# 4. The loss function used is the one given in Palmer and Gorse (2017): 
# 	 https://discovery.ucl.ac.uk/id/eprint/10115864/
# 5. A csv file with a summary of the calibration results is saved.
# 6. Finally, the following results are output, in terms of Average Pricing Error (APE), 
#    as defined e.g. in Schoutens et al. (2003): https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf
#    (p. 10), using the testing dataset:
# 	 a. Simple averaging over all 30 models, i.e. calculating the average price over the 30 NN, for
#       both "best" and "final" models
#    b. Linear regression over all 30 models, as in Palmer and Gorse (2017). The results "without
#       cheating" are calculated using a separate dataset for the regression.
#    c. Regression resuilts of sinh-transformed values.
# 
# NB The code can be run for for both (normalised) prices or implied volatilities. Some additional
# 	 tricks were used in the production of the price and vols dataset.
#    Also note that we fit each combination of NN to a subset of Heston parameter space. In practice
#    this is not an issue, since different calibrated models can be used in each parameter subspace.
#
#    Finally, note that this example is only for investigation, since in practice there is no need 
#    to use ML to price vanilla equity options, even under Heston. The aim is rather to use a similar 
#    approach for the pricing of exotics inside a MC engine of the type used for CVA, which needs to 
#    be very fasr and accurate, as explained in https://www.risk.net/media/download/957931/download
#
# Copy of output:
'''
Loading data...
Done
Processing data...

--------------
mean price:
0.21368496627247693
min price:
0.00645259104750896
std of price:
0.07424894327751709
--------------

Training set has shapes:
(1500001, 7)
(1500001, 1)
Testing set has shapes:
(250000, 7)
(250000, 1)
Regression set has shapes:
(25000, 7)
(25000, 1)

Starting calibrations:
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
Done. Now calculating errors.
APE of average multi model (saved):
0.002405919882511456
APE of average multi model (final):
0.0034857739710212076
Calculating regression results
APE multi model (regression over saved models):
0.0030874270997756576
APE multi model (regression over saved models, sinh-transformed):
0.005158775836133602
APE multi model (regression, non-saved models):
0.002995899672561941
APE multi model (regression, saved models, no cheating):
0.0030693068146884035
APE multi model (regression, saved models, no cheating, sinh-transform):
0.005094802522808573
'''

import tensorflow as tf
import numpy as np
import random as rn

my_seed = 102

np.random.seed(my_seed)
rn.seed(my_seed)
tf.set_random_seed(my_seed)

# setting this to false allows results to be reproducible
Shuffle = False


from sklearn import preprocessing, model_selection
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import pandas as pd


from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# ensure code is running single-threaded, to allow results to be reproducible
session_conf = tf.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
sess = tf.Session(graph = tf.get_default_graph(), config = session_conf)

#K.set_session(sess)

import time 
import datetime as dt
import pickle
import math
import statistics

# Use half data for training and half for testing
test_size = 0.5

# number of models (neural networks)
N_models = 30

# base of the logarithm, to be used in "softplus" function
base_log = 10.0;

lnbase = math.log(base_log)

# We stop the calibration after 15 epochs of non-improving performance
early_stopping_monitor = EarlyStopping(patience=5)

# Which model: Heston or BS
modeltype = 'heston'

# fit to price or vols
calibtype = 'price'
#calibtype = 'vol'

# set to 1 for verbose output to console during fitting
verbosetype = 0

# maximum number of datasets to be used overall in the calibration 
# (both for training and testing). E.g. to use 5000 training sets per 
# model (as in Palmer, Gorse, 2017) we need sets_per_model = 10000
sets_per_model = 50000
MaxN = sets_per_model*N_models + 1

# number of datasets to be used for final test, after model has been calibrated
MaxTestN = 5*sets_per_model

# number of datasets to be used for final regression, to find optimum weights
MaxRegN = 25000

optimizer = 'adamax'
no_epochs =  200

nodes = 64
activation = 'relu'
#activation = 'softplus'

layers_hidden_no = 3

# initial guess for the weights. Seems to give better results than available alternatives
weight_guess = [tf.keras.initializers.glorot_normal(seed=123), 'glorot_normal'] 

# Choice of normalisation, if any. None seems to give the best results
#normalization = 'gaussian'
#normalization = 'l2'
normalization = 'none'

def log(base,x):
	y = np.log(x) / np.log(base)
	return y

# log in base 10 in TensorFlow, where x is a tensor
def tfLog(x):
	y = tf.math.log(x) / lnbase
	return y

# implements softplus function in Python, for given base
def Tsp(x):
	y = log(lnbase,np.pow(lnbase,x) - 1.0)
	return y

# implements softplus inverse function in Python
def Tspinv(x):
	#y = math.log10(np.pow(10.0,x) + 1.0)
	y = log(lnbase,np.pow(lnbase,x) + 1.0)
	return y

# implements softplus function in TensorFlow
def tfTsp(x):
	y = tfLog(tf.pow(lnbase,x) - 1.0)
	return y

# implements softplus inverse function in TensorFlow
def tfTspinv(y):
	x = tfLog(tf.pow(lnbase,y) + 1.0)
	return y

# calculates loss as used in (Palmer and Gorse, 2017)
def loss_PG(y_true, y_pred):
	loss = K.mean(K.abs(y_pred - tfTsp(y_true))) * K.mean(tf.divide(K.abs(tfTspinv(y_pred) - y_true), y_true))
	return loss

# calculates average (relative) error, aka APE (average pricing error)
def calc_APE(y_test, y_pred):
	
	ape = np.zeros(y_test.shape[0])
	N = y_test.shape[1]
	
	for row in range(y_test.shape[0]):
		ape[row] = np.sum(np.absolute(y_test[row,:] - y_pred[row,:])) / np.mean(y_test[row,:]) / N
	return ape

# builds the model with the given characteristics
def build_model(optimizer_fn, loss_fn, nodes, act, layers_hidden_no, kern_init):
	model = tf.keras.models.Sequential()
	# add input layer, with the same number of nodes as there are elements in each row of x_train 
	# (which will contain the training set)
	model.add(layers.Dense(nodes, activation = act,kernel_initializer = kern_init, input_shape = (x_train.shape[1],)))

	# add hidden layers
	for _ in range(layers_hidden_no - 1):
		model.add(layers.Dense(nodes, activation = act,kernel_initializer = kern_init))

	# add output layer
	model.add(layers.Dense(y_train.shape[1], activation = 'linear',kernel_initializer=kern_init))

	model.compile(optimizer = optimizer_fn, loss = loss_fn, metrics = ['mape'])
	return model


# calculates average prediction based on input vector x and models vector models
def predict_multi_model(models, x):
	prediction = 0.0
	no_models = len(models)
	debug = False	

	count = 1 
	for model in models:
		if debug:
			print('Now at model ', count)
			count += 1
		prediction += model.predict(x)
		#print(prediction)

	prediction = prediction / no_models

	return prediction

# creates vector of predicted prices
def predict_vector_price(models, x):
	n = len(x)
	y_pred = np.zeros((n), dtype = np.float64)

	for iter, model in enumerate(models):
		prediction = model.predict(x)
		print(prediction.shape)
		y_pred[iter] = prediction

	return y_pred

# calculates the vector of regression coefficients
def find_regression_weights(models, x, y):
	n_mod = N_models
	n_x = len(x)

	fmatrix = np.zeros((n_x, n_mod), dtype = np.float64)

	col = np.zeros(n_x, dtype = np.float64)
	index = 0
	for model in models:
		col = model.predict(x)
		for col_index in range(len(col)):
			fmatrix[col_index, index] = col[col_index]
		index += 1

	# Calculation of the Moore-Penrose pseudo-inverse
	# Can also use something like
	# weights = np.linalg.inv(np.dot(fmatrix.T, fmatrix)).dot(fmatrix.T).dot(y)

	weights = np.linalg.pinv(fmatrix, 1E-10)
	weights = weights.dot(y)

	return weights

# calculates the predicted price using regression
def predict_weighted_price(models, x, weights):
	if weights is None:
		xr, xc = x.shape
		weights = ones(xr, xc)

	Nmatrix = np.zeros((len(x), weights.shape[0]), dtype = np.float64)
	
	for iter, model in enumerate(models):
		Ncol = model.predict(x)
		Nmatrix[:,iter] = Ncol.reshape((-1,))

	y_out = Nmatrix.dot(weights)
	return y_out


# Main program

loss = loss_PG

# this list will contain all the calibrated models, at the end of the calibration
models_list = []

# if we set save_best_only=True in the ModelCheckpoint callback, then the saved models are the 
# ones from the epoch in which the best validation results are observed, rather than when the ones
# from the last epoch, so these will differ from models_list.
models_list_saved = []


print('')

print('Loading data...')

if modeltype == 'heston':
	X_df = pd.read_csv('volSurf_Submodel_IV.csv')
else:
	X_df = pd.read_csv('BS.csv')


print('Done')
print('Processing data...')


# Create X,y numpy arrays (X_orig and y_orig contain the full data set)
X_orig = np.array(X_df.drop(['Halton', 'vol', 'price'],1))

y_df = X_df[calibtype]
y_orig = np.array(y_df)
y_orig = y_orig.reshape(y_orig.shape[0],1)
print('')
print('--------------')
print('mean price:')
print(np.mean(y_orig))
print('min price:')
print(np.min(y_orig))
print('std of price:')
print(np.std(y_orig))
print('--------------')


# We will take the first MaxN entries
maxN = int(np.minimum(MaxN,X_orig.shape[0]))
maxN = int(np.minimum(MaxN,y_orig.shape[0]))

# used for training and validation
X = X_orig[:maxN]
y = y_orig[:maxN]

# used for final testing
X2 = X_orig[maxN:]
y2 = y_orig[maxN:]

X2 = X2[:MaxTestN]
y2 = y2[:MaxTestN]

# used for regression
Xreg = X_orig[maxN + MaxTestN + 1: ]
yreg = y_orig[maxN + MaxTestN + 1: ]

Xreg = Xreg[ :MaxRegN ]
yreg = yreg[ :MaxRegN ]

print('')
print('Training set has shapes:')
print(X.shape)
print(y.shape)

print('Testing set has shapes:')
print(X2.shape)
print(y2.shape)

print('Regression set has shapes:')
print(Xreg.shape)
print(yreg.shape)
print('')

# partition the data into N_models datasets

bounds = np.arange(0, MaxN + 1, MaxN // N_models)

print('Starting calibrations:')

# the following are used in the output csv file
cols_in = ['Model', 'Sobol', 'Optim', 'normalization',  'Nodes',	'Layers',	'Act-in',	'Weights-guess',	'epochs']
cols_out  = ['MAPE in', 'MAPE out', 'APE in', 'APE out', 'APE in min', 'APE out min', 'APE in final', 
			'min loc in', 'min loc out', 'time']
cols = cols_in + cols_out

df = pd.DataFrame(columns = cols)

for j in range(0, len(bounds)-1 ):
	print(j+1)
	lower = bounds[j]
	upper = bounds[j+1]
	X_j = X[lower:upper]
	y_j = y[lower:upper]
	# Use half data for training and half for testing
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X_j, y_j, test_size=test_size, random_state = 42)
	noHalton = int(x_train.shape[0])

	# unique model name, to be used to track calibration performance in TensorBoard
	NAME = "{}-sobol-{}-calibtype-{}-optim-{}-activ-{}-layers-{}-nodes-{}-guess-{}".format(noHalton, calibtype, optimizer, activation, layers_hidden_no, nodes, weight_guess[1], int(time.time()))

	# create model and TensorBoard
	model_j = build_model(optimizer, loss, nodes, activation, layers_hidden_no, weight_guess[0])
	
	# callbacks
	tensorboard = TensorBoard(log_dir="logsTensorBoard/{}".format(NAME))
	filepath="models/model-100-{0:02d}.hdf5".format(j)
	model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

	# calibrate and save the model 
	time_0 = time.time()
	
	history = model_j.fit(x_train, y_train, epochs = no_epochs, verbose = verbosetype, validation_data = (x_test, y_test), batch_size=32,callbacks=[model_checkpoint, tensorboard], shuffle = Shuffle)
	time_elapsed = time.time() - time_0

	# store the model in model_list
	models_list.append(model_j)

	# load the saved model, with the best calibration
	model = models.load_model(filepath, custom_objects = {'loss_PG': loss_PG})
	models_list_saved.append(model)

	# outputs from train and test test
	y_out_train = model_j.predict(x_train)
	y_out = model_j.predict(x_test)

	# APE (out of sample and in sample)
	ape = calc_APE(y_test, y_out)
	ape_in = calc_APE(y_train, y_out_train)

	# MSE and MAPE (in sample and out of sample)
	# "_in" subscript denotes in sample errors
	mse_score_in, mape_score_in = model_j.evaluate(x_train, y_train, verbose = 0)
	mse_score, mape_score = model_j.evaluate(x_test, y_test, verbose = 0)
	

	apehist = history.history['val_mean_absolute_percentage_error']
	apehist_in = history.history['mean_absolute_percentage_error']
						
	# results to be stored in the CSV file						
	row_in = [j, noHalton, optimizer, normalization,  nodes, layers_hidden_no, activation, weight_guess[1], no_epochs]
	row_out = [mape_score_in, mape_score, np.mean(ape_in), np.mean(ape), np.min(apehist_in), np.min(apehist), 
			 	apehist_in[-1], np.argmin(apehist_in), np.argmin(apehist), time_elapsed]

	row = [row_in + row_out]

						#print(cols)

	dfrow  = pd.DataFrame(row, columns = cols)
	df = df.append(dfrow)	



print('Done. Now calculating errors.')

# calculating APE of final models (simple averaging)
y_out_multi = predict_multi_model(models_list, X2)
ape_multi = calc_APE(y2, y_out_multi)

# calculating APE of saved models (simple averaging)
y_out_multi_saved = predict_multi_model(models_list_saved, X2)
ape_multi_saved = calc_APE(y2, y_out_multi_saved)


print('APE of average multi model (saved):')
ape_multi_best = np.mean(ape_multi_saved)
print(np.mean(ape_multi_saved))

print('APE of average multi model (final):')
ape_multi = np.mean(ape_multi)
print(np.mean(ape_multi))

# perform the regression, first using the testing set ("cheating")
print('Calculating regression results')
w = find_regression_weights(models_list_saved, X2, y2)

# calcuate price using regression over saved models
y_out_reg_saved = predict_weighted_price(models_list_saved, X2, w)
ape_reg_saved = calc_APE(y2, y_out_reg_saved)

print('APE multi model (regression over saved models):')
print(np.mean(ape_reg_saved))


# perform the regression with sinh transform
y2_sinhtr = np.sinh(y2)
w_sinhtr = find_regression_weights(models_list_saved, X2, y2_sinhtr)

# calcuate price using regression over saved models
y_out_reg_saved_sinhtr = predict_weighted_price(models_list_saved, X2, w_sinhtr)
ape_reg_saved_sinhtr = calc_APE(y2_sinhtr, y_out_reg_saved_sinhtr)

print('APE multi model (regression over saved models, sinh-transformed):')
print(np.mean(ape_reg_saved_sinhtr))


# calcuate price using regression over final (non-saved) models
w_ns = find_regression_weights(models_list, X2, y2)

y_out_reg_ns = predict_weighted_price(models_list, X2, w_ns)
ape_reg_ns = calc_APE(y2, y_out_reg_ns)

print('APE multi model (regression, non-saved models):')
print(np.mean(ape_reg_ns))


# Now recalculate regression results using a different data set from the 
# one used for the testing ("no cheating")
w_nc = find_regression_weights(models_list_saved, Xreg, yreg)
y_out_reg_nc = predict_weighted_price(models_list_saved, X2, w_nc)
#print('Calculating APE, saved models, no cheating')
ape_reg_nc = calc_APE(y2, y_out_reg_nc)

print('APE multi model (regression, saved models, no cheating):')
print(np.mean(ape_reg_nc))

# same as above, but with sinh-transform
yreg_sinhtr = np.sinh(yreg)
w_nc_sinhtr = find_regression_weights(models_list_saved, Xreg, yreg_sinhtr)
y_out_reg_nc_sinhtr = predict_weighted_price(models_list_saved, X2, w_nc_sinhtr)
#print('Calculating APE, saved models, no cheating')
ape_reg_nc_sinhtr = calc_APE(y2_sinhtr, y_out_reg_nc_sinhtr)

print('APE multi model (regression, saved models, no cheating, sinh-transform):')
print(np.mean(ape_reg_nc_sinhtr))


df.to_csv('csv/Multimodel_Performance.csv')

