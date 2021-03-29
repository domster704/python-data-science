import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import losses_utils


# TODO: make new type of neuronet with new file.csv (new scaling data and cost per one square)
#  change below params
BATCH_SIZE = 21
PATH = 'model/X_model/X_model_v1'
HIDDEN_UNITS = [13, 780]
STEPS = 50000
EPOCHS = 2000
OPTIMIZER = "Adagrad"  # ('Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD')
ACTIVATION_FN = "relu"

# https://www.tensorflow.org/api_docs/python/tf/keras/activations?hl=ru
# deserialize: Returns activation function given a string identifier.
# elu: Exponential Linear Unit.
# exponential: Exponential activation function.
# gelu: Applies the Gaussian error linear unit (GELU) activation function.
# get: Returns function.
# hard_sigmoid: Hard sigmoid activation function.
# linear: Linear activation function (pass-through).
# relu: Applies the rectified linear unit activation function.
# selu: Scaled Exponential Linear Unit (SELU).
# serialize: Returns the string identifier of an activation function.
# sigmoid: Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)).
# softmax: Softmax converts a real vector to a vector of categorical probabilities.
# softplus: Softplus activation function, softplus(x) = log(exp(x) + 1).
# softsign: Softsign activation function, softsign(x) = x / (abs(x) + 1).
# swish: Swish activation function, swish(x) = x * sigmoid(x).
# tanh: Hyperbolic tangent activation function.

LOSS_REDUCTION = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

# https://www.tensorflow.org/api_docs/python/tf/keras/losses?hl=ru
# deserialize(...): Returns activation function given a string identifier.
# elu(...): Exponential Linear Unit.
# exponential(...): Exponential activation function.
# gelu(...): Applies the Gaussian error linear unit (GELU) activation function.
# get(...): Returns function.
# hard_sigmoid(...): Hard sigmoid activation function.
# linear(...): Linear activation function (pass-through).
# relu(...): Applies the rectified linear unit activation function.
# selu(...): Scaled Exponential Linear Unit (SELU).
# serialize(...): Returns the string identifier of an activation function.
# sigmoid(...): Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)).
# softmax(...): Softmax converts a real vector to a vector of categorical probabilities.
# softplus(...): Softplus activation function, softplus(x) = log(exp(x) + 1).
# softsign(...): Softsign activation function, softsign(x) = x / (abs(x) + 1).
# swish(...): Swish activation function, swish(x) = x * sigmoid(x).
# tanh(...): Hyperbolic tangent activation function.


def createModelForAllData():
	data = pd.read_csv(r'results/obhiy.csv')
	data = data.dropna(axis=0)

	x_data = data.drop(data.columns[[2]], axis=1)  # все столбцы кроме цены и не числового
	y_data = data['Cost']  # цена

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=101)

	scaler = MinMaxScaler()
	scaler.fit(x_train)

	area = tf.feature_column.numeric_column('Area')
	distance = tf.feature_column.numeric_column('DistanceToCity')
	ecology = tf.feature_column.numeric_column('Ecology')
	purity = tf.feature_column.numeric_column('Purity')
	utilities = tf.feature_column.numeric_column('Utilities')
	neighbors = tf.feature_column.numeric_column('Neighbors')
	children = tf.feature_column.numeric_column('Children')
	relax = tf.feature_column.numeric_column('SportsAndRecreation')
	shops = tf.feature_column.numeric_column('Shops')
	transports = tf.feature_column.numeric_column('Transport')
	security = tf.feature_column.numeric_column('Safety')
	lifecost = tf.feature_column.numeric_column('LifeCost')
	city = tf.feature_column.numeric_column('City')

	feat_cols = [area, distance, ecology, purity, utilities, neighbors, children, relax, shops, transports, security,
				 lifecost, city]

	input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=BATCH_SIZE,
															   num_epochs=EPOCHS, shuffle=True)

	model = tf.estimator.DNNRegressor(hidden_units=HIDDEN_UNITS, feature_columns=feat_cols, model_dir=PATH,
									  optimizer=OPTIMIZER, activation_fn=ACTIVATION_FN, loss_reduction=LOSS_REDUCTION)
	model.train(input_fn=input_func, steps=STEPS)

	predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=BATCH_SIZE,
																	   num_epochs=1, shuffle=False)
	pred_gen = model.predict(predict_input_func)
	predictions = list(pred_gen)

	final_y_preds = []
	for pred in predictions:
		final_y_preds.append(pred['predictions'])

	print(final_y_preds[0:30])


def create_model(a):
	data = pd.read_csv(r'results/obhiy.csv')
	data = data.dropna(axis=0)

	x_data = data.drop(data.columns[[2]], axis=1)  # все столбцы кроме цены и не числового
	y_data = data['Cost']  # цена

	ddt = pd.DataFrame({
		'Area': [a[0], a[0]],
		'DistanceToCity': [a[1], a[1]],
		'Ecology': [a[2], a[2]],
		'Purity': [a[3], a[3]],
		'Utilities': [a[4], a[4]],
		'Neighbors': [a[5], a[5]],
		'Children': [a[6], a[6]],
		'SportsAndRecreation': [a[7], a[7]],
		'Shops': [a[8], a[8]],
		'Transport': [a[9], a[9]],
		'Safety': [a[10], a[10]],
		'LifeCost': [a[11], a[11]],
		'City': [a[12], a[12]]})

	x_train, test_data, y_train, test_labels = train_test_split(x_data, y_data, test_size=0.2, random_state=101)

	scaler = MinMaxScaler()
	scaler.fit(x_train)
	x_test = pd.DataFrame(data=scaler.transform(ddt),
						  columns=ddt.columns, index=ddt.index)
	y_test = pd.Series([float(a[2]), float(a[2])])

	area = tf.feature_column.numeric_column('Area')
	distance = tf.feature_column.numeric_column('DistanceToCity')
	ecology = tf.feature_column.numeric_column('Ecology')
	purity = tf.feature_column.numeric_column('Purity')
	utilities = tf.feature_column.numeric_column('Utilities')
	neighbors = tf.feature_column.numeric_column('Neighbors')
	children = tf.feature_column.numeric_column('Children')
	relax = tf.feature_column.numeric_column('SportsAndRecreation')
	shops = tf.feature_column.numeric_column('Shops')
	transports = tf.feature_column.numeric_column('Transport')
	security = tf.feature_column.numeric_column('Safety')
	lifecost = tf.feature_column.numeric_column('LifeCost')
	city = tf.feature_column.numeric_column('City')

	feat_cols = [area, distance, ecology, purity, utilities, neighbors, children, relax, shops, transports, security,
				 lifecost, city]

	input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=BATCH_SIZE,
															   num_epochs=EPOCHS, shuffle=True)

	model = tf.estimator.DNNRegressor(hidden_units=HIDDEN_UNITS, feature_columns=feat_cols, model_dir=PATH,
									  optimizer=OPTIMIZER, activation_fn=ACTIVATION_FN, loss_reduction=LOSS_REDUCTION)
	model.train(input_fn=input_func, steps=STEPS)

	[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
	print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))
	print("Loss: " + loss)

	predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=BATCH_SIZE,
																	   num_epochs=1, shuffle=False)
	pred_gen = model.predict(predict_input_func)
	predictions = list(pred_gen)

	final_y_preds = []
	for pred in predictions:
		final_y_preds.append(pred['predictions'])


# return final_y_preds[0]


def getDataFromReadyNeural(a):
	data = pd.read_csv(r'results/obhiy.csv')
	data = data.dropna(axis=0)

	x_data = data.drop(data.columns[[2]], axis=1)  # все столбцы кроме цены и не числового
	y_data = data['Cost']  # цена

	if len(a) == 13:
		ddt = pd.DataFrame({
			'Area': [a[0], a[0]],
			'DistanceToCity': [a[1], a[1]],
			'Ecology': [a[2], a[2]],
			'Purity': [a[3], a[3]],
			'Utilities': [a[4], a[4]],
			'Neighbors': [a[5], a[5]],
			'Children': [a[6], a[6]],
			'SportsAndRecreation': [a[7], a[7]],
			'Shops': [a[8], a[8]],
			'Transport': [a[9], a[9]],
			'Safety': [a[10], a[10]],
			'LifeCost': [a[11], a[11]],
			'City': [a[12], a[12]]})
	else:
		y_test = pd.Series([a[2], a[2]])
		ddt = pd.DataFrame({
			'Area': [a[0], a[0]],
			'DistanceToCity': [a[1], a[1]],
			'Ecology': [a[3], a[3]],
			'Purity': [a[4], a[4]],
			'Utilities': [a[5], a[5]],
			'Neighbors': [a[6], a[6]],
			'Children': [a[7], a[7]],
			'SportsAndRecreation': [a[8], a[8]],
			'Shops': [a[9], a[9]],
			'Transport': [a[10], a[10]],
			'Safety': [a[11], a[11]],
			'LifeCost': [a[12], a[12]],
			'City': [a[13], a[13]]})

	x_train, _, y_train, __ = train_test_split(x_data, y_data, test_size=0.2, random_state=101)

	scaler = MinMaxScaler()
	scaler.fit(x_train)
	x_test = pd.DataFrame(data=scaler.transform(ddt),
						  columns=ddt.columns, index=ddt.index)
	y_test = pd.Series([float(a[2]), float(a[2])])

	area = tf.feature_column.numeric_column('Area')
	distance = tf.feature_column.numeric_column('DistanceToCity')
	ecology = tf.feature_column.numeric_column('Ecology')
	purity = tf.feature_column.numeric_column('Purity')
	utilities = tf.feature_column.numeric_column('Utilities')
	neighbors = tf.feature_column.numeric_column('Neighbors')
	children = tf.feature_column.numeric_column('Children')
	relax = tf.feature_column.numeric_column('SportsAndRecreation')
	shops = tf.feature_column.numeric_column('Shops')
	transports = tf.feature_column.numeric_column('Transport')
	security = tf.feature_column.numeric_column('Safety')
	lifecost = tf.feature_column.numeric_column('LifeCost')
	city = tf.feature_column.numeric_column('City')

	feat_cols = [area, distance, ecology, purity, utilities, neighbors, children, relax, shops, transports, security,
				 lifecost, city]

	model = tf.estimator.DNNRegressor(hidden_units=HIDDEN_UNITS, feature_columns=feat_cols, model_dir=PATH,
									  optimizer=OPTIMIZER, activation_fn=ACTIVATION_FN, loss_reduction=LOSS_REDUCTION)

	predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, batch_size=BATCH_SIZE,
																	   num_epochs=1, shuffle=False)
	pred_gen = model.predict(predict_input_func)
	predictions = list(pred_gen)

	final_y_preds = []
	for pred in predictions:
		final_y_preds.append(pred['predictions'])

	return final_y_preds[0][0] ** 2


if __name__ == "__main__":

	l = [
		[16.0, 0.0, 3000000.0, 3.3, 3.1, 2.7, 3.7, 3.4, 3.0, 4.2, 2.9, 3.4, 2.1, 1.0],
		[10.0, 4.0, 525000.0, 3.3, 3.1, 2.7, 3.7, 3.4, 3.0, 4.2, 2.9, 3.4, 3.1, 81],
		[1000.0, 4.0, 525000.0, 3.3, 3.1, 2.7, 3.7, 3.4, 3.0, 4.2, 2.9, 3.4, 3.1, 81],
		[7.0, 10.0, 600000.0, 3.4, 3.2, 3.0, 3.7, 3.5, 3.1, 4.2, 3.2, 3.5, 2.3, 49],
		[5.0, 3.0, 500000.0, 3.1, 2.9, 2.7, 3.6, 3.3, 3.0, 4.2, 2.4, 3.1, 2.3, 81],
		[15.0, 98.0, 850000.0, 1.4, 1.6, 1.9, 3.5, 2.8, 2.4, 3.9, 2.4, 2.7, 1.6, 81],
	]

	# createModelForAllData()

	array = [
		# [create_model(l[0]), l[0][2]],
		[getDataFromReadyNeural(l[1]), l[0][2]],
		# [getDataFromReadyNeural(l[2]), l[2][2]],
		# [getDataFromReadyNeural(l[3]), l[3][2]],
		# [getDataFromReadyNeural(l[4]), l[4][2]],
	]

	for i in range(len(array)):
		print(array[i], "Разница: ", abs(array[i][0] - l[i + 1][2]))
