import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load the dataset (in this example, we'll use the Iris dataset)
X, y = load_iris(return_X_y=True)

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the Perceptron model using Keras (MLPClassifier from scikit-learn)
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator

def create_perceptron_model(hidden_layer_size=1):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create a Keras-based Perceptron estimator
class KerasPerceptronEstimator(BaseEstimator):
    def __init__(self, hidden_layer_size=1, epochs=100, batch_size=10):
        self.hidden_layer_size = hidden_layer_size
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        self.model = create_perceptron_model(self.hidden_layer_size)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

# Create a Perceptron estimator
ANN = KerasPerceptronEstimator()

# Define the hyperparameters to search over
param_grid = {
    'hidden_layer_size': [1, 2, 3],  # Number of neurons in the hidden layer
    'epochs': [50, 100, 200],  # Number of training epochs
    'batch_size': [10, 20, 32],  # Mini-batch size
}

# Create a GridSearchCV object for hyperparameter optimization
grid_search = GridSearchCV(estimator=ANN, param_grid=param_grid, cv=3, scoring='accuracy')

# Perform hyperparameter optimization
grid_search.fit(X, y)

# Print the best hyperparameters and their corresponding accuracy
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

#########################################################################################

import os
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
import keras_tuner as kt
from tensorflow import distribute
from tensorflow.keras.callbacks import EarlyStopping

class ANN():
    def set_configurations(self):
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            tf.config.experimental.set_memory_growth(physical_devices[1], True)
        except:
            print('Invalid device or cannot modify virtual devices once initialized.')
        
    def __init__(self, phenotype):
        self.set_configurations()
        
        self.phenotype = phenotype
        if self.phenotype == 'optimum_ph' or self.phenotype == 'optimum_tmp':
            self.classification_or_regression = 'regression'
            label = False
        else:
            self.classification_or_regression = 'classification'
            label = True

        self.dataIO = DataIO( self.phenotype, self.classification_or_regression )
        x_train, y_train, x_test, y_test, labelize = self.dataIO.GetTrainingData('data.joblib',
                                                                                 labelize=label,
                                                                                 splitTrainTest=0.2)
        
        batchSize = 20
        train_dataset = Dataset.from_tensor_slices((x_train,y_train)).batch(batchSize,drop_remainder=True)
        val_dataset = Dataset.from_tensor_slices((x_test,y_test)).batch(batchSize,drop_remainder=True)
        best_config = self.GridSearch(x_train, y_train, x_test, y_test)
        #x_train, y_train, labelize = self.dataIO.GetTrainingData('data.joblib')
        #final_model = self.FinalModel(best_config, x_train, y_train)
        #self.dataIO.SaveModel(final_model, 'lgbm')

    def build_model(self, hp = kt.HyperParameters ):
        model = keras.Sequential()
        model.add( keras.layers.Dropout(rate = hp.Choice(name='dropout', values=[0.0,0.15,0.3,0.5])))
        model.add( keras.layers.Dense(units = hp.Choice('1_neurons', values=[200,500,800,1000,2000,5000,10000]),
                                      activation = hp.Choice('1_activation', values=['relu','sigmoid',
                                                                                     'softmax','softplus',
                                                                                     'softsign','tanh',
                                                                                     'selu','elu']),
                                      kernel_constraint = keras.constraints.NonNeg(), 
                                      bias_constraint = keras.constraints.NonNeg(), 
                                      kernel_regularizer = keras.regularizers.l2()))
        model.add( keras.layers.Dense(units = hp.Choice('2_neurons', values=[200,500,800,1000,2000,5000,10000]),
                                      activation = hp.Choice('2_activation', values=['relu','sigmoid',
                                                                                     'softmax','softplus',
                                                                                     'softsign','tanh',
                                                                                     'selu','elu']),
                                      kernel_constraint = keras.constraints.NonNeg(), 
                                      bias_constraint = keras.constraints.NonNeg(), 
                                      kernel_regularizer = keras.regularizers.l2()))
        model.add( keras.layers.Dense(units=len(labelize.classes_),
                                      activation= hp.Choice('output_activation',
                                                            values = ['relu','sigmoid',
                                                                      'softmax','softplus',
                                                                      'softsign','tanh',
                                                                      'selu','elu'])))
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp.Float("learning_rate", 
                                                                                 min_value=1e-4,
                                                                                 max_value=1e-1, 
                                                                                 sampling="log")),
                      loss = hp.Choice('loss',values=['categorical_crossentropy',
                                                      'binary_crossentropy',
                                                      'kullback_leibler_divergence']),
                      metrics=['accuracy'])
        return model

    def ClassWeights(self, data ):
        from sklearn.utils import class_weight
        data = numpy.argmax(data, axis=1)
        data = class_weight.compute_class_weight(class_weight = 'balanced', classes = numpy.unique(data), y = data)
        data = {i : data[i] for i in range(len(data))}
        return data

    def GridSearch(self, x_train, y_train, x_test, y_test):
        tuner = kt.BayesianOptimization(hypermodel = self.build_model,
                                        max_trials = 50,
                                        objective = 'val_accuracy',
                                        directory = self.dataIO.phenotype_directory+'/tuning_ann',
                                        distribution_strategy = distribute.MultiWorkerMirroredStrategy())

        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, min_delta=5.0e-2)

        tuner.search(train_dataset, validation_data=val_dataset,
                     callbacks = [early_stop], class_weight = ClassWeights(y_train),
                     use_multiprocessing=True, epochs=200, verbose=2)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.dataIO.WriteLog('Artificial Neural Network')
        self.dataIO.WriteLog("Dropout: {:.4f}".format(best_hps.get("dropout")))
        self.dataIO.WriteLog("\nNeurons 1st layer: {}".format(best_hps.get("1_neurons")))
        self.dataIO.WriteLog("\nActivation 1st layer: {}".format(best_hps.get("1_activation")))
        self.dataIO.WriteLog("\nNeurons 2nd layer: {}".format(best_hps.get("2_neurons")))
        self.dataIO.WriteLog("\nActivation 2nd layer: {}".format(best_hps.get("2_activation")))
        self.dataIO.WriteLog("\nActivation Output layer: {}".format(best_hps.get("output_activation")))
        self.dataIO.WriteLog("\nLearning rate: {:.4f}".format(best_hps.get("learning_rate")))
        self.dataIO.WriteLog("\nLoss: {}".format(best_hps.get("loss")))
        
        return best_hps

    def FinalModel(self, config, x, y):
        # Build the model with the optimal hyperparameters and train it to check best epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(train_dataset, validation_data=val_dataset, 
                            epochs=200, 
                            callbacks=[early_stop], 
                            class_weight=ClassWeights(y_train), 
                            verbose=2, 
                            use_multiprocessing=True)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 10
        Log("\nBest epoch: {}".format(best_epoch))

        # Build the model with the optimal hyperparameters and train with best epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(train_dataset, epochs=best_epoch,
                            callbacks = [early_stop], verbose=2,
                            class_weight = ClassWeights(y_train),
                            use_multiprocessing=True)

        model.save(phenotypeFolder+proccessID+'ANN')

        #https://stackoverflow.com/questions/14645684/filtering-a-list-of-lists-in-a-smart-way

        y_pred_tmp = model.predict(x_val)
        y_pred = numpy.zeros_like(y_pred_tmp)
        for i in range(len(y_pred_tmp)):
            y_pred[i][numpy.argmax(y_pred_tmp[i])] = 1

        y_pred = LB.inverse_transform( y_pred )
        y_val = LB.inverse_transform(y_val)
        Metrics(y_val,y_pred)