import os
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
import keras_tuner as kt        
from tensorflow import distribute
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

class ANN_Classification():
    os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
    except:
        print('Invalid device or cannot modify virtual devices once initialized.')

    x_train, y_train, x_val, y_val, LB = GetTrainingData(phenotypeFolder+'data.joblib',labelize = True, splitTrainTest = 0.3)
    batchSize = 20
    train_dataset = Dataset.from_tensor_slices((x_train,y_train)).batch(batchSize,drop_remainder=True)
    val_dataset = Dataset.from_tensor_slices((x_val,y_val)).batch(batchSize,drop_remainder=True)

    def build_model( hp = kt.HyperParameters ):
        model = keras.Sequential()
        model.add( keras.layers.Dropout( rate = hp.Choice(name='dropout', values=[0.0,0.15,0.3,0.4])))
        model.add( keras.layers.Dense(units = hp.Choice('1_neurons', values=[200,500,800,1000,2000,5000,10000]),
                                      activation = hp.Choice('1_activation', values=['relu','sigmoid','softmax','softplus',
                                                                                     'softsign','tanh','selu','elu']),
                                      kernel_constraint = keras.constraints.NonNeg(), bias_constraint = keras.constraints.NonNeg(), 
                                      kernel_regularizer = keras.regularizers.l2()))
        model.add( keras.layers.Dense(units=len(LB.classes_),
                                      activation= hp.Choice('output_activation',values = ['softmax','relu','sigmoid'])))
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp.Float("learning_rate", min_value=1e-4,
                                                                                 max_value=1e-1, sampling="log")),
                      loss = hp.Choice('loss',values=['categorical_crossentropy','binary_crossentropy','kullback_leibler_divergence']),
                      metrics=['accuracy'])
        return model

    def ClassWeights( data ):
        from sklearn.utils import class_weight
        data = numpy.argmax(data, axis=1)
        data = class_weight.compute_class_weight(class_weight = 'balanced', classes = numpy.unique(data), y = data)
        data = {i : data[i] for i in range(len(data))}
        return data

    tuner = kt.BayesianOptimization(hypermodel = build_model,
                                    max_trials = 50,
                                    objective = 'val_accuracy',
                                    directory = phenotypeFolder+'TuningTests',
                                    distribution_strategy = distribute.MultiWorkerMirroredStrategy())

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, min_delta=5.0e-2)

    tuner.search(train_dataset, validation_data=val_dataset,
                 callbacks = [early_stop], class_weight = ClassWeights(y_train),
                 use_multiprocessing=True, epochs=200, verbose=2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    Log('Artificial Neural Network')
    Log("Dropout: {:.4f}".format(best_hps.get("dropout")))
    Log("\nNeurons 1st layer: {}".format(best_hps.get("1_neurons")))
    Log("\nActivation 1st layer: {}".format(best_hps.get("1_activation")))
    Log("\nActivation Output layer: {}".format(best_hps.get("output_activation")))
    Log("\nLearning rate: {:.4f}".format(best_hps.get("learning_rate")))
    Log("\nLoss: {}".format(best_hps.get("loss")))

    # Build the model with the optimal hyperparameters and train it to check best epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=200, callbacks=[early_stop], class_weight=ClassWeights(y_train), verbose=2, use_multiprocessing=True)

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