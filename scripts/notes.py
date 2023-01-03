def PlotModel( yTrue, yPred):
    yPred = LB.inverse_transform(yPred)
    yTrue = LB.inverse_transform(yTrue)
    from sklearn.metrics import classification_report, multilabel_confusion_matrix
    #metrics_report = classification_report(numpy.argmax(yTrue,axis=1),numpy.argmax(yPred,axis=1))
    metrics_report = classification_report(yTrue,yPred)
    #confusion_matrix = multilabel_confusion_matrix(y_val, y_pred)
    with open(log, '+a') as f:
        f.write('\n'+str(metrics_report))
    """
    fpr, tpr, _ = roc_curve(yTrue, yPred)
    auc = roc_auc_score(yTrue, yPred)
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(str(HGS.best_estimator_).split('(')[0]+'.png', dpi=300)
    """

def ChosenModels():
    Log('Weighted Assembly')
    
    @jit
    def PlotIdentity(y_train, y_train_pred, y_test, y_test_pred, name):
        plt.scatter(y_train,y_train_pred, label='Train', c='red')
        plt.scatter(y_test, y_test_pred, label='Test', c='blue')
        plt.xlabel('Predicted value')
        plt.ylabel('True value')
        if os.path.isdir(phenotypeFolder+'graphs/') == False:
            os.mkdir(phenotypeFolder+'graphs/')
        plt.savefig(phenotypeFolder+'graphs/'+name+'.png')
    
    @jit
    def Fit(model, x, y, xtest, ytest):
        model.fit(x,y)
        yTestPred = model.predict(xtest)
        testCOD = r2_score(ytest, yTestPred)
        name = str(model).split(sep='(')[0]
        Log(name)
        Log(regression_report(y_test, yTestPred))
        yTrainPred = model.predict(x)
        PlotIdentity(y, yTrainPred, ytest, yTestPred, name)
        return (yTestPred,testCOD)

    x_train, y_train, x_test, y_test = GetTrainingData(phenotypeFolder+'dataV.joblib', splitTrainTest=0.2)
    Log('Training data: '+str(len(y_train)))
    Log('Testing data: '+str(len(y_test)))
    metrics = []

    from lightgbm import LGBMRegressor
    metrics.append( Fit(LGBMRegressor(num_leaves=100,n_estimators=100,min_child_samples=10,learning_rate=0.1,reg_alpha=0.1, reg_lambda=0.1,min_split_gain=0.1), x_train, y_train, x_test, y_test))
    
    from sklearn.ensemble import RandomForestRegressor
    metrics.append( Fit(RandomForestRegressor(bootstrap='False',ccp_alpha=0.0,max_depth=None,max_features='sqrt',
                                              n_estimators=100,n_jobs=-1), x_train, y_train, x_test, y_test))
    
    from sklearn.cross_decomposition import PLSRegression
    metrics.append( Fit(PLSRegression(n_components=10,max_iter=500,tol = 0.1), x_train, y_train, x_test, y_test))

    yTestPred = [(x*metrics[0][1]+y*metrics[1][1]+w*metrics[2][1])/(metrics[0][1]+metrics[1][1]+metrics[2][1]) 
                  for x,y,w in zip(metrics[0][0],metrics[1][0],metrics[2][0])]

    Log('Weighted')
    Log( regression_report(y_test, yTestPred) )
    
Não funciona:

self.GridSearch(NuSVC(), 
{'nu':[0.02,0.05,0.1,0.2],
'kernel':['linear','poly','rbf','sigmoid'],
'degree':[1,2,3,4,5,6,7,8,9],
'coef0':[0.0, 0.01, 0.1, 1.0],
'tol':[0.001, 0.01, 0.1],
'cache_size':[200000]} )

self.GridSearch(RidgeClassifierCV(),
{'normalize':[True, False],
'class_weight':['balanced'],
'alphas':[0.2,0.5,1.,2.,5.,10.]})


Métricas de regressao
"""
self.WriteLog('Mean absolute error ' + str(mean_absolute_error(y_true, y_pred)))
self.WriteLog('Median absolute error '+ str(median_absolute_error(y_true, y_pred)))
self.WriteLog('Mean squared error '+ str(mean_squared_error(y_true, y_pred)))
self.WriteLog('Max error '+ str(max_error(y_true, y_pred)))
self.WriteLog('Explained variance score '+ str(explained_variance_score(y_true, y_pred)))
error = y_true - y_pred
percentil = [5,25,50,75,95]
percentil_value = numpy.percentile(error, percentil)
for i in range(len(percentil)):
    self.WriteLog('Percentil '+str(percentil[i])+': '+str(percentil_value[i]))
"""