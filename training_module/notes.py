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

REGRESSION

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

#incluir na tese https://www.frontiersin.org/articles/10.3389/fgene.2017.00072/full