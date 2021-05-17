from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

class CaMeliaModel():
    def __init__(self, dropnans=False, learning_rate=0.1, max_depth=7, verbose=100,
                 eval_metric='AUC', device='GPU'):
        self.dropnans = dropnans
        self.model = CatBoostClassifier(learning_rate=learning_rate, max_depth=max_depth,
                                        verbose=verbose, eval_metric=eval_metric
                                        task_type=device, cat_features=['DNA'+str(i) for i in range(20)])
        
    def fit(self, X_train, y_train):    
        if self.dropnans:
            drop = pd.isnull(X_train).values.sum(-1) == 0
            X_train = X_train.iloc[drop]
            y_train = y_train[drop]
            
        self.model.fit(X_train, y_train)
        
    def test(self, X_test, y_test, pos_test, save_location):
        if self.dropnans:
            drop = pd.isnull(X_test).values.sum(-1) == 0
            X_test = X_test.iloc[drop]
            y_test = y_test[drop]
            pos_test = pos_test[drop]
        
        y_pred = self.model.predict_proba(X_test)[:,1]     
        outs = np.stack([y_pred, y_test])
        with open(save_location, 'wb') as f:
            np.savez_compressed(f, outs, pos_test)
    
    def save_model(self, save_location):
        self.model.save_model(save_location)