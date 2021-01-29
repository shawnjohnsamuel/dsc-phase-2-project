import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import eli5

metrics = pd.DataFrame(columns = ['Model Name', 'R2', 'MAE', 'RMSE']) #df to keep track of metrics

def model_analysis(name, model, df, continuous, categoricals, log=True, OHE=True, scale=True, scaler=MinMaxScaler(), seed=42):
    '''
    Performs a train-test split & evaluates a model
    
    Returns: model, X, y, X_train, y_train, y_train_pred, y_test_pred
   
    --
   
    Inputs:
     - name - string, name describing model
     - model - Instantiated sklearn model
     - df - pandas dataframe, containing all independent variables & target
     - continuous - list of all continuous independent variables
     - categoricals - list of all categorical independant variables
     - log - boolean, whether continuous variables should be logged 
     - OHE - boolean, whether categorical variables should be One Hot Encoded
     - scale - boolean, whether to scale the data with a MinMax Scaler
     - scaler - set to MinMaxScaler as default
     - seed - integer, for the random_state of the train test split

    Outputs:
     - R2, MAE and RMSE for training and test sets
     - Scatter plot of risiduals from training and test sets
     - Stats model summary of model
     - Metrics Dataframe which lists R2, Mean Absolute Error & Root Mean Square. 
       (This df will be listed in reverse index, with latest results at the top)
        
    Returns:
     - model - fit sklearn model
     - X, y, X_train, y_train - if needed for OLS
     - y_train_preds - predictions for the training set
     - y_test_preds - predictions for the test set
    '''
    preprocessed = df.copy()
    
    if log == True:
        pp_cont = preprocessed[continuous]
        log_names = [f'{column}_log' for column in pp_cont.columns]
        pp_log = np.log(pp_cont)    
        pp_log.columns = log_names
        preprocessed.drop(columns = continuous, inplace = True)
        preprocessed = pd.concat([preprocessed['price'], pp_log, preprocessed[categoricals]], axis = 1)
    else:
        preprocessed = pd.concat([preprocessed['price'], preprocessed[continuous], preprocessed[categoricals]], axis = 1)
        
    if OHE == True:
        preprocessed = pd.get_dummies(preprocessed, prefix = categoricals, columns = categoricals, drop_first=True)
 
    # define X and y       
    X_cols = [c for c in preprocessed.columns.to_list() if c not in ['price']]
    X = preprocessed[X_cols]
    y = preprocessed.price
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                    random_state=seed)
    
    # create copies for OLS
    X_train_copy = X_train.copy()
    
    # Min Max Scaler
    if scale==True:
        scaler = scaler
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    #fit model
    model.fit(X_train, y_train)

    #predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
        
    #calculate residuals
    residuals_train = y_train_pred - y_train
    residuals_test = y_test_pred - y_test
    
    #print train and test R2, MAE, RMSE
    print(f"Train R2: {r2_score(y_train, y_train_pred):.3f}")
    print(f"Test R2: {r2_score(y_test, y_test_pred):.3f}")
    print("---")
    print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
    print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
    print("---")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
                    
    #risduals plot training and test predictions
    plt.figure(figsize=(7,5))
    plt.scatter(y_train_pred, residuals_train, alpha=.75, label = "Train")
    plt.scatter(y_test_pred, residuals_test, color='g', alpha=.75, label = "Test")
    plt.axhline(y=0, color='black')
    plt.legend()
    plt.title(f'Residuals for {name}')
    plt.ylabel('Residuals')
    plt.xlabel('Predicted Values')
    plt.show()
    
    # display feature weights using ELI5
    display(eli5.show_weights(model, feature_names=list(X.columns)))
        
    #add name, metrics and description to new row
    new_row = []
    new_row.append(name)
    new_row.append(format(r2_score(y_test, y_test_pred),'.3f'))
    new_row.append(format(mean_absolute_error(y_test, y_test_pred),'.2f'))
    new_row.append(format(np.sqrt(mean_squared_error(y_test, y_test_pred)),'.2f'))
    
    #add the row to metrics df and list in reverse order so current is at top
    metrics.loc[len(metrics.index)] = new_row
    display(metrics.sort_index(ascending=False, axis=0))

    return model, X, y, X_train_copy, y_train, y_train_pred, y_test_pred