import pandas as pd
import statsmodels.api as sm


def VAR(data, maxlags):
    '''Vector Auto-Regressive (VAR) model fit.
    
    Fits multivariate data and returns:
    results = VAR fit
    forecast = VAR forecast for a specified number of steps
    '''

    model = sm.tsa.VAR(data)
    results = model.fit(maxlags)
    
    lag_order = results.k_ar
    forecast = results.forecast(data.values[-lag_order:], forecast_steps)
    
    return results, forecast


def load_data(filename, num_features):
    '''Load csv data into a pandas dataframe.'''

    feature_names = []
    
    i = 0
    
    while i < num_features:
        feature_names.append('feat' + str(i))
        i += 1
    
    data = pd.read_csv(filename, names=(feature_names))

    return data


if __name__ == '__main__':
    
    # Choose data file and specify number of features
    filename = 'test_oscill.csv'
    num_features = 3

    # Set VAR parameters
    maxlags = 10
    forecast_steps = 60

    # Load data and setup VAR model
    data = load_data(filename, num_features)
    
    # Run VAR and plot data with forecast
    results, forecast = VAR(data, maxlags)
    results.plot_forecast(forecast_steps)
