import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sklearn.metrics

def linear_regression(data, params):
    epochs = 1000
    for i in range(epochs):
        rmse = calculate_current_rmse(data,params)
        calculate_new_params(data, params, rmse)

def calculate_new_params(data, params, rmse):
    coefs = [np.size(data)/5, data.x, data.x2, data.x3, data.x4]
    lr = 0.0001
    for i in range(len(params)):
        params[i] = params[i] - (lr * 1/(2 * coefs[0]) * 1/(rmse ** 2) * np.sum(coefs [i])) 
    
def load_data_into_matrix():
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
    train = pd.read_csv("train.csv")

    df = pd.DataFrame({
    'x': np.array(train.Width),
    'x2': np.array(train.Width) ** 2,
    'x3': np.array(train.Width) ** 3,
    'x4': np.array(train.Width) ** 4,
    'y': np.array(train.Weight)
    })

    return df
    
def calculate_current_rmse(data, params):
    prediction = np.empty(len(data))
    for index, row in data.iterrows():
        prediction[index] = predict(row, params)

    rmse = sklearn.metrics.mean_squared_error(data.y, prediction)
    # rmse = (1/2*len(data))*np.sum((prediction - data.y)**2)
    rmse = math.sqrt(rmse)
    return rmse

def plot_data(data, params):
    x = data.x
    y = data.y
    X_series = pd.Series(x, name='Width')
    Y_series = pd.Series(y, name='Weight')
    df = X_series.to_frame().join(Y_series)
    df.plot.scatter(x="Width", y="Weight")
    plt.xlabel("Width")
    plt.ylabel("Weight")
    x = np.sort(x)
    plt.plot(x, params[4] * np.power(x,4) + params[3] * np.power(x, 3) + params[2] * np.power(x, 2) + params[1] * np.power(x, 1)+ params[0])
    plt.show()

def predict(row, params):
    return params[0] + params[1] * row['x'] + params[2] * row['x2'] + params[3]*row['x3'] + params[4] * row['x4']

def main():
    data = load_data_into_matrix()
    params = [3, 0.48853934683599044, 0.48264918249185674, 0.4495971437597263, 0.32691368114396576]             

    rmse_start = calculate_current_rmse(data, params)
    print(rmse_start)
    linear_regression(data, params)
    rmse_end = calculate_current_rmse(data, params)    
    print(rmse_end)
    print(params)
    
    plot_data(data, params)

if __name__ == "__main__":
    main()


    

