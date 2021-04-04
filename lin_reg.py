import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import math
import sys

def linear_regression(data, params):
    epochs = 1
    for i in range(epochs):
        rmse = calculate_current_rmse(data,params)
        calculate_new_params(data, params, rmse)

def normal_eq(data_x, data_y):
    data_x_transpose = np.transpose(data_x)
    data_x_transpose_dot_data_x = data_x_transpose.dot(data_x)
    temp_1 = np.linalg.inv(data_x_transpose_dot_data_x)
    temp_2 = data_x_transpose.dot(data_y)
    theta = temp_1.dot(temp_2)
    return theta

def calculate_new_params(data, params, rmse):
    coefs = [np.size(data)/5, data.x, data.x2, data.x3]
    lr = 0.0001
    for i in range(len(params)):
        params[i] = params[i] - (lr * 1/(2 * coefs[0]) * 1/(rmse ** 2) * np.sum(coefs [i])) 
    
def load_data_into_matrix(path):
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
    train = pd.read_csv(path)

    df = pd.DataFrame({
    '1': np.ones(len(train.Width)),   
    'x': np.array(train.Width),
    'x2': np.array(train.Width) ** 2,
    'x3': np.array(train.Width) ** 3,
    'y': np.array(train.Weight)
    })

    return df
    
def calculate_current_rmse(data, params):
    prediction = np.empty(len(data))
    for index, row in data.iterrows():
        prediction[index] = predict(row, params)    

    rmse = (1/len(data.y))*np.sum((np.subtract(prediction, data.y))**2)
    rmse = math.sqrt(rmse)
    return rmse

# def plot_data(data, params):
#     x = data.x
#     y = data.y
#     X_series = pd.Series(x, name='Width')
#     Y_series = pd.Series(y, name='Weight')
#     df = X_series.to_frame().join(Y_series)
#     df.plot.scatter(x="Width", y="Weight")
#     plt.xlabel("Width")
#     plt.ylabel("Weight")
#     x = np.sort(x)
#     plt.plot(x, params[3] * np.power(x, 3) + params[2] * np.power(x, 2) + params[1] * np.power(x, 1)+ params[0])
#     plt.show()

def predict(row, params):
    return params[0] + params[1] * row['x'] + params[2] * row['x2'] + params[3]*row['x3']

def main(train_path, test_path):
    data_train = load_data_into_matrix(train_path)
    data_test = load_data_into_matrix(test_path)
             
    params = normal_eq(data_train.iloc[:, :4], data_train.iloc[:, 4:5])

    rmse_end = calculate_current_rmse(data_test, params)    
    print(rmse_end)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])


    

