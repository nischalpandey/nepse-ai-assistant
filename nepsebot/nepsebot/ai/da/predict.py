# 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import os

from io import BytesIO
import base64



# Load the json data from the file

def getpredictchart():
    json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latestdata/priceNEPSE.json')

    stocks_data = pd.read_json(json_file)

    stocks_data = stocks_data.drop(['s'], axis=1)
    # Remove the columns that are not needed
    # set time as index
    stocks_data = stocks_data.set_index('t')
    # convert the index to datetime
    stocks_data.index = pd.to_datetime(stocks_data.index, unit='s')

    stocks_data.tail(10)

    # 

    # Convert the dataframe to a numpy array
    dataset = stocks_data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    training_data_len

    # 
    # Scale the data because the data is not in the same range
    # It is a good practice to scale the data before feeding it to the model
    # It converts the data into a range of 0 and 1


    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    scaled_data

    # 
    # Create the training data set 
    # Create the scaled training data set 
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()
            
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_train.shape

    # Build the Regression model from sklearn


    # Split the data into 80% training and 20% testing data sets
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    # Create and train the model
    model = LinearRegression()
    model.fit(x_train, y_train )

    # Get the models predicted price values

    predictions = model.predict(x_test)
    predictions

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    rmse

    # Plot the data
    train = stocks_data[:training_data_len]
    valid = stocks_data[training_data_len:]
    valid['Predictions'] = predictions[:len(valid)]
    print(valid)


    plt.figure(figsize=(10,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price NPR', fontsize=18)
    plt.plot(train['c'])
    plt.plot(valid[['c', 'Predictions']])
    plt.legend(['Train',  'Predictions'], loc='lower right')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic1 = base64.b64encode(image_png).decode('utf-8')

    # Compare the actual data with the predicted data
    plt.figure(figsize=(10,6))
    plt.title('Actual Price')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price NPR', fontsize=18)
    stocks_data['c'].plot()

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic2 = base64.b64encode(image_png).decode('utf-8')

    return f"Here is the predicted chart for the stock price:\n<img src='data:image/png;base64,{graphic1}'/>\n\nHere is the actual price chart:\n<img src='data:image/png;base64,{graphic2}'/> \n\n The RMSE value is {rmse} \n\n The model is trained on 95% of the data and tested on 5% of the data \n\n It is a simple linear regression model trained on the closing price of the stock"




