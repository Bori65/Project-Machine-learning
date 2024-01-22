
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)

df = pd.read_csv('house_data.csv')


df = df.drop(['date','street','city','statezip','country'],axis=1)
df = df[df['price'] <= 10000000]
df = df[df['sqft_lot'] <= 50000]

results = {}
for target in df.keys():

    x = df.drop([target],axis=1).values[:-4]
    y = df[target].values[:-4]
    to_predict_x = df.drop([target],axis=1).values[-1]
    to_predict_y = df[target].values[-1]
    print("to predict data: ",to_predict_x, to_predict_y)

    X_train,X_test,y_train,y_test= train_test_split(x , y , test_size=0.2, random_state=0)
    print('x_train', X_train.shape)
    print('x_test', X_test.shape)
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)

    sc_x = preprocessing.StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    history =  model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    # Evaluate the model on the test set
    mse = model.evaluate(X_test, y_test)
    print("Target:", target)
    print(f'Mean Squared Error on Test Set: {mse}')
    print(history.history.keys())
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    title = "house prices dataset, column: " + target
    plt.title(title)
    plt.legend()
    plt.savefig('House_loss\stellar_'+ target +'.png')
    plt.show()

    new_data_scaled = sc_x.transform(to_predict_x.reshape(1, -1))
    predicted_output_scaled = model.predict(new_data_scaled)
    print("predicted_output", predicted_output_scaled)

    results[target] = mse

with open('house_results.txt', 'w') as file:
    # Write header
    file.write("Target\tMSE\n")

    # Write data
    for target, mse in results.items():
        file.write(f"{target}\t{mse}\n")