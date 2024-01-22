
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)

df = pd.read_csv('star_classification.csv')
LE = preprocessing.LabelEncoder()
df['class'] = LE.fit_transform(df['class'])


df = df.drop(['obj_ID','run_ID','field_ID','spec_obj_ID','fiber_ID'],axis=1)
results = {}
for target in df.keys():
    x = df.drop([target],axis=1).values[:10000]
    y = df[target].values[:10000]
    X_train,X_test,y_train,y_test= train_test_split(x , y , test_size=0.2, random_state=0)

    print('x_train', X_train.shape)
    print('x_test', X_test.shape)
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)

    sc = preprocessing.StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
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
    title = "stellar dataset, target: " + target
    plt.title(title)
    plt.legend()
    plt.savefig('Stallar_loss\stellar_'+ target +'.png')
    plt.show()
    results[target] = mse

with open('stellar_results.txt', 'w') as file:
    # Write header
    file.write("Target\tMSE\n")

    # Write data
    for target, mse in results.items():
        file.write(f"{target}\t{mse}\n")