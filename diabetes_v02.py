import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import neighbors
from sklearn import tree
from sklearn import neural_network
import joblib

# read CSV file to pandas dataframe
# remove title
df = pd.read_csv("diabetes.csv")
df.dropna(axis=1, inplace=True)
df.head(9)


# define the function to normnization the numric data
def norm_x(item):
    X = np.array(df[item]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    df[item] = X_scaled.reshape(1, -1)[0]


# the max and min value of
# "preg", "plas", "pres", "skin","insu", "mass", "pedi", "age"
preg_max = df["preg"].max()
preg_min = df["preg"].min()
plas_max = df["plas"].max()
plas_min = df["plas"].min()
pres_max = df["pres"].max()
pres_min = df["pres"].min()
skin_max = df["skin"].max()
skin_min = df["skin"].min()
insu_max = df["insu"].max()
insu_min = df["insu"].min()
mass_max = df["mass"].max()
mass_min = df["mass"].min()
pedi_max = df["pedi"].max()
pedi_min = df["pedi"].min()
age_max = df["age"].max()
age_min = df["age"].min()

# normnize the Xs
norm_x("preg")
norm_x("plas")
norm_x("pres")
norm_x("skin")
norm_x("insu")
norm_x("mass")
norm_x("pedi")
norm_x("age")

# one-hot-encoder to numric Y from class
encoder = OneHotEncoder(handle_unknown="ignore")
encoder_df = pd.DataFrame(encoder.fit_transform(df[["class"]]).toarray())

# put the processed data into a new CSV file
df = df.join(encoder_df)
df.drop("class", axis=1, inplace=True)
df.drop(0, axis=1, inplace=True)
df.columns = ["preg", "plas", "pres", "skin",
              "insu", "mass", "pedi", "age", "diabete"]
df.to_csv("processed_data.csv", index=False)

# Training model with processed data
df = pd.read_csv("processed_data.csv")
X = df[["preg", "plas", "pres", "skin",
        "insu", "mass", "pedi", "age"]]
y = df["diabete"]
# training set (88%) test set (12%)
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.12)
# Create the Linear Regression model
model = tree.DecisionTreeClassifier()
# model = neural_network.MLPClassifier(hidden_layer_sizes=(
#     50, ), max_iter=1000, alpha=1e-8, solver='sgd', verbose=1, tol=1e-8, random_state=1, learning_rate_init=.01)
# model = neural_network.MLPClassifier(hidden_layer_sizes=(
#     50, 50, 50, 50, 50, ), max_iter=1000, alpha=1e-8, solver='sgd', verbose=1, tol=1e-8, random_state=1, learning_rate_init=.01)
# model = neighbors.KNeighborsRegressor(3)
# model = LinearRegression()
# Train the model and save the trained model to a file
model.fit(X_train, y_train)
joblib.dump(model, "diabete_diagnose_model.pkl")
# Report an error rate on the training set
print("Model training results:")
mse_train = mean_absolute_error(y_train, model.predict(X_train))
print(f" - Training Set Error: {mse_train}")
# Report an error rate on the test set
mse_test = mean_absolute_error(y_test, model.predict(
    X_test))
print(f" - Test Set Error: {mse_test}")

# normnize patient information


def nom_input(item, item_min, item_max):
    return (float(item) - item_min) / (item_max - item_min)


user_choice = "y"
while user_choice.lower() != "n":
    if user_choice.lower() == "y":
        # input patient information
        print("preg range\t" + str(preg_min)+" to " + str(preg_max))
        preg = input("Put patient preg number : ")
        preg_nom = nom_input(preg, preg_min, preg_max)
        print("plas range\t" + str(plas_min)+" to " + str(plas_max))
        plas = input("Put patient plas number : ")
        plas_nom = nom_input(plas, plas_min, plas_max)
        print("pres range\t" + str(pres_min)+" to " + str(pres_max))
        pres = input("Put patient pres number : ")
        pres_nom = nom_input(pres, pres_min, pres_max)
        print("skin range\t" + str(skin_min)+" to " + str(skin_max))
        skin = input("Put patient skin number : ")
        skin_nom = nom_input(skin, skin_min, skin_max)
        print("insu range\t" + str(insu_min)+" to " + str(insu_max))
        insu = input("Put patient insu number : ")
        insu_nom = nom_input(insu, insu_min, insu_max)
        print("mass range\t" + str(mass_min)+" to " + str(mass_max))
        mass = input("Put patient mass number : ")
        mass_nom = nom_input(mass, mass_min, mass_max)
        print("pedi range\t" + str(pedi_min)+" to " + str(pedi_max))
        pedi = input("Put patient pedi number : ")
        pedi_nom = nom_input(pedi, pedi_min, pedi_max)
        print("age range\t" + str(age_min)+" to " + str(age_max))
        age = input("Put patient age number : ")
        age_nom = nom_input(age, age_min, age_max)

        # Load our trained mode
        model = joblib.load("diabete_diagnose_model.pkl")
        patient_input = [preg_nom, plas_nom, pres_nom, skin_nom,
                         insu_nom, mass_nom, pedi_nom, age_nom]
        # scikit-learn assumes you want to predict the result for multiple patients at once, so it expects an array.
        # We only want to estimate the value of a patient, so there will only be one item in our array.
        patient_list = [
            patient_input
        ]

        # Make a prediction for each patient
        patient_out = model.predict(patient_list)

        # Since we are only predicting the price of one house, grab the first prediction returned
        predicted_result = patient_out[0]

        # Print the results
        if predicted_result > 0.5:
            print(predicted_result, "Diabetie Positive")
        else:
            print(predicted_result, "Diabetie Nagetive")

    user_choice = input("\nDo the research again? (y/n)")
