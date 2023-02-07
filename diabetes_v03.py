import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import tkinter as tk

# part 1
# read CSV file and convert to data ready for training
df = pd.read_csv("diabetes.csv")
df.dropna(axis=1, inplace=True)
df.head(9)

# normalize the numric data


def norm_x(item):
    X = np.array(df[item]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    df[item] = X_scaled.reshape(1, -1)[0]


def nom_input(item, item_min, item_max):
    return (float(item) - item_min) / (item_max - item_min)


# the max and min value
# to normalize user input data
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

# one-hot-encoder to convert Y
encoder = OneHotEncoder(handle_unknown="ignore")
encoder_df = pd.DataFrame(encoder.fit_transform(df[["class"]]).toarray())

# put the processed data and new titles into a new CSV file
df = df.join(encoder_df)
df.drop("class", axis=1, inplace=True)
df.drop(0, axis=1, inplace=True)
df.columns = ["preg", "plas", "pres", "skin",
              "insu", "mass", "pedi", "age", "diabete"]
df.to_csv("processed_data.csv", index=False)


# part 2
# Training model with processed data
model = LinearRegression()  # set default model


def training():
    df = pd.read_csv("processed_data.csv")
    X = df[["preg", "plas", "pres", "skin",
            "insu", "mass", "pedi", "age"]]
    y = df["diabete"]
    # training set (80%) test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2)
    model.fit(X_train, y_train)
    joblib.dump(model, "diabete_diagnose_model.pkl")
    mse_train = mean_absolute_error(y_train, model.predict(X_train))
    mse_test = mean_absolute_error(y_test, model.predict(X_test))
    lbl_error1["text"] = f" - Training Set Error: {mse_train}"
    lbl_error2["text"] = f" - Test Set Error: {mse_test}"


# part 3
# use trained model to predict the user input

def prediction():
    preg = ent_preg.get()
    plas = ent_preg.get()
    pres = ent_preg.get()
    skin = ent_preg.get()
    insu = ent_preg.get()
    mass = ent_preg.get()
    pedi = ent_preg.get()
    age = ent_preg.get()

    preg_nom = nom_input(preg, preg_min, preg_max)
    plas_nom = nom_input(plas, plas_min, plas_max)
    pres_nom = nom_input(pres, pres_min, pres_max)
    skin_nom = nom_input(skin, skin_min, skin_max)
    insu_nom = nom_input(insu, insu_min, insu_max)
    mass_nom = nom_input(mass, mass_min, mass_max)
    pedi_nom = nom_input(pedi, pedi_min, pedi_max)
    age_nom = nom_input(age, age_min, age_max)

    model = joblib.load("diabete_diagnose_model.pkl")
    patient_input = [preg_nom, plas_nom, pres_nom, skin_nom,
                     insu_nom, mass_nom, pedi_nom, age_nom]
    patient_list = [
        patient_input
    ]
    patient_out = model.predict(patient_list)
    predicted_result = patient_out[0]
    if predicted_result > 0.5:
        lbl_predic["text"] = "Diabeties Positive"
    else:
        lbl_predic["text"] = "Diabeties Nagetive"


# part 5
# GUI for final user
def linear_btn():
    model = LinearRegression()
    return model


def logistic_btn():
    model = LogisticRegression()
    return model


def SGD_btn():
    model = SGDRegressor()
    return model


window = tk.Tk()
window.title("Diabetes Diagnosis Prediction")
window.resizable(width=False, height=False)

frm_buttons = tk.Frame()
frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)
btn_LinearRegression = tk.Button(
    master=frm_buttons,
    text="LinearRegression",
    command=linear_btn
)
btn_LinearRegression.pack(side=tk.LEFT, ipadx=10)
btn_LogisticRegression = tk.Button(
    master=frm_buttons,
    text="LogisticRegression",
    command=logistic_btn
)
btn_LogisticRegression.pack(side=tk.LEFT, ipadx=10)
btn_SGDRegressor = tk.Button(
    master=frm_buttons,
    text="SGDRegressor",
    command=SGD_btn
)
btn_SGDRegressor.pack(side=tk.LEFT, ipadx=10)

frm_training = tk.Frame()
frm_training.pack(fill=tk.X, ipadx=5, ipady=5)
btn_training = tk.Button(
    master=frm_training,
    text="\N{RIGHTWARDS BLACK ARROW}",
    command=training
)
lbl_error1 = tk.Label(master=frm_training,
                      text="0.0")
lbl_error2 = tk.Label(master=frm_training,
                      text="0.0")
btn_training.grid(row=0, column=0, pady=10)
lbl_error1.grid(row=1, column=1, padx=10)
lbl_error2.grid(row=2, column=1, padx=10)

frm_form = tk.Frame(relief=tk.SUNKEN, borderwidth=3)
frm_form.pack()

lbl_preg = tk.Label(master=frm_form, text="Preg")
ent_preg = tk.Entry(master=frm_form, width=50)
lbl_preg.grid(row=0, column=0, sticky="e")
ent_preg.grid(row=0, column=1)
lbl_plas = tk.Label(master=frm_form, text="Plas")
ent_plas = tk.Entry(master=frm_form, width=50)
lbl_plas.grid(row=1, column=0, sticky="e")
ent_plas.grid(row=1, column=1)
lbl_pres = tk.Label(master=frm_form, text="Pres")
ent_pres = tk.Entry(master=frm_form, width=50)
lbl_pres.grid(row=2, column=0, sticky="e")
ent_pres.grid(row=2, column=1)
lbl_skin = tk.Label(master=frm_form, text="Skin")
ent_skin = tk.Entry(master=frm_form, width=50)
lbl_skin.grid(row=3, column=0, sticky="e")
ent_skin.grid(row=3, column=1)
lbl_insu = tk.Label(master=frm_form, text="Insu")
ent_insu = tk.Entry(master=frm_form, width=50)
lbl_insu.grid(row=0, column=2, sticky="e")
ent_insu.grid(row=0, column=3)
lbl_mass = tk.Label(master=frm_form, text="Mass")
ent_mass = tk.Entry(master=frm_form, width=50)
lbl_mass.grid(row=1, column=2, sticky="e")
ent_mass.grid(row=1, column=3)
lbl_pedi = tk.Label(master=frm_form, text="Pedi")
ent_pedi = tk.Entry(master=frm_form, width=50)
lbl_pedi.grid(row=2, column=2, sticky="e")
ent_pedi.grid(row=2, column=3)
lbl_age = tk.Label(master=frm_form, text="Age")
ent_age = tk.Entry(master=frm_form, width=50)
lbl_age.grid(row=3, column=2, sticky="e")
ent_age.grid(row=3, column=3)


frm_buttons = tk.Frame()
frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)
btn_submit = tk.Button(
    master=frm_buttons,
    text="Run the prediction",
    command=prediction
)
btn_submit.pack(side=tk.LEFT, padx=10, ipadx=10)
lbl_predic = tk.Label(master=frm_buttons, text="prediction")
lbl_predic.pack(side=tk.LEFT, padx=10, ipadx=10)


window.mainloop()
