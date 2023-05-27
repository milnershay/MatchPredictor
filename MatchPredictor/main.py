import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

with open('models/DecisionTreeClassifier.pkl', 'rb') as file:
    model1 = pickle.load(file)
with open('models/GradientBoostingClassifier.pkl', 'rb') as file:
    model2 = pickle.load(file)
with open('models/KNeighborsClassifier.pkl', 'rb') as file:
    model3 = pickle.load(file)
with open('models/RandomForestClassifier.pkl', 'rb') as file:
    model4 = pickle.load(file)

X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')
name_id_data = pd.read_csv('data/name_id_map.csv')

# Create the main window
window = tk.Tk()
window.title("Soccer Match Prediction")

# Create a style for the dropdown menus
style = ttk.Style(window)
style.theme_use("clam")
style.configure("TLabel", background="white", foreground="black")
style.configure("TButton", background="blue", foreground="white")

# Add a label for match selection
match_label = tk.Label(window, text="Select a match:")
match_label.pack()


# Merge x_test_data with name_id_data to get team names
x_test_data_with_names = X_test.merge(name_id_data, left_on='home_team_api_id', right_on='team_api_id', how='left')
x_test_data_with_names = x_test_data_with_names.merge(name_id_data, left_on='away_team_api_id', right_on='team_api_id', how='left')

# Create the match options with team names
match_options = [f"{home_team} vs {away_team}" for home_team, away_team in zip(x_test_data_with_names['team_long_name_x'], x_test_data_with_names['team_long_name_y'])]

# Add a dropdown to select the match
match_var = tk.StringVar()
match_dropdown = ttk.Combobox(window, textvariable=match_var, values=match_options, state='readonly', style="TCombobox")
match_dropdown.configure(width=50)
match_dropdown.pack()

# Add a label for model selection
model_label = tk.Label(window, text="Select a model:")
model_label.pack()

# Add a dropdown to select the model
models = [type(model1).__name__, type(model2).__name__, type(model3).__name__, type(model4).__name__]
model_var = tk.StringVar()
model_dropdown = ttk.Combobox(window, textvariable=model_var, values=models, state='readonly')
model_dropdown.pack()


# Function to handle prediction
def predict_match():
    selected_match = match_dropdown.get()
    selected_model = model_dropdown.get()
    if not selected_model or not selected_model:
        messagebox.showinfo("Error", "Must choose match and model in order to predict")
    # Retrieve team names based on selected match
    home_team_name, away_team_name = selected_match.split(' vs ')
    home_team_api_id = x_test_data_with_names.loc[x_test_data_with_names['team_long_name_x'] == home_team_name, 'home_team_api_id'].values[0]
    away_team_api_id = x_test_data_with_names.loc[x_test_data_with_names['team_long_name_y'] == away_team_name, 'away_team_api_id'].values[0]

    # Get the selected model based on the model name
    if selected_model == type(model1).__name__:
        selected_model = model1
    elif selected_model == type(model2).__name__:
        selected_model = model2
    elif selected_model == type(model3).__name__:
        selected_model = model3
    elif selected_model == type(model4).__name__:
        selected_model = model4

    # Prepare the input data for prediction
    input_data = X_test.loc[(X_test['home_team_api_id'] == home_team_api_id) & (X_test['away_team_api_id'] == away_team_api_id)].drop(['home_team_api_id', 'away_team_api_id'], axis=1)

    # Perform the prediction using the selected model
    prediction = selected_model.predict(input_data)
    result = y_test.loc[(y_test['home_team_api_id'] == home_team_api_id) & (
                y_test['away_team_api_id'] == away_team_api_id), 'outcome'].values[0]
    print(prediction)
    print(result)
    if prediction == 0:
        prediction = 'Draw'
    elif prediction == 1:
        prediction = 'Home win'
    else:
        prediction = 'Away win'
    if result == 0:
        result= 'Draw'
    elif result == 1:
        result = 'Home win'
    else:
        result = 'Away win'
    # Display the predicted value
    prediction_label.config(text=f"Prediction: {prediction}")
    true_label.config(text=f"Actual result: {result}")


# Add a button to initiate prediction
predict_button = tk.Button(window, text="Predict", command=predict_match)
predict_button.pack(pady=10)


# Add a prediction label to show prediction
prediction_label = tk.Label(window, text="Prediction:")
prediction_label.pack()

# Add a truth label to show true value
true_label = tk.Label(window, text="Actual result:")
true_label.pack()


# Set the window size and center it on the screen
window_width = 500
window_height = 200
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x_coordinate = int((screen_width / 2) - (window_width / 2))
y_coordinate = int((screen_height / 2) - (window_height / 2))
window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Run the GUI main loop
window.mainloop()

