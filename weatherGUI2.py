#Task 2 :-- Weather predictor using Classification

import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create GUI window
window = tk.Tk()
window.title("Weather Prediction By Raj")
window.geometry("500x600+50+50")
window.configure(bg="azure")
f = ("Cambria", 20, "bold")

lab_header = tk.Label(window, text="Weather Predictor", font=f,bg="azure")
lab_header.pack(pady=30)

# Load the weather dataset
def load_dataset():
    data = pd.read_csv("C:/demo/ML/internship/weather_predictor/weather.csv")
    X = data[["precipitation", "temp_max", "temp_min", "wind"]]
    y = data[["weather"]]
    return X, y

# Train a classification model
def train_model(X, y):
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    return model

# Function to predict weather
def predict_weather():
    # Get user inputs from GUI
    precipitation = (entry_precipitation.get())
    temp_max = (entry_temp_max.get())
    temp_min = (entry_temp_min.get())
    wind = (entry_wind.get())

    if not precipitation:
             messagebox.showerror("Error", "Please Enter precipitation")
             return

    if not temp_max:
             messagebox.showerror("Error", "Please Enter temp_max")
             return

    if not temp_min:
             messagebox.showerror("Error", "Please Enter temp_min")
             return

    if not wind:
             messagebox.showerror("Error", "Please Enter wind")
             return

    if precipitation.strip() == "":
             messagebox.showerror("Error", "precipitation cannot be spaces")
             return

    if temp_max.strip() == "":
             messagebox.showerror("Error", "temp_max cannot be spaces")
             return

    if temp_min.strip() == "":
             messagebox.showerror("Error", "temp_min cannot be spaces")
             return

    if wind.strip() == "":
             messagebox.showerror("Error", "wind cannot be spaces")
             return

    if precipitation.isalpha():
             messagebox.showerror("Error", "precipitation cannot be text")
             return

    if temp_max.isalpha():
             messagebox.showerror("Error", "temp_max cannot be text")
             return

    if temp_min.isalpha():
             messagebox.showerror("Error", "temp_min cannot be text")
             return

    if wind.isalpha():
             messagebox.showerror("Error", "wind cannot be text")
             return

    if not precipitation.replace('.', '', 1).isdigit():
             messagebox.showerror(f"Error", "precipitation cannot be Special Characters")
             return

    if not temp_max.replace('.', '', 1).isdigit():
             messagebox.showerror(f"Error", "temp_max cannot be Special Characters")
             return

    #if not temp_min.replace('.', '', 1).isdigit():
             #messagebox.showerror(f"Error", "temp_min cannot be Special Characters")
             #return

    if not wind.replace('.', '', 1).isdigit():
             messagebox.showerror(f"Error", "wind cannot be Special Characters")
             return

    try:
             precitation = float(precipitation)
             temp_max = float(temp_max)
             temp_min = float(temp_min)
             wind = float(wind)

    except ValueError as e:
             print("Error", "Something went wrong!")
             return

    entry_precipitation.delete(0 ,'end')
    entry_temp_max.delete(0, 'end')
    entry_temp_min.delete(0, 'end')
    entry_wind.delete(0, 'end')
    entry_precipitation.focus()


    # Load and preprocess the dataset
    X, y = load_dataset()

    # Train the model
    model = train_model(X, y)

    # Make prediction
    prediction = model.predict([[precipitation, temp_max, temp_min, wind]])

    # Display prediction
    label_result.config(text="Predicted Weather : " + str(prediction[0]))

# Create precipitation label and entry field
label_precipitation = tk.Label(window, text="precipitation:", font=f,bg="azure")
label_precipitation.pack(pady=5)
entry_precipitation = tk.Entry(window,font=f)
entry_precipitation.pack(pady=5)

# Create temp_max label and entry field
label_temp_max = tk.Label(window, text="temp_max:", font=f,bg="azure")
label_temp_max.pack(pady=5)
entry_temp_max = tk.Entry(window,font=f)
entry_temp_max.pack(pady=5)

# Create temp_min label and entry field
label_temp_min = tk.Label(window, text="temp_min:", font=f,bg="azure")
label_temp_min.pack(pady=5)
entry_temp_min = tk.Entry(window,font=f)
entry_temp_min.pack(pady=5)

# Create wind label and entry field
label_wind = tk.Label(window, text="wind:", font=f,bg="azure")
label_wind.pack(pady=5)
entry_wind = tk.Entry(window,font=f)
entry_wind.pack(pady=5)

# Create prediction button
button_predict = tk.Button(window, text="Predict", command=predict_weather, font=f)
button_predict.pack(pady=5)

# Create result label
label_result = tk.Label(window, text="", font=f,bg="azure")
label_result.pack(pady=5)

# Run the GUI
window.mainloop()