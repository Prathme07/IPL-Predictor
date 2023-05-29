import tkinter as tk
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the model

# Create the Tkinter window
window = tk.Tk()
window.title('IPL Predictor')
window.geometry("100000x10000")

# Function to go back to the home screen
def go_back():
    home_frame.pack()
    toss_frame.pack_forget()
    ipl_frame.pack_forget()
    score_frame.pack_forget()

# Function for toss win predictor
def toss_predictor():
    def predict_toss():
        # Read the CSV file
        df = pd.read_csv('matches.csv')

        # Filter the relevant columns
        df = df[['team1', 'team2', 'toss_winner', 'toss_decision']]

        # Drop rows with missing values
        df = df.dropna()

        # Prepare the data
        X = df[['team1', 'team2', 'toss_decision']]
        y = df['toss_winner']
        
        # Convert categorical variables to numerical using one-hot encoding
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Create a logistic regression model and fit it to the training data
        model = LogisticRegression()
        model.fit(X_train, y_train)
        X_input = X_test[0:1]
       
        for ind in X_input.index:
            for col in X_input.columns:
                X_input[col][ind] = 0
       
        
        # Make prediction on the provided team combination and toss decision
        toss_decision = str(toss_decision_var.get())
        column1 = "team1_" + str(team1_var.get())
        column2 = 'team2_' + str(team2_var.get())
        column3 = 'toss_decision_field'
        
        for ind in X_input.index:
            X_input[column1][ind] = 1
            X_input[column2][ind] = 1
            if(toss_decision == 'field'):
                X_input[column3][ind] = 1
        print(X_input)
        prediction = model.predict(X_input)
        print(prediction)
        # Display the predicted toss winner
        prediction_label.config(text=f"The predicted toss winner is: {prediction[0]}")\
            

    # Create the Tkinter window
    root = tk.Tk()
    root.title('Toss Predictor')

    title=tk.Label(root,text="Toss Predictor",bg='blue',fg='white')
    title.pack()

    # Create team input labels and entry fields
    team1_label = tk.Label(root, text='Team 1:')
    team1_label.pack()
    team1_var=tk.StringVar(root)
    team1_var.set("Select Team1")
    team_1_dropdown=tk.OptionMenu(root,team1_var,*sorted(teams))
    team_1_dropdown.config(bg="light blue",fg="black")
    team_1_dropdown.pack()

    team2_label = tk.Label(root, text='Team 2:')
    team2_label.pack()
    team2_var=tk.StringVar(root)
    team2_var.set("Select Team2")
    team_2_dropdown=tk.OptionMenu(root,team2_var,*sorted(teams))
    team_2_dropdown.config(bg="light blue",fg="black")
    team_2_dropdown.pack()

    # Create toss decision radio buttons
    toss_decision_label = tk.Label(root, text='Toss Decision:')
    toss_decision_label.pack()

    toss_decision_var = tk.StringVar()
    toss_decision_var.set('Bat') 

    toss_decision_bat = tk.Radiobutton(root, text='Bat', variable=toss_decision_var, value='bat')
    toss_decision_bat.pack()

    toss_decision_field = tk.Radiobutton(root, text='Field', variable=toss_decision_var, value='field')
    toss_decision_field.pack()

    # Create predict button
    predict_button = tk.Button(root, text='Predict', command=predict_toss,bg="orange",fg="black")
    predict_button.pack()

    # Create label for displaying the prediction
    prediction_label = tk.Label(root, text="")

    prediction_label.pack()

    # Run the Tkinter event loop
    root.mainloop()
# Function for IPL win predictor


def ipl_predictor():
    root = tk.Tk()
    root.title('IPL Win Predictor')

    title=tk.Label(root,text="IPL WIN PREDICTOR",bg='blue',fg='white')
    title.pack()
    pipe = pickle.load(open('pipe.pkl', 'rb'))

    Target=tk.StringVar()
    Score=tk.StringVar()
    Overs=tk.StringVar()
    Wickets=tk.StringVar()
    Target.set(" ")
    Score.set(" ")
    Overs.set(" ")
    Wickets.set(" ")

    batting_team_var = tk.StringVar(root)
    batting_team_var.set("Select Batting Team")
    batting_team_dropdown = tk.OptionMenu(root, batting_team_var, *sorted(teams))
    batting_team_dropdown.config(bg="light blue",fg="black")
    batting_team_dropdown.pack()


    bowling_team_var = tk.StringVar(root)
    bowling_team_var.set("Select Bowling Team")
    bowling_team_dropdown = tk.OptionMenu(root, bowling_team_var, *sorted(teams))
    bowling_team_dropdown.config(bg="light blue",fg="black")
    bowling_team_dropdown.pack()

    selected_city_var = tk.StringVar(root)
    selected_city_var.set("Select City")
    selected_city_dropdown = tk.OptionMenu(root, selected_city_var, *sorted(cities))
    selected_city_dropdown.config(bg="light blue",fg="black")
    selected_city_dropdown.pack()

    target_Label = tk.Label(root,text="Enter Target")
    target_entry = tk.Entry(root,textvariable=Target)
    target_Label.config(bg="light blue",fg="black")
    target_Label.pack()
    target_entry.pack()

    score_Label = tk.Label(root,text="Enter Score")
    score_entry = tk.Entry(root,textvariable=Score)
    score_Label.config(bg="light blue",fg="black")
    score_Label.pack()
    score_entry.pack()

    overs_Label = tk.Label(root,text="Enter Overs")
    overs_entry = tk.Entry(root,textvariable=Overs)
    overs_Label.config(bg="light blue",fg="black")
    overs_Label.pack()
    overs_entry.pack()

    wickets_Label = tk.Label(root,text="Enter Wickets")
    wickets_entry = tk.Entry(root,textvariable=Wickets)
    wickets_Label.config(bg="light blue",fg="black")
    wickets_Label.pack()
    wickets_entry.pack()

    result_label1 = tk.Label(root, text="")
    result_label1.pack()
    result_label2 = tk.Label(root, text="")
    result_label2.pack()

    def predict_probability():
        batting_team = batting_team_var.get()
        bowling_team = bowling_team_var.get()
        selected_city = selected_city_var.get()
        target = int(target_entry.get())
        score = int(score_entry.get())
        overs = float(overs_entry.get())
        wickets = int(wickets_entry.get())

        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        result_label1.config(text=f"{batting_team} - {round(win*100)}%")
        result_label2.config(text=f"{bowling_team} - {round(loss*100)}%")

    predict_button = tk.Button(root, text="Predict Probability", command=predict_probability,bg="orange",fg="black")
    predict_button.pack()

    root.mainloop()

# Function for score predictor


def score_predictor():
    pipe = pickle.load(open('lr-model.pkl', 'rb'))

    # Create the Tkinter window
    root = tk.Tk()
    root.title('IPL Score Predictor')

    title=tk.Label(root,text="IPL Score Predictor",bg='blue',fg='white')
    title.pack()

    # Function to predict the score
    def predict_score():
        batting_team = batting_team_var.get()
        bowling_team = bowling_team_var.get()
        overs = float(overs_entry.get())
        wickets = int(wickets_entry.get())
        runs = int(runs_entry.get())
        runs_in_prev_5 = int(runs_in_prev_5_entry.get())
        wickets_in_prev_5 = int(wickets_in_prev_5_entry.get())

        # Create a list for the input features
        temp_array = []

        # Batting Team
        if batting_team == 'Chennai Super Kings':
          temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Capitals':
          temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Kings XI Punjab':
          temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
          temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
          temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Rajasthan Royals':
          temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
          temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif batting_team == 'Sunrisers Hyderabad':
          temp_array = temp_array + [0,0,0,0,0,0,0,1]

        # Bowling Team
        if bowling_team == 'Chennai Super Kings':
          temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif bowling_team == 'Delhi Capitals':
          temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Kings XI Punjab':
          temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
          temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif bowling_team == 'Mumbai Indians':
          temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Rajasthan Royals':
          temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
          temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif bowling_team == 'Sunrisers Hyderabad':
          temp_array = temp_array + [0,0,0,0,0,0,0,1]

        
        temp_array.extend([overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5])

        # Convert the list into a numpy array
        input_array = np.array([temp_array])

        # Perform the prediction using the loaded model
        score_prediction = int(pipe.predict(input_array)[0])
        
        
        # Display the predicted score
        result_label.config(text=f"The predicted score is: {score_prediction}")

    # Create labels and entry fields for user input
    batting_team_var = tk.StringVar(root)
    batting_team_var.set("Select Batting Team")
    batting_team_dropdown = tk.OptionMenu(root, batting_team_var, *sorted(teams))
    batting_team_dropdown.config(bg="light blue",fg="black")
    batting_team_dropdown.pack()

    bowling_team_var = tk.StringVar(root)
    bowling_team_var.set("Select Bowling Team")
    bowling_team_dropdown = tk.OptionMenu(root, bowling_team_var, *sorted(teams))
    bowling_team_dropdown.config(bg="light blue",fg="black")
    bowling_team_dropdown.pack()

    overs_label = tk.Label(root, text="Overs played:")
    overs_label.pack()
    overs_entry = tk.Entry(root)
    overs_label.config(bg="light blue",fg="black")
    overs_entry.pack()

    wickets_label = tk.Label(root, text="Wickets done:")
    wickets_label.pack()
    wickets_entry = tk.Entry(root)
    wickets_label.config(bg="light blue",fg="black")
    wickets_entry.pack()

    runs_label = tk.Label(root, text="Runs obtained:")
    runs_label.pack()
    runs_entry = tk.Entry(root)
    runs_label.config(bg="light blue",fg="black")
    runs_entry.pack()

    runs_in_prev_5_label = tk.Label(root, text="Runs in Previous 5 Overs:")
    runs_in_prev_5_label.pack()
    runs_in_prev_5_entry = tk.Entry(root)
    runs_in_prev_5_label.config(bg="light blue",fg="black")
    runs_in_prev_5_entry.pack()

    wickets_in_prev_5_label = tk.Label(root, text="Wickets in Previous 5 Overs:")
    wickets_in_prev_5_label.pack()
    wickets_in_prev_5_entry = tk.Entry(root)
    wickets_in_prev_5_label.config(bg="light blue",fg="black")
    wickets_in_prev_5_entry.pack()

    predict_button = tk.Button(root, text="Predict Score", command=predict_score,bg="orange",fg="black")
    predict_button.pack()

    result_label = tk.Label(root, text="")
    result_label.pack()

    # Start the Tkinter event loop
    root.mainloop()

# Create frames for each screen
home_frame = tk.Frame(window)
toss_frame = tk.Frame(window)
ipl_frame = tk.Frame(window)
score_frame = tk.Frame(window)

# Home screen
title = tk.Label(home_frame, text="IPL Predictor", bg='blue', fg='white')
title.pack()

toss_button = tk.Button(home_frame, text="Toss Win Predictor", command=toss_predictor, bg="orange", fg="black")
toss_button.pack()

ipl_button = tk.Button(home_frame, text="IPL Win Predictor", command=ipl_predictor, bg="orange", fg="black")
ipl_button.pack()

score_button = tk.Button(home_frame, text="Score Predictor", command=score_predictor, bg="orange", fg="black")
score_button.pack()

# Toss win predictor screen
toss_label = tk.Label(toss_frame, text="Toss Win Predictor")
toss_label.pack()

# Add your toss win predictor form or logic here

back_button1 = tk.Button(toss_frame, text="Back", command=go_back)
back_button1.pack()

# IPL win predictor screen
ipl_label = tk.Label(ipl_frame, text="IPL Win Predictor")
ipl_label.pack()

# Add your IPL win predictor form or logic here

back_button2 = tk.Button(ipl_frame, text="Back", command=go_back)
back_button2.pack()

# Score predictor screen
score_label = tk.Label(score_frame, text="Score Predictor")
score_label.pack()

# Add your score predictor form or logic here

back_button3 = tk.Button(score_frame, text="Back", command=go_back)
back_button3.pack()

# Display the home screen initially
home_frame.pack()

# Start the Tkinter event loop
window.mainloop()
