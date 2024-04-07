import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Suppress UserWarning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

# Load the dataset from the specified path
data = pd.read_excel("C:/users/zack3/Downloads/baseball.xlsx")

# Extract the required columns: Team, Year, Runs Scored, Runs Allowed, Wins, OBP, SLG, Team Batting Average, and Playoffs
# Note: Column indexing starts from 0, so column D is index 3, column E is index 4, and so on
required_columns = ['Team', 'Year', 'Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average', 'Playoffs']
data = data[required_columns]

# Separate features (X) and target variable (y)
X = data[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']]
y = data['Playoffs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)

# Create a dictionary to store the number of correct predictions for each team
team_correct_predictions = {}

# Now, let's predict whether each team for each year will make the playoffs and compare with the actual playoff outcomes
# Iterate through each row in the dataset
for index, row in data.iterrows():
    # Extract team, year, and features
    team = row['Team']
    year = row['Year']
    features = row[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']].values.reshape(1, -1)
    
    # Make prediction
    prediction = clf.predict(features)[0]
    
    # Check if the prediction matches the actual playoff outcome
    if prediction == row['Playoffs']:
        # Increment the count of correct predictions for this team
        team_correct_predictions[team] = team_correct_predictions.get(team, 0) + 1

# Print out the number of correct predictions for each team
print("Number of correct predictions for each team:")
for team, correct_predictions in team_correct_predictions.items():
    print(f"{team}: {correct_predictions} out of {len(data[data['Team'] == team])}")

# Calculate the total number of correct predictions across all teams
total_correct_predictions = sum(team_correct_predictions.values())

# Print out the total number of correct predictions across all teams
print(f"\nTotal number of correct predictions: {total_correct_predictions} out of {len(data)}")
print("Go Brewers!")
