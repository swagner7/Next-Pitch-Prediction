import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, brier_score_loss, log_loss
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

data = pd.read_csv('pitches')  # read pitches file using pandas
data = data.loc[:, data.notna().any()]  # filter out any columns that have no data

data = data[['inning', 'at_bat_num', 'pcount_at_bat', 'pcount_pitcher', 'balls', 'strikes', 'fouls', 'outs',
             'away_team_runs', 'home_team_runs', 'pitcher_id', 'pitch_type']].dropna()


# Loop through unique pitch types and calculate running totals for each pitcher
for pitch_type in data['pitch_type'].unique():
    # Create a boolean mask for the current pitch type
    mask = data['pitch_type'] == pitch_type

    # Calculate cumulative count of pitches for each pitch type,
    # adjusting the counting to ignore the current row for each pitcher id
    data[pitch_type + '_total'] = (mask.groupby(data['pitcher_id']).cumsum() - mask.astype(int))

# Convert counts to percentages
for pitcher_id in data['pitcher_id'].unique():
    total_pitcher = data.loc[data['pitcher_id'] == pitcher_id, [col for col in data.columns if col.endswith('_total')]].sum(
        axis=1)
    total_pitcher = total_pitcher.astype(float)  # Cast to float
    data.loc[
        data['pitcher_id'] == pitcher_id, [col for col in data.columns if col.endswith('_total')]] /= total_pitcher.values[
                                                                                                  :, None]



data = data.dropna(axis=0)
X = data.drop(['pitch_type', 'pitcher_id'], axis=1)
y = data['pitch_type'].str.upper()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop("pitch_type", axis=1))
scaled_data = pd.DataFrame(scaled_data, columns=data.columns.drop("pitch_type"))

# Update the dataset
data = pd.concat([scaled_data, data["pitch_type"]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')

# Confusion matrix
confmat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Get class names from the classifier
class_names = clf.classes_

plt.figure()
heatmap = sns.heatmap(confmat, annot=True, fmt='d', cmap="Blues")  # fmt='d' for standard notation

# Set tick labels to class names
heatmap.set_xticklabels(class_names, rotation=45)
heatmap.set_yticklabels(class_names, rotation=0)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
