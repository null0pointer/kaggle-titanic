import pandas as pd
import math
from sklearn import svm

# Read datasets
print('Reading datasets...')
train_dataset = pd.read_csv('train.csv', header=0, index_col=0)
prediction_dataset = pd.read_csv('test.csv', header=0, index_col=0)
combined_dataset = train_dataset.append(prediction_dataset)

# Create deck column
print('Creating new columns...')
deck_set = {'C'}
ticket_deck_map = {}

# Creating the deck values
for index, row in combined_dataset.iterrows():
    cabin_string = str(row['Cabin'])
    if not cabin_string == 'nan':
        deck_string = cabin_string[0]
        deck_set.add(deck_string)
        ticket_deck_map[row['Ticket']] = deck_string

# Populate the deck column using deck C as the default
train_deck = []
for index, row in train_dataset.iterrows():
    if row['Ticket'] in ticket_deck_map:
        train_deck.append(ticket_deck_map[row['Ticket']])
    else:
        train_deck.append('C')
        
prediction_deck = []
for index, row in prediction_dataset.iterrows():
    if row['Ticket'] in ticket_deck_map:
        prediction_deck.append(ticket_deck_map[row['Ticket']])
    else:
        prediction_deck.append('C')

# Add deck column
print('Add new columns...')
train_dataset.insert(2, 'Deck', train_deck)
prediction_dataset.insert(2, 'Deck', prediction_deck)

# Remove unused columns
print('Removing name...')
train_dataset = train_dataset.drop('Name', 1)
prediction_dataset = prediction_dataset.drop('Name', 1)
print('Removing ticket...')
train_dataset = train_dataset.drop('Ticket', 1)
prediction_dataset = prediction_dataset.drop('Ticket', 1)
print('Removing cabin...')
train_dataset = train_dataset.drop('Cabin', 1)
prediction_dataset = prediction_dataset.drop('Cabin', 1)

# Map the values of some of the features
print('Mapping some basic values...')
port_map = {'C': 0, 'Q': 1, 'S': 2}
sex_map = {'male': 0, 'female': 1}

deck_map = {}
index = 0
for deck in deck_set:
    deck_map[deck] = index
    index = index + 1
    
final_map = {'Embarked': port_map, 'Sex': sex_map, 'Deck': deck_map}
train_dataset = train_dataset.replace(final_map)
prediction_dataset = prediction_dataset.replace(final_map)

# Fill NaN age values
print('Filling the NaN age values...')
total_age = 0
count = 0
for index, row in combined_dataset.iterrows():
    if not math.isnan(row['Age']):
        total_age = total_age + row['Age']
        count = count + 1

average_age = total_age / count

for index, row in train_dataset.iterrows():
    if math.isnan(row['Age']):
        train_dataset.set_value(index, 'Age', average_age)

for index, row in prediction_dataset.iterrows():
    if math.isnan(row['Age']):
        prediction_dataset.set_value(index, 'Age', average_age)
        
# Fill NaN Embarked values
print('Filling the NaN embarked values')
for index, row in train_dataset.iterrows():
    if math.isnan(row['Embarked']):
        train_dataset.set_value(index, 'Embarked', 2.0)

for index, row in prediction_dataset.iterrows():
    if math.isnan(row['Embarked']):
        prediction_dataset.set_value(index, 'Embarked', 2.0)
        
# Fill NaN fare values
print('Filling NaN fare values')
for index, row in train_dataset.iterrows():
    if math.isnan(row['Fare']):
        train_dataset.set_value(index, 'Fare', 0.0)

for index, row in prediction_dataset.iterrows():
    if math.isnan(row['Fare']):
        prediction_dataset.set_value(index, 'Fare', 0.0)

# Split training data from training targets
print('Split training data into features and targets...')
target_dataset = train_dataset.drop('Age', 1)
target_dataset = target_dataset.drop('Pclass', 1)
target_dataset = target_dataset.drop('Deck', 1)
target_dataset = target_dataset.drop('Sex', 1)
target_dataset = target_dataset.drop('SibSp', 1)
target_dataset = target_dataset.drop('Parch', 1)
target_dataset = target_dataset.drop('Fare', 1)
target_dataset = target_dataset.drop('Embarked', 1)

train_dataset = train_dataset.drop('Survived', 1)

training_matrix = train_dataset.as_matrix()
target_matrix = target_dataset.as_matrix().flatten()

# Train SVM
print('Training model...')
classifier = svm.SVC(verbose=True)
classifier.fit(training_matrix, target_matrix)

# Predicting survival
print('Predicting...')
predictions = classifier.predict(prediction_dataset)

indices = prediction_dataset.index.values

output_data = {'PassengerId': indices, 'Survived': predictions}
output_dataframe = pd.DataFrame(data=output_data)
output_dataframe.to_csv('predictions.csv', index=False)
print(output_dataframe)