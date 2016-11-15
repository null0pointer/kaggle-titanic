import pandas as pd

# Read datasets
print('Reading datasets...')
train_dataset = pd.read_csv('train.csv', header=0, index_col=0)
prediction_dataset = pd.read_csv('test.csv', header=0, index_col=0)
combined_dataset = train_dataset.append(prediction_dataset)

# Create deck column
print('Creating new columns...')
deck_set = {'None'}
ticket_deck_map = {}

# Creating the deck values
for index, row in combined_dataset.iterrows():
    cabin_string = str(row['Cabin'])
    if not cabin_string == 'nan':
        deck_string = cabin_string[0]
        deck_set.add(deck_string)
        ticket_deck_map[row['Ticket']] = deck_string

# Populate the 

# Add deck column
print('Add new columns...')


# Remove unused columns
print('Removing name...')
train_dataset = train_dataset.drop('Name', 1)
prediction_dataset = prediction_dataset.drop('Name', 1)
print('Removing ticket...')
train_dataset = train_dataset.drop('Ticket', 1)
prediction_dataset = prediction_dataset.drop('Ticket', 1)

# Map the values of some of the features
print('Mapping some basic values...')
port_map = {'C': 0, 'Q': 1, 'S': 2}
sex_map = {'male': 0, 'female': 1}
final_map = {'Embarked': port_map, 'Sex': sex_map}
train_dataset = train_dataset.replace(final_map)

print(train_dataset)