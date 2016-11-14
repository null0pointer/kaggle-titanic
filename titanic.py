import pandas as pd

# Read datasets
print('Reading training dataset...')
train_dataset = pd.read_csv('train.csv', header=0, index_col=0)
print('Reading prediction dataset...')
prediction_dataset = pd.read_csv('test.csv', header=0, index_col=0)

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