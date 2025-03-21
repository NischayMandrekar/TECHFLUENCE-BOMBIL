import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Determine the maximum length of features
max_length = max(len(sample) for sample in data)

# Pad all samples to have the same length
data_padded = np.array([sample + [0] * (max_length - len(sample)) for sample in data])

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test Model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save Model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)