# Getting started
mlrequest-python is a Python client for the [mlrequest](https://mlrequest.com) machine learning API. The client allows you to do a few significant things with minimal code.

* Deploy latency-routed **scikit-learn** and **online** machine learning models to 5 data centers around the world, providing < 60ms global response time and automatic failover.
* Get thousands of model predictions per second
* Train online models with thousands of training examples per second

You will need an API key to get started with mlrequest-python. You can obtain one for free that provides 5,000 monthly model transactions at https://mlrequest.com/signup.html. The free plan is limited to the deployment of a single online learning model or scikit-learn model file less than 1 MB in size. Scikit-learn model transactions are prioritized for paid accounts, and will generally receive up to 50ms faster response time than free accounts.

For more transactions, larger scikit-learn model files (up to 100 MB), more models, and faster scikit-learn response times, see our paid plans at https://mlrequest.com/pricing.html. Check out our [documentation](https://docs.mlrequest.com) for more information.

## Install
```
pip install mlrequest
```

## Scikit-Learn
### Create and Deploy a Scikit-Learn Model

```python
from sklearn.ensemble import RandomForestClassifier
from mlrequest import SKLearn

rf_clf = RandomForestClassifier()
rf_clf.fit(X, y)

sklearn = SKLearn('your-api-key')
sklearn.deploy('my-rf-classifier', rf_clf)

#Make predictions
pred = sklearn.predict('my-rf-classifier')
```

## Online learning
### Create a Model
Models are created automatically by calling one of the model endpoints below.

### Classifier
Currently classification is limited to logistic regression. Email support@mlrequest.com to request other online classifier models.
```python
from mlrequest import Classifier
classifier = Classifier('your-api-key')

# Learn single
training_data = {'features': {'feature1': 23.1, 'feature2': 'some-value'}, 'label': 1}
# Learn batch
training_data = [{'features': {'feature1': 23.1, 'feature2': 'some-value'}, 'label': 1}, ...]

r = classifier.learn(training_data=training_data, model_name='my-model', class_count=2)
r.content # A single response or list of responses

# Predict single
features = {'feature1': 23.1, 'feature2': 'some-value'}
# Predict batch
features = [{'feature1': 23.1, 'feature2': 'some-value'}, ...]

r = classifier.predict(features=features, model_name='my-model', class_count=2)
r.predict_result # A single predicted class or a list of predicted classes
```

### Regression
Currently regression is limited to linear regression. Email support@mlrequest.com to request other online regression models.
```python
from mlrequest import Regression
regression = Regression('your-api-key')

# Learn single
training_data = {'features': {'feature1': 23.1, 'feature2': 'some-value'}, 'label': 1.25}
# Learn batch
training_data = [{'features': {'feature1': 23.1, 'feature2': 'some-value'}, 'label': 1.25}, ...]

r = regression.learn(training_data=training_data, model_name='my-model')
r.content # A single response or list of responses

# Predict single
features = {'feature1': 23.1, 'feature2': 'some-value'}
# Predict batch
features = [{'feature1': 23.1, 'feature2': 'some-value'}, ...]

r = regression.predict(features=features, model_name='my-model')
r.predict_result # A single predicted value or a list of predicted values
```

### Reinforcement Learning
```python
from mlrequest import RL
rl = RL('your-api-key')

# Predict
# Note: epsilon, and action_list fields are optional - see the docs at https://docs.mlrequest.com for more information
features = {'feature1': 23.1, 'feature2': 'some-value'}

r = rl.predict(features=features, model_name='my-model', session_id='some-session-id', negative_reward=0, action_count=2)
r.predict_result # A list of actions, ordered by rank (choose r.predict_data[0] for the best action)

# Reward - important note: only the first action from predict_data should be rewarded. Other actions can be used but should not be rewarded.
r = rl.reward(model_name=model_name, session_id='some_session', reward=1)
r.content # A single response
```

## Account
```python
from mlrequest import Account
account = Account('your-api-key')

# Get account information
r = account.get_details()
r.content # Account info response

# Delete a model
r = account.delete_model(model_name='some-model')
r.content # Delete success response
```
