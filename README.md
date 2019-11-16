# Getting started
mlrequest-python is a Python client for the [mlrequest](https://mlrequest.com) machine learning API. The client allows you to do a few significant things with minimal code.

* Deploy latency-routed **scikit-learn** and **online** machine learning models to 5 data centers around the world, providing < 60ms global response time and automatic failover.
* Get thousands of model predictions per second
* Train online models with thousands of training examples per second

You will need an API key to get started with mlrequest-python. You can obtain one for free that provides 5,000 monthly model transactions at https://mlrequest.com/signup.html. The free plan is limited to the deployment of a single online learning model or scikit-learn model file less than 1 MB in size. Scikit-learn model transactions are prioritized for paid accounts, and will generally receive up to 50ms faster response time than free accounts. Additionally, free accounts are restricted to a single data center (of your choosing) and will not benefit from latency routing.

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

clf = RandomForestClassifier()
clf.fit(X, y)

sklearn = SKLearn('your-api-key')
sklearn.deploy(clf, 'rf-model-name')

# Make predictions
features = [[1, 2, 3]]
pred = sklearn.predict(features, 'rf-model-name')
```
### Deploy a Scikit-Learn Model to a Specific Region
If you have a free or single region account, you will only be permitted to deploy to one data center (region) at a time. Your model will automatically failover to another region when loss of service is experienced in your deployed region. Below is an example of how to specify the region to deploy to.

```python
from sklearn.ensemble import RandomForestClassifier
from mlrequest import SKLearn
from mlrequest import regions

clf = RandomForestClassifier()
clf.fit(X, y)

sklearn = SKLearn('your-api-key')
sklearn.deploy(clf, 'rf-model-name', regions.US_EAST)

# Make predictions
features = [[1, 2, 3]]
pred = sklearn.predict(features, 'rf-model-name', regions.US_EAST)
```
Use any of the following regions.
* `regions.US_WEST` (N. California)
* `regions.US_EAST` (Ohio)
* `regions.EU_CENTRAL` (Frankfurt)
* `regions.AP_SOUTH` (Mumbai)
* `regions.AP_NORTHEAST` (Seoul)

## Online Learning
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

r = classifier.learn(training_data=training_data, model_name='clf-model-name', class_count=2)
r.content # A single response or list of responses

# Predict single
features = {'feature1': 23.1, 'feature2': 'some-value'}
# Predict batch
features = [{'feature1': 23.1, 'feature2': 'some-value'}, ...]

r = classifier.predict(features=features, model_name='clf-model-name', class_count=2)
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

r = regression.learn(training_data=training_data, model_name='reg-model-name')
r.content # A single response or list of responses

# Predict single
features = {'feature1': 23.1, 'feature2': 'some-value'}
# Predict batch
features = [{'feature1': 23.1, 'feature2': 'some-value'}, ...]

r = regression.predict(features=features, model_name='reg-model-name')
r.predict_result # A single predicted value or a list of predicted values
```

### Reinforcement Learning
```python
from mlrequest import RL
rl = RL('your-api-key')

# Predict
# Note: epsilon, and action_list fields are optional - see the docs at https://docs.mlrequest.com for more information
features = {'feature1': 23.1, 'feature2': 'some-value'}

r = rl.predict(features=features, model_name='rl-model-name', session_id='some-session-id', negative_reward=0, action_count=2)
r.predict_result # A list of actions, ordered by rank (choose r.predict_data[0] for the best action)

# Reward - important note: only the first action from predict_data should be rewarded. Other actions can be used but should not be rewarded.
r = rl.reward(reward=1, model_name='rl-model-name', session_id='some-session-id')
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
r = account.delete_model(model_name='some-model-name')
r.content # Delete success response
```
