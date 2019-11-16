from mlrequest import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from unittest.mock import patch, Mock
import unittest
from os import path
import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.features = {'a': 1, 'b': 2}
        self.features_batch = []
        for i in range(0, 250):
            self.features_batch.append(self.features)

        self.training_data = {'features': self.features, 'label': 1}
        self.training_data_batch = []
        for i in range(0, 250):
            self.training_data_batch.append(self.training_data)

    def test_classifier_learn(self):
        classifier = Classifier('test')
        model_name = 'python-classifier-test'
        response = classifier.learn(self.training_data, model_name, 2)
        self.assertEqual(response.content['message'], 'OK')

    def test_classifier_predict(self):
        classifier = Classifier('test')
        model_name = 'python-classifier-test'
        response = classifier.predict(self.features, model_name, 2)
        self.assertIsInstance(response.predict_result, int)

    def test_classifier_learn_batch(self):
        classifier = Classifier('test')
        model_name = 'python-classifier-test'
        response = classifier.learn(self.training_data_batch, model_name, 2)
        self.assertEqual(response.content[0]['message'], 'OK')

    def test_classifier_predict_batch(self):
        classifier = Classifier('test')
        model_name = 'python-classifier-test'
        response = classifier.predict(self.features_batch, model_name, 2)
        self.assertIsInstance(response.predict_result[0], int)

    def test_regression_learn(self):
        regression = Regression('test')
        model_name = 'python-regression-test'
        response = regression.learn(self.training_data, model_name)
        self.assertEqual(response.content['message'], 'OK')

    def test_regression_predict(self):
        regression = Regression('test')
        model_name = 'python-regression-test'
        response = regression.predict(self.features, model_name)
        self.assertIsInstance(response.predict_result, float)

    def test_regression_learn_batch(self):
        regression = Regression('test')
        model_name = 'python-regression-test'
        response = regression.learn(self.training_data_batch, model_name)
        self.assertEqual(response.content[0]['message'], 'OK')

    def test_regression_predict_batch(self):
        regression = Regression('test')
        model_name = 'python-regression-test'
        response = regression.predict(self.features_batch, model_name)
        self.assertIsInstance(response.predict_result[0], float)

    def test_rl_predict(self):
        rl = RL('test')
        model_name = 'python-rl-test'
        response = rl.predict(self.features, model_name, 'some-session', 0, 2, 0.15)
        self.assertIsInstance(response.predict_result[0], int)

    def test_rl_reward(self):
        rl = RL('test')
        model_name = 'python-rl-test'
        response = rl.reward(1, model_name, 'some_session')
        self.assertEqual(response.content['message'], 'OK')

    def test_account_info(self):
        account = Account('test')
        response = account.get_details()
        self.assertIsInstance(response.content, dict)

    def test_delete_model(self):
        account = Account('test')
        response = account.delete_model('some-model')
        self.assertIsInstance(response.content, dict)

    @patch('mlrequest.mlrequest.requests.put')
    @patch('mlrequest.mlrequest.requests.post')
    def test_sklearn_deploy(self, mock_post, mock_put):
        model_name = 'test-model'

        mock_post_response = Mock()
        mock_post_response.json.return_value = {'url': 'http://test-url', 'tier': 0}
        mock_post_response.status_code = 200
        mock_post.return_value = mock_post_response

        mock_put_response = Mock()
        mock_put_response.json.return_value = {}
        mock_put_response.status_code = 200
        mock_put.return_value = mock_put_response

        if path.exists(model_name):
            clf = skljson.from_json(model_name)
        else:
            X, y = make_classification(n_samples=50, n_features=3, n_classes=3, n_informative=3, n_redundant=0,
                                       random_state=0, shuffle=False)
            clf = RandomForestClassifier()
            clf.fit(X, y)
            skljson.to_json(clf, model_name)

        sklearn = SKLearn('test')
        sklearn.deploy(clf, model_name)

        mock_post.assert_called_once()
        mock_put.assert_called_once()

    @patch('mlrequest.mlrequest.requests.post')
    def test_sklearn_model_exceeded(self, mock_post):
        model_name = 'test-model-1mb'

        mock_post_response = Mock()
        mock_post_response.json.return_value = {'url': 'http://test-url', 'tier': 0}
        mock_post_response.status_code = 200
        mock_post.return_value = mock_post_response

        if path.exists(model_name):
            clf = skljson.from_json(model_name)
        else:
            X, y = make_classification(n_samples=15000, n_features=10, n_classes=3, n_informative=3, n_redundant=0,
                                       random_state=0, shuffle=False)
            clf = RandomForestClassifier()
            clf.fit(X, y)
            skljson.to_json(clf, model_name)

        sklearn = SKLearn('test')
        with self.assertRaises(mlrequest.ModelSizeExceeded) as exception:
            sklearn.deploy(clf, model_name)
        mock_post.assert_called_once()

    def test_sklearn_predict(self):
        sklearn = SKLearn('test')
        response = sklearn.predict([[1,2,3]], 'python-classifier-test')
        print(response)
        self.assertEqual([0], response['predict_result'])

    def test_sklearn_predict_bad_feature_format(self):
        sklearn = SKLearn('test')
        with self.assertRaises(ValueError) as exception:
            sklearn.predict([1,2,3], 'python-classifier-test')
