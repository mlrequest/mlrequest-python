from mlrequest import *
import unittest


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
        response = rl.reward(model_name, 'some_session', 1)
        self.assertEqual(response.content['message'], 'OK')

    def test_account_info(self):
        account = Account('test')
        response = account.get_details()
        self.assertIsInstance(response.content, dict)

    def test_delete_model(self):
        account = Account('test')
        response = account.delete_model('some-model')
        self.assertIsInstance(response.content, dict)
