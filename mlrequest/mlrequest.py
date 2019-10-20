from requests_futures.sessions import FuturesSession
import json


class Classifier:
    session = FuturesSession()
    base_url = 'https://api.mlrequest.com'
    model_path = '/v1/classifier/'
    batch_model_path = '/v1/classifier/batch/'
    predict_path = 'predict'
    learn_path = 'learn'
    batch_size = 100

    def __init__(self, api_key=None):
        if not api_key:
            raise MissingAPIKey("An API Key is required. Visit https://mlrequest.com for a free or paid API Key.")
        self.api_key = api_key

    def predict(self, features, model_name, class_count):
        self._validate_predict_fields(features, model_name, class_count)

        if isinstance(features, list):
            return self.batch_predict(features, model_name, class_count)
        else:
            payload = {
                'api_key': self.api_key,
                'features': features,
                'model_name': model_name,
                'class_count': class_count
            }
            future = self.session.post(f'{self.base_url}{self.model_path}{self.predict_path}', json=payload)
            response = json.loads(future.result().content)

            if ('predict_result' in response) and ('class_scores' in response):
                return Response({'predict_result': response['predict_result'], 'class_scores': response['class_scores']})
            else:
                return Response({'predict_result': response})

    def batch_predict(self, features, model_name, class_count):
        class_scores = []
        predict_results = []
        errors = []
        for i in range(0, len(features), self.batch_size):
            futures = []
            feature_batch = features[i:i+self.batch_size]
            payload = {
                'api_key': self.api_key,
                'features': feature_batch,
                'model_name': model_name,
                'class_count': class_count
            }
            futures.append(self.session.post(f'{self.base_url}{self.batch_model_path}{self.predict_path}', json=payload))

            for future in futures:
                response = json.loads(future.result().content)
                if ('class_scores' in response) and ('predict_result' in response):
                    class_scores += response['class_scores']
                    predict_results += response['predict_result']
                else:
                    class_scores += [None]
                    predict_results += [None]
                    errors.append(response)

        return Response({'predict_result': predict_results, 'class_scores': class_scores, 'errors': errors})

    def learn(self, training_data, model_name, class_count):
        self._validate_learn_fields(training_data, model_name, class_count)

        if isinstance(training_data, list):
            return self.batch_learn(training_data, model_name, class_count)
        else:
            payload = {
                'api_key': self.api_key,
                'features': training_data['features'],
                'model_name': model_name,
                'class_count': class_count,
                'label': training_data['label']
            }
            future = self.session.post(f'{self.base_url}{self.model_path}{self.learn_path}', json=payload)
            return Response({'content': json.loads(future.result().content)})

    def batch_learn(self, training_data, model_name, class_count):
        responses = []
        for i in range(0, len(training_data), self.batch_size):
            futures = []
            training_batch = training_data[i:i+self.batch_size]
            payload = {
                'api_key': self.api_key,
                'training_data': training_batch,
                'model_name': model_name,
                'class_count': class_count,
            }
            futures.append(self.session.post(f'{self.base_url}{self.batch_model_path}{self.learn_path}', json=payload))

            for future in futures:
                responses.append(json.loads(future.result().content))

        return Response({'content': responses})

    def _validate_predict_fields(self, features, model_name, class_count):
        if not (isinstance(features, dict) or isinstance(features, list)):
            raise ValueError(f'features must be a dict or a list of dicts')

        if len(features) == 0:
            raise ValueError(f'features must not be empty')

        if not isinstance(model_name, str):
            raise ValueError(f'model_name must be a string')

        if not isinstance(class_count, int):
            raise ValueError(f'class_count must be an integer')

    def _validate_learn_fields(self, training_data, model_name, class_count):

        if not (isinstance(training_data, dict) or isinstance(training_data, list)):
            raise ValueError(f'training_data must be a dict or a list of dicts containing features and an associated label.')

        if len(training_data) == 0:
            raise ValueError(f'training data must not be empty')

        if not isinstance(model_name, str):
            raise ValueError(f'model_name must be a string')

        if not isinstance(class_count, int):
            raise ValueError(f'class_count must be an integer')


class Regression:
    session = FuturesSession()
    base_url = 'https://api.mlrequest.com'
    model_path = '/v1/regression/'
    batch_model_path = '/v1/regression/batch/'
    predict_path = 'predict'
    learn_path = 'learn'
    batch_size = 100

    def __init__(self, api_key=None):
        if not api_key:
            raise MissingAPIKey("An API Key is required. Visit https://mlrequest.com for a free or paid API Key.")
        self.api_key = api_key

    def predict(self, features, model_name):
        self._validate_predict_fields(features, model_name)

        if isinstance(features, list):
            return self.batch_predict(features, model_name)
        else:
            payload = {
                'api_key': self.api_key,
                'features': features,
                'model_name': model_name
            }
            future = self.session.post(f'{self.base_url}{self.model_path}{self.predict_path}', json=payload)

            response = json.loads(future.result().content)
            if 'predict_result' in response:
                return Response({'predict_result': response['predict_result']})
            else:
                return Response({'content': response})

    def batch_predict(self, features, model_name):
        predict_results = []
        errors = []
        for i in range(0, len(features), self.batch_size):
            futures = []
            feature_batch = features[i:i+self.batch_size]
            payload = {
                'api_key': self.api_key,
                'features': feature_batch,
                'model_name': model_name
            }
            futures.append(self.session.post(f'{self.base_url}{self.batch_model_path}{self.predict_path}', json=payload))

            for future in futures:
                response = json.loads(future.result().content)
                if 'predict_result' in response:
                    predict_results += response['predict_result']
                else:
                    predict_results += [None]
                    errors.append(response)

        return Response({'predict_result': predict_results, 'errors': errors})

    def learn(self, training_data, model_name):
        self._validate_learn_fields(training_data, model_name)

        if isinstance(training_data, list):
            return self.batch_learn(training_data, model_name)
        else:
            payload = {
                'api_key': self.api_key,
                'features': training_data['features'],
                'model_name': model_name,
                'label': training_data['label']
            }
            future = self.session.post(f'{self.base_url}{self.model_path}{self.learn_path}', json=payload)
            return Response({'content': json.loads(future.result().content)})

    def batch_learn(self, training_data, model_name):
        responses = []
        for i in range(0, len(training_data), self.batch_size):
            futures = []
            training_batch = training_data[i:i+self.batch_size]
            payload = {
                'api_key': self.api_key,
                'training_data': training_batch,
                'model_name': model_name
            }
            futures.append(self.session.post(f'{self.base_url}{self.batch_model_path}{self.learn_path}', json=payload))

            for future in futures:
                responses.append(json.loads(future.result().content))

        return Response({'content': responses})

    def _validate_predict_fields(self, features, model_name):
        if not (isinstance(features, dict) or isinstance(features, list)):
            raise ValueError(f'features must be a dict or a list of dicts')

        if len(features) == 0:
            raise ValueError(f'features must not be empty')

        if not isinstance(model_name, str):
            raise ValueError(f'model_name must be a string')

    def _validate_learn_fields(self, training_data, model_name):

        if not (isinstance(training_data, dict) or isinstance(training_data, list)):
            raise ValueError(f'training_data must be a dict or a list of dicts containing features and an associated label.')

        if len(training_data) == 0:
            raise ValueError(f'training data must not be empty')

        if not isinstance(model_name, str):
            raise ValueError(f'model_name must be a string')


class RL:
    session = FuturesSession()
    base_url = 'https://api.mlrequest.com'
    model_path = '/v1/rl/'
    predict_path = 'predict'
    learn_path = 'reward'

    def __init__(self, api_key=None):
        if not api_key:
            raise MissingAPIKey("An API Key is required. Visit https://mlrequest.com for a free or paid API Key.")
        self.api_key = api_key

    def predict(self, features, model_name, session_id, negative_reward,
                action_count, epsilon=0.2, action_list=None):
        self._validate_predict_fields(features, model_name, session_id, negative_reward,
                                      action_count, epsilon, action_list)

        payload = {
            'api_key': self.api_key,
            'features': features,
            'model_name': model_name,
            'session_id': session_id,
            'negative_reward': negative_reward,
            'action_count': action_count,
            'epsilon': epsilon,
            'action_list': action_list
        }
        future = self.session.post(f'{self.base_url}{self.model_path}{self.predict_path}', json=payload)
        response = json.loads(future.result().content)

        if 'predict_result' in response:
            return Response({'predict_result': response['predict_result']})
        else:
            return Response({'content': response})

    def reward(self, model_name, session_id, reward):
        self._validate_reward_fields(model_name, session_id, reward)

        payload = {
            'api_key': self.api_key,
            'model_name': model_name,
            'session_id': session_id,
            'reward': reward
        }
        future = self.session.post(f'{self.base_url}{self.model_path}{self.learn_path}', json=payload)
        return Response({'content': json.loads(future.result().content)})

    def _validate_predict_fields(self, features, model_name, session_id, negative_reward,
                                 action_count, epsilon, action_list):
        if not isinstance(features, dict):
            raise ValueError(f'features must be a dict')

        if len(features) == 0:
            raise ValueError(f'features must not be empty')

        if not isinstance(model_name, str):
            raise ValueError(f'model_name must be a string')

        if not isinstance(session_id, str):
            raise ValueError(f'session_id must be a string')

        if not isinstance(negative_reward, (int, float)):
            raise ValueError(f'negative_reward must be a number')

        if (not isinstance(action_count, int)) or action_count < 0:
            raise ValueError(f'negative_reward must be an int greater than 0')

        if not isinstance(epsilon, float) or epsilon < 0 or epsilon > 1:
            raise ValueError(f'negative_reward must be an float between 0 and 1')

        if (action_list is not None) and (not isinstance(action_list, list)):
            raise ValueError(f'action_list must be None or a list')


    def _validate_reward_fields(self, model_name, session_id, reward):

        if not isinstance(model_name, str):
            raise ValueError(f'model_name must be a string')

        if not isinstance(session_id, str):
            raise ValueError(f'session_id must be a string')

        if not isinstance(reward, (int, float)):
            raise ValueError(f'model_name must be a number')


class Account:
    session = FuturesSession()
    base_url = 'https://api.mlrequest.com'

    def __init__(self, api_key=None):
        if not api_key:
            raise MissingAPIKey("An API Key is required. Visit https://mlrequest.com for a free or paid API Key.")
        self.api_key = api_key

    def delete_model(self, model_name):
        payload = {
            'api_key': self.api_key,
            'model_name': model_name
        }
        future = self.session.post(f'{self.base_url}/v1/model/delete', json=payload)
        response = json.loads(future.result().content)
        return Response({'content': response})

    def get_details(self):
        payload = {
            'api_key': self.api_key
        }
        future = self.session.post(f'{self.base_url}/v1/account', json=payload)
        response = json.loads(future.result().content)
        return Response({'content': response})


class Response(dict):
    def __init__(self, *args, **kwargs):
        super(Response, self).__init__(*args, **kwargs)
        self.__dict__ = self


class MissingAPIKey(Exception):
    pass
