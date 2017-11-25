import nltk
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class Utils():
    @staticmethod
    def count(x):
        from collections import Counter
        c = Counter(x)
        return c


class Dataset():
    def __init__(self, filename=None, test_size=None):
        self.raw = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.classes = None

        if filename:
            self.read(filename=filename, test_size=test_size)

    def read(self, filename, test_size=None):
        self.raw = load_files(filename, description=None,
                              load_content=True,
                              encoding='utf-8',
                              decode_error='ignore')

        self.classes = self.raw['target_names']
        if test_size is not None:
            self.split(test_size=test_size)

    def split(self, test_size=0.1):
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.raw.data,
                                                                            self.raw.target,
                                                                            test_size=test_size,
                                                                            random_state=42)

    def get_train_data(self):
        return [self.trainX, self.trainY]

    def get_test_data(self):
        return [self.testX, self.testY]

    def num_classes(self):
        if self.classes is None:
            return 0
        return len(self.classes)

    def num_data(self):
        if self.raw.target is None:
            return 0
        return len(self.raw.target)

    def info(self):
        print("No. of classes: {}".format(self.num_classes()))
        print ("Class labels: {}".format(self.classes))
        print ("Total data samples: {}".format(self.num_data()))

        if self.trainY is not None:
            print("Train samples: {}".format(len(self.trainY)))
            trainStat = Utils.count(self.trainY)
            for k in trainStat.keys():
                print("\t {}:{} = {}".format(k, self.classes[k], trainStat.get(k, 0)))

        if self.testY is not None:
            print ("Test stats: {}".format(len(self.testY)))
            testStat = Utils.count(self.testY)
            for k in testStat.keys():
                print("\t {}:{} = {}".format(k, self.classes[k], testStat.get(k, 0)))


class TfIdfFeatureExtractor():
    def __init__(self, preprocessor=None, stop_words=None):
        self.extractor = TfidfVectorizer(tokenizer=lambda x: x.split(" "),
                                         sublinear_tf=True, encoding='utf-8',
                                         decode_error='ignore',
                                         preprocessor=preprocessor,
                                         max_df=0.5,
                                         min_df=10,
                                         stop_words=stop_words)

    def get_extractor(self):
        return self.extractor

    def extract(self, data):
        return self.extractor.fit_transform(data)


class NewsClassifier():
    def __init__(self):
        self.preprocessor = None
        self.feature_extractor = None
        self.setup()

    def setup(self):
        # The stop words file is copied into the stopwords directory of nltk.data\corpora folder
        stop_words = set(nltk.corpus.stopwords.words('nepali'))
        self.feature_extractor = TfIdfFeatureExtractor(stop_words=stop_words, preprocessor=self.preprocessor)

    def grid_search(self, estimator, param_grid, features, targets):
        print("\nGrid search for algorithm:  {}".format(estimator))
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, verbose=10, n_jobs=6)
        grid.fit(features, targets)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        return grid

    def train_and_evaluate(self, estimator, trainX, trainY, testX, testY):
        estimator.fit(trainX, trainY)
        print("Accuracy on train Set: ")
        print(estimator.score(trainX, trainY))
        print("Accuracy on Test Set: ")
        print(estimator.score(testX, testY))
        outputs = estimator.predict(testX)
        print("Classification Report: ")
        print(metrics.classification_report(testY, outputs))
        print("Confusion Matrix: ")
        print(metrics.confusion_matrix(testY, outputs))

    def naive_bayes(self, trainX, trainY, testX, testY, grid_search=False, train=True):
        print('\nMultinominal Naive Bayes')
        alpha = 0.03
        if grid_search:
            estimator = MultinomialNB(alpha=alpha, fit_prior=True)
            alpha_range = [0.001, 0.002, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.08, 0.09, 0.1, 0.5, 1, 1.2,
                           1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 100, 500, 1000]

            param_grid = dict(alpha=alpha_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=self.feature_extractor.extract(trainX), targets=trainY)
            alpha = grid.best_params_['alpha']

            # 20 nepali news data
            # The best parameters are  for naive {'alpha': 0.03} with a score of 0.70

        if train:
            estimator = MultinomialNB(alpha=alpha)
            clf = Pipeline([
                ('vect', self.feature_extractor.get_extractor()),
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)

    def svm_linear(self, trainX, trainY, testX, testY, grid_search=False, train=True):
        print('\nSVM with Linear Kernel')
        c = 1.5
        gamma = 'auto'
        if grid_search:
            estimator = SVC(kernel='linear', random_state=42, verbose=False, C=c, gamma=gamma)
            C_range = [0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 100, 500, 1000]
            gamma_range = [0.001, 0.01, 0.1, 1, 2, 3, "auto"]
            param_grid = dict(gamma=gamma_range, C=C_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=self.feature_extractor.extract(trainX), targets=trainY)
            c = grid.best_params_['C']
            gamma = grid.best_params_['gamma']

            # 20 nepali news data
            # The best parameters are  for linear svm {'C': 1.5, 'gamma': 0.001} with a score of 0.75

        if train:
            estimator = SVC(kernel='linear', random_state=42, verbose=False, C=c, gamma=gamma)
            clf = Pipeline([
                ('vect', self.feature_extractor.get_extractor()),
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)

    def svm_rbf(self, trainX, trainY, testX, testY, grid_search=False, train=True):
        print('\nSVM with RBF Kernel')
        c = 100
        gamma = 0.01
        if grid_search:
            estimator = SVC(kernel='rbf', random_state=42, verbose=False, C=c, gamma=gamma)
            C_range = [0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 100, 500, 1000]
            gamma_range = [0.001, 0.01, 0.1, 1, 2, 3, "auto"]
            param_grid = dict(gamma=gamma_range, C=C_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=self.feature_extractor.extract(trainX), targets=trainY)
            c = grid.best_params_['C']
            gamma = grid.best_params_['gamma']

            # 20 nepali news data
            # # The best parameters are for rbf svm  {'C': 100, 'gamma': 0.01} with a score of 0.75

        if train:
            estimator = SVC(kernel='rbf', random_state=42, verbose=False, C=c, gamma=gamma)
            clf = Pipeline([
                ('vect', self.feature_extractor.get_extractor()),
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)

    def mlp(self, trainX, trainY, testX, testY, grid_search=False, train=True):
        print('\nMLP Neural Network')
        solver = 'adam'
        alpha = 0.000001
        learning_rate = 'adaptive'
        learning_rate_init = 0.0025
        momentum = 0.9
        hidden_layer_sizes = (256,)
        max_iter = 1000
        early_stopping = True
        if grid_search:
            estimator = MLPClassifier(solver=solver, alpha=alpha, learning_rate=learning_rate,
                                      learning_rate_init=learning_rate_init, momentum=momentum,
                                      hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42,
                                      verbose=False, early_stopping=early_stopping)
            solver_range = ['adam']
            alpha_range = [1e-6, 1e-5, 0.00001, 0.0001, 0.0005, 0.001, 0.002, 0.01, 0.1, 0.5, 1, 1.5]
            learning_rate_range = ['constant', 'adaptive']
            max_iter_range = [200, 500, 1000]
            momentum_range = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
            early_stopping_range = [True, False]
            learning_rate_init_range = [0.0001, 0.001, 0.0025, 0.01, 0.1, 1]
            hidden_layer_sizes_range = [(100,), (100, 50), (128, 64), (256, 64), (256, 128, 64)]

            param_grid = dict(solver=solver_range, alpha=alpha_range, learning_rate=learning_rate_range,
                              learning_rate_init=learning_rate_init_range,
                              hidden_layer_sizes=hidden_layer_sizes_range, max_iter=max_iter_range,
                              momentum=momentum_range, early_stopping=early_stopping_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=self.feature_extractor.extract(trainX), targets=trainY)
            solver = grid.best_params_['solver']
            alpha = grid.best_params_['alpha']
            learning_rate = grid.best_params_['learning_rate']
            learning_rate_init = grid.best_params_['learning_rate_init']
            momentum = grid.best_params_['momentum']
            hidden_layer_sizes = grid.best_params_['hidden_layer_sizes']
            max_iter = grid.best_params_['max_iter']
            early_stopping = grid.best_params_[early_stopping]

            # 20 nepali news data
            # hidden_layer_sizes=(100,), learning_rate_init=0.001, learning_rate=constant, max_iter=100,
            #  early_stopping=False, alpha=1e-06, momentum=0.7,0.9, score=0.737506, total= 5.2min

        if train:
            estimator = MLPClassifier(solver=solver, alpha=alpha, learning_rate=learning_rate,
                                      learning_rate_init=learning_rate_init, momentum=momentum,
                                      hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42,
                                      verbose=False, early_stopping=early_stopping)
            clf = Pipeline([
                ('vect', self.feature_extractor.get_extractor()),
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)


if __name__ == '__main__':
    train = True
    grid_search = False
    estimators = ['naive', 'svm_linear', 'svm_rbf', 'mlp']  # ['naive', 'svm_linear', 'svm_rbf', 'mlp']

    filename = 'data/20NepaliNews/'

    dataset = Dataset(filename=filename, test_size=0.2)
    print(dataset.info())
    assert train or grid_search, "Enable the training or grid_search."
    nc = NewsClassifier()
    for estimator in estimators:
        if estimator == 'naive':
            nc.naive_bayes(trainX=dataset.trainX, trainY=dataset.trainY, testX=dataset.testX, testY=dataset.testY,
                           grid_search=grid_search, train=train)
        elif estimator == 'svm_linear':
            nc.svm_linear(trainX=dataset.trainX, trainY=dataset.trainY, testX=dataset.testX, testY=dataset.testY,
                          grid_search=grid_search, train=train)
        elif estimator == 'svm_rbf':
            nc.svm_rbf(trainX=dataset.trainX, trainY=dataset.trainY, testX=dataset.testX, testY=dataset.testY,
                       grid_search=grid_search, train=train)
        elif estimator == 'mlp':
            nc.mlp(trainX=dataset.trainX, trainY=dataset.trainY, testX=dataset.testX, testY=dataset.testY,
                   grid_search=grid_search, train=train)
        else:
            print("Unknown estimator: {}".format(estimator))
