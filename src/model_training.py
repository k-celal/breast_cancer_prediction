from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, model_type, params):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
        self.params = params
        self.model = None

    def train_model(self):
        if self.model_type == 'SVM':
            self.model = SVC(C=self.params['C'],kernel=self.params['kernel'])
        elif self.model_type == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=self.params['K'])
        else:
            self.model = GaussianNB(var_smoothing=self.params['var_smoothing'])
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model eğitilmedi!")
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)

        return accuracy, precision, recall, f1, confusion
    def grid_search_train_model(self):
        if self.model_type == 'SVM':
            grid_params = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly']}
        elif self.model_type == 'KNN':
            grid_params = {'n_neighbors': range(1, 16)}
        else:
            grid_params = {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]}
        grid_search = GridSearchCV(estimator=self.model, param_grid=grid_params, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        accuracy, precision, recall, f1, confusion=self.evaluate_model()
        return best_params,accuracy, precision, recall, f1, confusion