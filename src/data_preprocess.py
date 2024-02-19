from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
class DataPreprocess:
    data=None
    def __init__(self):
        self.data=None
        self.scaler = StandardScaler()
        self.pca = PCA()
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)

    def clean_data(self):
        self.data = self.data.drop(['id'],axis=1)
        self.data = self.data.drop(['Unnamed: 32'], axis=1)

    def show_last_rows(self, n=10):
        return self.data.tail(n)

    def encode_diagnosis(self):
        self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})

    def separate_features_labels(self):
        X = self.data.drop(['diagnosis'], axis=1)
        y = self.data['diagnosis']
        return X, y

    def plot_correlation_matrix(self):
        corr = self.data.corr()
        sns.set(font_scale=1.2)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.xlabel('Features')
        plt.ylabel('Features')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, format='png')
        plt.close()
        with open(temp_file.name, 'rb') as f:
            image_data = f.read()

        return image_data

    def plot_diagnosis_scatter(self):
        malignant_data = self.data[self.data['diagnosis'] == 1]
        benign_data = self.data[self.data['diagnosis'] == 0]

        plt.figure(figsize=(10, 8))
        plt.scatter(malignant_data['radius_mean'], malignant_data['texture_mean'], color='red', label='Malignant')
        plt.scatter(benign_data['radius_mean'], benign_data['texture_mean'], color='green', label='Benign')

        plt.xlabel('Radius Mean')
        plt.ylabel('Texture Mean')
        plt.title('Malignant vs Benign Tumors Scatter Plot')

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, format='png')
        plt.close()
        with open(temp_file.name, 'rb') as f:
            image_data = f.read()

        return image_data

    def split_data(self, test_size=0.2, random_state=42):
        X, y = self.separate_features_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def apply_pca(self, X_train_scaled, X_test_scaled):
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        return X_train_pca, X_test_pca