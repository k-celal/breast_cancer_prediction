import streamlit as st
from data_preprocess import DataPreprocess
from sklearn.metrics import  ConfusionMatrixDisplay
from model_training import ModelTrainer
class App:
    def __init__(self):
        self.dataset_name = None
        self.dataPreProcess = DataPreprocess()
        self.modelTrainer=None
        self.classifier_name = None
        self.init_streamlit_page()
        self.params = dict()
        self.clf = None
        self.X_train, self.X_test, self.y_train, self.y_test= None,None,None,None
    def run(self):
        self.get_dataset()
        self.get_dataprocess()
        self.add_parameter_ui()
        self.generate()
        self.grid_Search_Train()

    def init_streamlit_page(self):
        st.title('Streamlit Breast Cancer Pred')

        st.write("""
                # Explore different classifier and datasets
                Which one is the best?
                """)

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer',)
        )
        st.write(f"## {self.dataset_name} Dataset")

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Naive Bayes')
        )
    def get_dataset(self):
        data = None
        if self.dataset_name == 'Breast Cancer':
            self.dataPreProcess.load_data('data/breast_cancer_wisconsin(diagnostic)_data.csv')
            data=self.dataPreProcess.data
        st.write("İlk 10 Satır:")
        st.write(data.head(10))
        st.write("Sütunlar:")
        st.write(data.columns)

    def get_dataprocess(self):
        self.dataPreProcess.clean_data()
        self.dataPreProcess.encode_diagnosis()
        st.write("Son 10 Satır:")
        st.write(self.dataPreProcess.show_last_rows())
        st.subheader('Korelasyon Matrisi')
        correlation_image_data = self.dataPreProcess.plot_correlation_matrix()
        st.image(correlation_image_data, caption='Correlation Matrix', use_column_width=True)
        st.subheader('diagnosis_scatter')
        correlation_image_data = self.dataPreProcess.plot_diagnosis_scatter()
        st.image(correlation_image_data, caption='diagnosis_scatter', use_column_width=True)
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataPreProcess.split_data(0.2, 1234)

        self.modelTrainer = ModelTrainer(self.X_train,self.y_train,self.X_test,self.y_test,self.classifier_name,self.params)
    def add_parameter_ui(self):
        if self.classifier_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 100.0)
            kernel = st.sidebar.selectbox('Kernel', ['linear', 'rbf', 'poly'])
            self.params['C'] = C
            self.params['kernel']=kernel
            self.modelTrainer.X_train,self.modelTrainer.X_test=self.dataPreProcess.scale_data(self.X_train,self.X_test)
        elif self.classifier_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            self.modelTrainer.X_train, self.modelTrainer.X_test = self.dataPreProcess.scale_data(self.X_train,self.X_test)
            self.params['K'] = K
        else:
            var_smoothing = st.sidebar.slider('Var Smoothing', 1e-10, 1e-6, format="%.1e")
            self.params['var_smoothing'] = var_smoothing

    def generate(self):
        self.modelTrainer.train_model()
        accuracy, precision, recall, f1, confusion = self.modelTrainer.evaluate_model()

        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy =', accuracy)
        st.write(f'Precision =', precision)
        st.write(f'Recall =', recall)
        st.write(f'f1 =', f1)
        st.write(f'Confusion: ',confusion)
        self.plot_confusion_matrix(confusion)
        st.write('-------------------------------------')


    def grid_Search_Train(self):
        best_params,accuracy, precision, recall, f1, confusion = self.modelTrainer.grid_search_train_model()
        st.write(f'En iyi parametreler: ', best_params)
        st.write(f'En iyi parametreler ile score = ', accuracy)
        st.write(f'En iyi parametreler ile Precision = ', precision)
        st.write(f'En iyi parametreler ile Recall = ', recall)
        st.write(f'En iyi parametreler ile f1 = ', f1)
        st.write(f'En iyi parametreler ile confusion: ',confusion)
        self.plot_confusion_matrix(confusion)

    def plot_confusion_matrix(self, confusion):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        disp=ConfusionMatrixDisplay(confusion_matrix=confusion)
        disp.plot()
        st.pyplot()
        # print(confusion)
        # sns.heatmap(confusion, annot=True)
        # plt.xlabel('y_pred')
        # plt.ylabel('y_true')
        # plt.title('Confusion Matrix')
        # plt.tight_layout()
        # plt.savefig("deneme",format='png')