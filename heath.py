import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score


class Health(pd.DataFrame):
    def __init__(self, filename='./data/healthcare-dataset-stroke-data.csv'):
        super().__init__(pd.read_csv(filename))

        self.__fixed__ = False
        self.__filename__ = filename

        # Fix "Unknown" smoking habits w/ NaN
        self.replace('Unknown', np.NaN, inplace=True)
        # Fix binary values with YES/NO
        self[['hypertension', 'heart_disease', 'stroke']].replace([0, 1], ['No', 'Yes'], inplace=True)
        # Drop row with "Other" gender
        self.drop(index=3116, axis=0, inplace=True)
        self.reset_index(drop=True, inplace=True)

    def graph(self, cat):
        if cat == 'age':
            plt.close('all')
            print(min(self['age']), max(self['age']))

    def fixNull(self, method='kNN'):
        if method not in ['del', 'mean', 'LR', 'kNN'] or self.__fixed__: return

        self.__fixed__ = True  # Tag object fixed

        if method == 'del':
            # Drop columns with NaN
            res = self.drop(['bmi', 'smoking_status'], axis=1, inplace=True)
        elif method == 'mean':
            # Replace with mean
            self.smoking_status.replace(['never smoked', 'formerly smoked', 'smokes'], [0, 1, 2], inplace=True)
            self['bmi'].fillna((self['bmi'].mean()), inplace=True)
            self['smoking_status'].fillna((np.round(self['smoking_status'].mean())), inplace=True)
            self.smoking_status.replace([0, 1, 2], ['never smoked', 'formerly smoked', 'smokes'], inplace=True)
        elif method == 'LR':
            # Replace with linear interpolation
            self.interpolate('linear', inplace=True)
        elif method == 'kNN':
            # Create training and prediction data sets
            train_cats = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                          'Residence_type',
                          'avg_glucose_level', 'stroke']
            predict_cats = ['bmi', 'smoking_status']
            train_data = self[self['smoking_status'].notna() & self['bmi'].notna()].replace(
                ['Never_worked', 'Private', 'Self-employed', 'Govt_job', 'children', 'Rural', 'Urban', 'Male', 'Female',
                 'Yes', 'No', 'never smoked', 'formerly smoked', 'smokes'],
                [0, 1, 2, 4, 5, True, False, True, False, True, False, 0, 1, 2]
            )
            predict_data = self.replace(
                ['Never_worked', 'Private', 'Self-employed', 'Govt_job', 'children', 'Rural', 'Urban', 'Male', 'Female',
                 'Yes', 'No'],
                [0, 1, 2, 4, 5, True, False, True, False, True, False]
            )

            # Fit and predict data
            model = KNeighborsRegressor(n_neighbors=25)
            model.fit(train_data[train_cats], train_data[predict_cats])
            prediction = pd.DataFrame(model.predict(predict_data[train_cats]), columns=['bmi', 'smoking_status'])

            # Fill NaN
            self['bmi'].fillna(prediction['bmi'], inplace=True)
            self['smoking_status'].fillna(np.round(prediction['smoking_status']), inplace=True)
            self['smoking_status'].replace([0, 1, 2], ['never smoked', 'formerly smoked', 'smokes'], inplace=True)

        return self


def RandomForest(dataset, exclude=[]):
    # Fix Labels
    dataset.gender = LabelEncoder().fit_transform(dataset['gender'])
    dataset.hypertension = LabelEncoder().fit_transform(dataset['hypertension'])
    dataset.heart_disease = LabelEncoder().fit_transform(dataset['heart_disease'])
    dataset.ever_married = LabelEncoder().fit_transform(dataset['ever_married'])
    dataset.work_type = LabelEncoder().fit_transform(dataset['work_type'])
    dataset.Residence_type = LabelEncoder().fit_transform(dataset['Residence_type'])
    if 'smoking_status' not in exclude:
        dataset.smoking_status = LabelEncoder().fit_transform(dataset['smoking_status'])
    dataset.stroke = LabelEncoder().fit_transform(dataset['stroke'])

    # Get train/test data
    x = dataset[[i for i in ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                             'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'] if i not in exclude]]
    y = dataset['stroke']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    model = RandomForestClassifier(n_estimators=250, bootstrap=False, warm_start=True)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print('F1: {0:.2f}%,   Precision: {1:.2f}%,   Recall: {2:.2f}%'.format(f1_score(y_test, y_pred, average='macro') * 100,
          precision_score(y_test, y_pred, average='macro') * 100, recall_score(y_test, y_pred, average='macro') * 100))

    # return f1_score(y_test, y_pred, average='macro'), precision_score(y_test, y_pred, average='macro'), \
    #        recall_score(y_test, y_pred, average='macro')


rem = Health().fixNull('del')
mean = Health().fixNull('mean')
lr = Health().fixNull('LR')
knn = Health().fixNull('kNN')

print('Remove columns with NaN'.rjust(40, ' '), end='  ->  ')
RandomForest(rem, ['bmi', 'smoking_status'])

print('Replace NaN with column mean'.rjust(40, ' '), end='  ->  ')
RandomForest(mean)

print('Replace NaN using Linear Regression'.rjust(40, ' '), end='  ->  ')
RandomForest(lr)

print('Replace NaN using k-Nearest Neighbours'.rjust(40, ' '), end='  ->  ')
RandomForest(knn)
