import pandas as pd
import numpy as np
import matplotlib as plt
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def model(X_train, y_train):
    
    forest = RandomForestClassifier( n_estimators=10, random_state=0)
    forest.fit(X_train,y_train)
    print("Las: {0}".format(forest.score(X_train,y_train)) )
    
    lreg =LogisticRegression()
    lreg.fit(X_train,y_train)
    print("Regresja logistyczna: {0}".format(lreg.score(X_train,y_train)) )
  
    tree =DecisionTreeClassifier()
    tree.fit(X_train,y_train)
    print("Drzewa decyzyjne: {0}".format(tree.score(X_train,y_train)) )

    return forest, lreg, tree


def main():
    ''' Pobranie danych '''
    base_data = pd.read_csv('ml/colors.csv')

    ''' Wybór kolumn '''
    cols = ["R","G","B","Name"]
    data = base_data[cols].copy()

    ''' eksploracja i uzupełnienie danych '''
    #print(data.isnull().any())
    #sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    #plt.show()
    
    data["Age"].fillna((data["Age"].mean()), inplace=True) # Fill missing age values with the mean
    data["Embarked"].fillna("C", inplace=True)

    #print(data.isnull().any())
    #sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    #plt.show()

    encoder = LabelEncoder()
    data.loc[:,"Sex"] = encoder.fit_transform(data.loc[:,"Sex"])
    # male = 1, female = 0

    encoder = LabelEncoder()
    data.loc[:,"Embarked"] = encoder.fit_transform(data.loc[:,"Embarked"])
    # C = Cherbourg = 0, Q = Queenstown = 1, S = Southampton = 2

    
    '''
    #Kto przeżył
    sns.set_style('whitegrid')
    sns.countplot(x='Survived',data=data)
    plt.show()
    '''
    
    '''
    #Kto przeżył -> w zależności od płci
    sns.set_style('whitegrid')
    sns.countplot(x='Survived',hue='Sex',data=data)
    plt.show()
    '''
    
    '''
    #boxplot dla wieku oraz klasy pasażerów
    plt.figure(figsize=(10, 10))
    sns.boxplot(x='Pclass',y='Age',data=data)
    plt.show()
    '''

    ''' Trenowanie modelu '''
    y = data.iloc[:,0] # survived - zmienna, którą będziemy chcieli przewidzieć
    x = data.iloc[:,1:8] # zmienne na podstawie, których chcemy przewidzieć
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    # test size odnosi się do liczby obserwacji przeznaczonej do wytrenowania modelu

    forest, logistic_regression, tree = model(X_train, y_train)

    ''' ocena modelów '''
    target_names = ["Died", "Survived"]

    print('\n==================== \n')

    y1_predict = forest.predict(X_test)
    print("Random Forest {0}".format(accuracy_score(y_test, y1_predict)))

    y2_predict = logistic_regression.predict(X_test)
    print("Logistic Regresion {0}".format(accuracy_score(y_test, y2_predict)))

    y3_predict = tree.predict(X_test)
    print("Decision Tree {0}".format(accuracy_score(y_test, y3_predict)))

    print('\n==================== \n')

    print("\nOcena modelu 1. Las")
    print(classification_report(y_test,y1_predict))
    # uzyskujemy informację o precyzji, recall, f1, etc.

    print("\nOcena modelu 2. Regresja logistyczna")
    print(classification_report(y_test,y2_predict))

    print("\nOcena modelu 3. Drzewa decyzyjne")
    print(classification_report(y_test,y3_predict))

    
    ''' eksport pierwszego modelu - forest'''
    filename = "ml/forest_model.sv"
    pickle.dump(forest, open(filename,'wb'))


    '''przykładowe przewidywanie'''
    my_data =[
            [
             2,  #"Pclass"
             1,  #"Sex", Sex 0 = Female, 1 = Male
             23,  #"Age", Age
             1,  #"SibSp"
             0,  #"Parch"
             0,  #"Fare", 
             2,  #"Embarked"
        ]
    ]

    res = forest.predict(my_data)
    if(res[0] == 0):
        print('Niestety - nie przeżyjesz :(')
    else:
        print('Uratowałeś się :)')





if __name__ == '__main__':
    main()