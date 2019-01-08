import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# from adspy_shared_utilities import plot_fruit_knn

"""
previous references is : references is : https://github.com/Starignus
references is : https://github.com/mhsedaghat/knn_to_pycharm_for_quiz_Sedaghat

convert code to pycharm

"""


def main():
    fruits = pd.read_table('fruit_data_with_colors.txt')

    X = fruits[['mass', 'width', 'height']]  # *** ke mitone 2 class bashad X = fruits[['mass', 'width']] ***
    # amma bayad dar hengam moghayese
    # dar edame kod mad nazar gharar girad
    """    ^
           |
           |
      dar bala tain tedade class haye vorodi va 

      dar pain tain tedade class khoroji 
           |
          \/
      """

    y = fruits['fruit_label']

    lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
    print(lookup_fruit_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                         weights='uniform')

    knn.fit(X_train, y_train)

    print("--------------------------All fruit----------------------------")
    print(fruits)
    print("--------------------------All input X----------------------------")
    print(X)
    print("--------------------------All output y----------------------------")
    print(y)
    print("------------------------------------------------------")
    print("------------------------input X_train------------------------------")
    print("input data train :", len(X_train))
    print(X_train)
    print("------------------------input X_test-------------------------------")
    print("input data test : ", len(X_test))
    print(X_test)
    print("------------------------output y_train------------------------------")
    print("output class train :", len(y_train))
    print(y_train)
    print("-------------------------output y_test------------------------------")
    print("output class test : ", len(y_test))
    print(y_test)
    print("------------------------------------------------------")

    print("")
    print("")
    print("")
    print("--------- k = 5 ----------------")
    print("Accuracy :", knn.score(X_test, y_test))
    print("-------------------------")
    # show five number of data from x_test
    print(" show five number of data from x_test ")
    print(" VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV ")

    print(X_test.head())
    # predict = knn.predict(X_test)
    # print(predict)
    # print(accuracy_score(y_test, predict))

    # moarefi yek miveh baraye pishbini

    fruit_prediction = knn.predict([[100, 6.3, 8.5]])  # gharar dadan dar andis 0 fruit_prediction

    lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
    print("K = 5")
    print("test algorithm with one of fruits")
    print("mass : 100, width : 6.3, height : 8.5")

    print("----------- label fruit --------------")
    print(lookup_fruit_name[fruit_prediction[0]])  # namayesh dadan az andis 0 fruit_prediction
    print("-------------------------")

    knn1 = KNeighborsClassifier(n_neighbors=1)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                         weights='uniform')

    knn1.fit(X_train, y_train)
    print("")
    print("")
    print("")
    print("--------- k = 1 ----------------")
    print("Accuracy :", knn1.score(X_test, y_test))  # moghayese ba x_train va y_train
    print("-------------------------")

    print(X_test.head())

    print("-------------------------")

    predict1 = knn1.predict(X_test)  # moghayese andis be andis dadeha test ba train ke natige 6 ra dadeh ast
    # x test ra be vorodi dadehim va ba mohasebeh i ke anjam midahad khoroji mishavad  perdict1
    print(predict1)
    # hala agar ma khoroje y test ra negah konim ke besorat sotoni neshan dadaeh yek shabahathaii beyne khoroji
    # khoroji vaghe ba khoroji bala ke be sorate satri ast mibinim
    print(y_test)
    # dar pain mizan shebahat beyne in do khoroji ba adad 0.6 namayesh dadeh ast

    """dar inja ba az inke algoritm amozesh dadeh shod dadaehaye x_test ke baraye emtehan kardan ast ra
      be algoritm midahim ta bare ma pish bini konad
      ba az pishbini ma mizan shebahat pishbini algoritm knn ra ba y_test kh vagheyat ast 
      control mikonim ba metode accuacy
      va be an emtiaz midahm"""

    print(accuracy_score(y_test, predict1))

    # va hala dar pain yek bar digar chek ikonim

    fruit_prediction1 = knn1.predict([[100, 6.3, 8.5]])  # *** agar dar bala 2 class tarif kardim code mishavad ***
    # fruit_prediction1 = knn1.predict([[100, 6.3]])
    # inja bayad 2 class bedahim

    lookup_fruit_name1 = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
    print("K = 1")
    print("test again algorithm with one of fruits")
    print("mass : 100, width : 6.3, height : 8.5")
    print("----------- label fruit --------------")
    print(lookup_fruit_name1[fruit_prediction1[0]])
    print("-------------------------")


main()
