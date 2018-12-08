import pandas as pd
#from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from adspy_shared_utilities import plot_fruit_knn

"""
references is : https://github.com/Starignus

convert code to pycharm

"""

def main():
    fruits = pd.read_table('fruit_data_with_colors.txt')

    X = fruits[['mass', 'width', 'height']]
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
    print("--------------------------input X----------------------------")
    print(X)
    print("--------------------------output y----------------------------")
    print(y)
    print("------------------------------------------------------")
    print("------------------------X_train------------------------------")
    print("data train :", len(X_train))
    print(X_train)
    print("------------------------X_test-------------------------------")
    print("data test : ", len(X_test))
    print(X_test)
    print("------------------------y_train------------------------------")
    print("class train :", len(y_train))
    print(y_train)
    print("-------------------------y_test------------------------------")
    print("class test : ", len(y_test))
    print(y_test)
    print("------------------------------------------------------")
    print("-----------------label fruit -------------------------------------")

    print("--------- k = 5 ----------------")
    print(knn.score(X_test, y_test))
    print("-------------------------")

    print(X_test.head())

   # predict = knn.predict(X_test)
   # print(predict)
   # print(accuracy_score(y_test, predict))



    fruit_prediction = knn.predict([[100, 6.3, 8.5]])

    lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

    print(lookup_fruit_name[fruit_prediction[0]])

    knn1 = KNeighborsClassifier(n_neighbors=1)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                         weights='uniform')

    knn1.fit(X_train, y_train)

    print("--------- k = 1 ----------------")
    print(knn1.score(X_test, y_test))
    print("-------------------------")

    print(X_test.head())


    predict1 = knn1.predict(X_test)

    print(predict1)

    print(y_test)

    """dar inja ba az inke algoritm amozesh dadeh shod dadaehaye x_test ke baraye emtehan kardan ast ra
      be algoritm midahim ta bare ma pish bini konad
      ba az pishbini ma mizan shebahat pishbini algoritm knn ra ba y_test kh vagheyat ast 
      control mikonim ba metode accuacy
      va be an emtiaz midahm"""

    print(accuracy_score(y_test, predict1))

    fruit_prediction1 = knn1.predict([[100, 6.3, 8.5]])

    lookup_fruit_name1 = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

    print(lookup_fruit_name1[fruit_prediction1[0]])


main()
