
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier



from sklearn.naive_bayes import MultinomialNB
from read_file import read_data
from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier


#Accuracy based on Splitting Data
def Data_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

    model1 = MultinomialNB()
    model1.fit(X_train,y_train)
    print("Naive Bayes Accuracy: ",model1.score(X_test, y_test))

    model2 = GradientBoostingClassifier()
    model2.fit(X_train, y_train)
    print("Gradient Boosting Classifier's Accuracy = " + str(model2.score(X_test, y_test)))

    model3=SVC()
    model3.fit(X_train, y_train)
    print("SVC Accuracy: ",model3.score(X_test, y_test))

    model4=LogisticRegression()
    model4.fit(X_train,y_train)
    print("Logistic Regression Accuracy: ",model4.score(X_test, y_test))

    model5 = KNeighborsClassifier(n_neighbors=2)
    model5.fit(X_train, y_train)
    print("Kneighbour Classifier's Accuracy = " + str(model5.score(X_test, y_test)))

    model6 = ExtraTreesClassifier()
    model6.fit(X_train, y_train)
    print("Extra-tree Classifier's Accuracy = " + str(model6.score(X_test, y_test)))

    model7 = DecisionTreeClassifier()
    model7.fit(X_train, y_train)
    print("Decision-tree Classifier's Accuracy = " + str(model7.score(X_test, y_test)))

#Accuracy based on Cross validation
def Cross_Validation(X,y):
    model1 = MultinomialNB()
    scores1 = cross_val_score(model1, X, y, cv=5, scoring='accuracy')
    print ("Naive  Bayes with 5 Cross validation Accuracy:" ,(np.mean(np.sqrt(abs(scores1)))))

    model2 = GradientBoostingClassifier()
    scores2 = cross_val_score(model2, X, y, cv=5, scoring='accuracy')
    print ("Gradient Boost with 5 Cross validation Accuracy:" ,(np.mean(np.sqrt(abs(scores2)))))

    model3=SVC()
    scores3 = cross_val_score(model3, X, y, cv=5, scoring='accuracy')
    print ("SVC with 5 Cross validation Accuracy:" ,(np.mean(np.sqrt(abs(scores3)))))

    model4=LogisticRegression()
    scores4 = cross_val_score(model4, X, y, cv=5, scoring='accuracy')
    print ("Logistic Regression with 5 Cross validation Accuracy:" ,(np.mean(np.sqrt(abs(scores4)))))

    model5 = KNeighborsClassifier(n_neighbors=2)
    scores5 = cross_val_score(model5, X, y, cv=5, scoring='accuracy')
    print ("K-Neighbours-Classifier with 5 Cross validation Accuracy:" ,(np.mean(np.sqrt(abs(scores5)))))

    model6 = ExtraTreesClassifier()
    scores6 =cross_val_score(model6, X, y, cv=5, scoring='accuracy')
    print ("Tree-Classifier with 5 Cross validation Accuracy:" ,(np.mean(np.sqrt(abs(scores6)))))

    model7 = DecisionTreeClassifier()
    scores7 = cross_val_score(model7, X, y, cv=5, scoring='accuracy')
    print ("Decision-Tree-Classifier with 5 Cross validation Accuracy:" ,(np.mean(np.sqrt(abs(scores7)))))

    model8 = MLPClassifier(solver='adam', alpha=0.01, hidden_layer_sizes=(10, 10))
    scores8 = cross_val_score(model8, X, y, cv=5, scoring='accuracy')
    print("MLP classifier's with 5 Cross validation Accuracy:" ,(np.mean(np.sqrt(abs(scores8)))))



X,y = read_data("/home/looser/Downloads/ODI-Cricket-Match-Result-Prediction-master/data/final_dataset.csv")
np.random.seed(0)
Cross_Validation(X,y)
