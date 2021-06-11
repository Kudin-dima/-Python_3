#Наивный байесовский классификатор по диагностической базе данных рака молочной железы в Висконсине
import sklearn
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 

data = load_breast_cancer() # загрузка набора данных

label_names = data['target_names'] 
labels = data['target']
feature_names = data['feature_names'] 
features = data['data'] 

print(label_names, "\n")
print(labels[0], "\n")

#создадуние имен и значений функций
print(feature_names[0], "\n")
print(features[0], "\n")

#разделение данных на две части, обучающий набор и тестовый набор, используем 40% данных для тестирования,оставшиеся отанутся для обучения
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.40, random_state = 42)

gnb = GaussianNB()
model = gnb.fit(train, train_labels) 

#оценка модели, благодаря прогнозам на тестовых данных, за одно узнаем точность.
preds = gnb.predict(test)
print(preds, "\n")
print(accuracy_score(test_labels, preds), "\n")

#оценка модели по прогнозам на данных для обучения  
preds = gnb.predict(train)
print(preds, "\n")
print(accuracy_score(train_labels, preds), "\n")
