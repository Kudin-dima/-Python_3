#Наивный байесовский классификатор по диагностической базе данных рака молочной железы в Висконсине
import sklearn
from sklearn.datasets import load_breast_cancer # диагностическая базу данных рака молочной железы
from sklearn.model_selection import train_test_split #импорт функции для разделения данных на наборы
from sklearn.naive_bayes import GaussianNB #наивный байесовский алгоритм для построения модели
from sklearn.metrics import accuracy_score #для определения точности

data = load_breast_cancer() # загрузка набора данных

label_names = data['target_names'] # Имена меток классификации
labels = data['target'] # Фактические метки
feature_names = data['feature_names'] # Имена атрибутов / функций
features = data['data'] # Атрибут

print(label_names, "\n")# Печатаем имена классов
print(labels[0], "\n") #отображение в двоичном значении 0-злокачеств. 1-доброкачеств.

#создадуние имен и значений функций
print(feature_names[0], "\n")
print(features[0], "\n")

#разделение данных на две части, обучающий набор и тестовый набор, используем 40% данных для тестирования,оставшиеся отанутся для обучения
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.40, random_state = 42)

gnb = GaussianNB() #инициализация модели
model = gnb.fit(train, train_labels) # обучение модели, подгоняя ее к данным с помощью gnb.fit()

#оценка модели, благодаря прогнозам на тестовых данных, за одно узнаем точность.
preds = gnb.predict(test)# для прогнозирования используем функцию предиката(отвечают на какой-то вопрос и возвращают либо true , либо false)
print(preds, "\n")
print(accuracy_score(test_labels, preds), "\n")# процент точности модели

#оценка модели по прогнозам на данных для обучения  
preds = gnb.predict(train)
print(preds, "\n")
print(accuracy_score(train_labels, preds), "\n")