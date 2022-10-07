from sklearn.datasets import load_iris
iris_dataset = load_iris()

print(f"Ключі iris_dataset: \n{iris_dataset.keys()}")
print(iris_dataset['DESCR'][:193] + "\n...")

print(f"Назви відповідей: {iris_dataset['target_names']}")
print(f"Назви ознак: {iris_dataset['feature_names']}")
print(f"Тип даних: {type(iris_dataset['data'])}")
print(f"Розмір даних: {iris_dataset['data'].shape}")
print(f"Перші п'ять рядків даних:\n{iris_dataset['data'][:5]}")
print(f"Тип масиву відповідей: {type(iris_dataset['target'])}")
print(f"Розмір масиву відповідей: {iris_dataset['target'].shape}")
print(f"Відповіді:\n{iris_dataset['target']}")
