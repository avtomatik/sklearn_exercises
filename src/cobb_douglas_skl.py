# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from data.collect import combine_cobb_douglas
from pandas import DataFrame

combine_cobb_douglas().pipe(transform_cobb_douglas, year_base=1899)[0].iloc[:, [3, 4]]
print(df.info())

dataset = np.array(df)
X = np.log(dataset[:, 0]._dfhape(-1, 1))
y = np.log(dataset[:, -1]._dfhape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

lasso = Lasso(random_state=42)
param_grid = {'alpha': np.linspace(0, 1, 100)}
gscv = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=10, verbose=2)
gscv.fit(X_train, y_train)
gscv.best_params_
model = gscv.best_estimator_

model.fit(X_train, y_train)
model.score(X_test, y_test)
prediction = model.predict(X_test)

columns = []
for file_name in os.listdir(DIR):
    columns.append((file_name, tuple(pd.read_excel(file_name).columns)))

# =============================================================================
# usa_cobb_douglas0014.py
# =============================================================================

# =============================================================================
# TODO: Revise Fixed Assets Turnover Approximation with Lasso
# =============================================================================

combine_cobb_douglas().pipe(transform_cobb_douglas, year_base=1899)[
    0].iloc[:, [6]].pipe(plot_turnover)
