from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

# 1. 데이터 준비
raw_iris = datasets.load_iris()
X = raw_iris.data
y = raw_iris.target
pd.DataFrame(X).head()

# 2. 데이터 전처리
X_tn, X_te, y_tn, y_te = train_test_split(X, y, random_state = 1)
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

# 3. 모델 학습
clf_svm_ln = svm.SVC(kernel = 'linear', random_state = 1)
clf_svm_ln.fit(X_tn_std, y_tn)
clf_svm_sg = svm.SVC(kernel = 'sigmoid', random_state = 1)
clf_svm_sg.fit(X_tn_std, y_tn)
clf_svm_rbf = svm.SVC(kernel = 'rbf', random_state = 1)
clf_svm_rbf.fit(X_tn_std, y_tn)

cv_scores = cross_val_score(clf_svm_ln, X_tn_std, y_tn, cv = 20, scoring = 'accuracy')
print(cv_scores)
print(cv_scores.mean())
print(cv_scores.std())
cv_scores = cross_val_score(clf_svm_sg, X_tn_std, y_tn, cv = 20, scoring = 'accuracy')
print(cv_scores)
print(cv_scores.mean())
print(cv_scores.std())
cv_scores = cross_val_score(clf_svm_rbf, X_tn_std, y_tn, cv = 20, scoring = 'accuracy')
print(cv_scores)
print(cv_scores.mean())
print(cv_scores.std())

# 4. 결과 확인
pred_svm_ln = clf_svm_ln.predict(X_te_std)
print(pred_svm_ln)
pred_svm_sg = clf_svm_sg.predict(X_te_std)
print(pred_svm_sg)
pred_svm_rbf = clf_svm_rbf.predict(X_te_std)
print(pred_svm_rbf)

accuracy_ln = accuracy_score(y_te, pred_svm_ln)
print(accuracy_ln)
accuracy_sg = accuracy_score(y_te, pred_svm_sg)
print(accuracy_sg)
accuracy_rbf = accuracy_score(y_te, pred_svm_rbf)
print(accuracy_rbf)

conf_matrix_ln = confusion_matrix(y_te, pred_svm_ln)
print(conf_matrix_ln)
conf_matrix_sg = confusion_matrix(y_te, pred_svm_sg)
print(conf_matrix_sg)
conf_matrix_rbf = confusion_matrix(y_te, pred_svm_rbf)
print(conf_matrix_rbf)

class_report_ln = classification_report(y_te, pred_svm_ln)
print("---- linear ----")
print(class_report_ln)
class_report_sg = classification_report(y_te, pred_svm_sg)
print("---- sigmoid ----")
print(class_report_sg)
class_report_rbf = classification_report(y_te, pred_svm_rbf)
print("---- rbf ----")
print(class_report_rbf)