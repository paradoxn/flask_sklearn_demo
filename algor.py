from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from flask import Flask, request,make_response,render_template, redirect, url_for,g,session
import pandas as pd



def algor_xgboost():
    model=XGBClassifier


def all():
    request_content = request.form.to_dict()
    print(session.get('file'))
    df = pd.read_csv(session.get('file'))
    X_train = df.iloc[:, :-1]
    Y_train = df.iloc[:, -1]
    LL = []
    result_dict = {}
    for i in request_content.keys():
        if i == 'algor_regression':
            reg = linear_model.LinearRegression()
            reg.fit(X_train, Y_train)
            acc_reg = round(reg.score(X_train, Y_train) * 100, 2)
            LL.append({'algor_regression': acc_reg})
            result_dict["algor_regression"] = acc_reg
            print(reg)

        if i == 'log_regression':
            logreg = LogisticRegression()
            logreg.fit(X_train, Y_train)
            acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
            LL.append({'log_regression': acc_log})
            result_dict["log_regression"] = acc_log

        if i == 'DTree':
            decision_tree = DecisionTreeClassifier(max_depth=4)
            decision_tree.fit(X_train, Y_train)
            acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
            LL.append({'DTree': acc_decision_tree})
            result_dict["DTree"] = acc_decision_tree

        if i == 'Random_Forest':
            random_forest = RandomForestClassifier(n_estimators=100, max_depth=3)
            random_forest.fit(X_train, Y_train)
            acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
            LL.append({'Random_Forest': acc_random_forest})
            result_dict["Random_Forest"] = acc_random_forest

        if i == 'SVC':
            svc = SVC()
            svc.fit(X_train, Y_train)
            acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
            result_dict["SVC"] = acc_svc

        if i == 'LinearSVC':
            linear_svc = LinearSVC()
            linear_svc.fit(X_train, Y_train)
            acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
            result_dict["LinearSVC"] = acc_linear_svc

        if i == 'KNeighborsClassifier':
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, Y_train)
            acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
            result_dict["KNeighborsClassifier"] = acc_knn

        if i == "XGBoost":
            xgb_model = xgboost.XGBClassifier()
            xgb_model.fit(X_train, Y_train)
            acc_xgb = round(xgb_model.score(X_train, Y_train) * 100, 2)
            result_dict["XGBoostClassifier"] = acc_xgb

        if i == "LightGBM":
            lgb_model = lightgbm.LGBMClassifier()
            lgb_model.fit(X_train, Y_train)
            acc_lgb = round(lgb_model.score(X_train, Y_train) * 100, 2)
            result_dict["LGBMClassifier"] = acc_lgb

    print(str(result_dict).replace('\"', '\''))
    return str(result_dict).replace('\'', '\"')






# def get_algor_score(request_content,df):
#     result_dict = {}
#     LL = []
#     X_train = df.iloc[:,:-1]
#     Y_train = df.iloc[:,-1]
#     splits = requests_content.get('splits')
#     if splits != 0 :
#         X_test,X_train,Y_test,Y_train =
#         train_test_split(X_train,Y_train,test_size=splits)
#
#
#
#     for i in request_content.keys():
#         if i == 'algor_regression':
#             reg = linear_model.LinearRegression()
#             reg.fit(X_train,Y_train)
#             acc_reg_train = round(reg.score(X_train,Y_train) * 100, 2)
#             acc_reg_test = round(reg.score(X_test,Y_test) * 100,2)
#             print(acc_reg_train,acc_reg_test)
#
#             LL.append({'algor_regression':acc_reg})
#             result_dict["algor_regression"] = acc_reg
#
#             metrics.accuracy_score()
#
#         if i == 'log_regression':
#             logreg = LogisticRegression()
#             logreg.fit(X_train, Y_train)
#             acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#             LL.append({'log_regression': acc_log})
#             result_dict["log_regression"] = acc_log
#
#         if i == 'DTree':
#             decision_tree = DecisionTreeClassifier()
#             decision_tree.fit(X_train, Y_train)
#             acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
#             LL.append({'DTree':acc_decision_tree})
#             result_dict["DTree"] = acc_decision_tree
#
#         if i == 'Random_Forest':
#             random_forest = RandomForestClassifier(n_estimators=100)
#             random_forest.fit(X_train, Y_train)
#             acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#             LL.append({'Random_Forest':acc_random_forest})
#             result_dict["Random_Forest"] = acc_random_forest
#
#         if i == 'SVC':
#             svc = SVC()
#             svc.fit(X_train, Y_train)
#             acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
#             result_dict["SVC"] = acc_svc
#
#         if i == 'LinearSVC':
#             linear_svc = LinearSVC()
#             linear_svc.fit(X_train, Y_train)
#             acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
#             result_dict["LinearSVC"] = acc_linear_svc
