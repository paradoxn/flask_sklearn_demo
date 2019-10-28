from flask import Flask, request,make_response,render_template, redirect, url_for,g,session
import config
from werkzeug.utils import secure_filename
from os import path
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import ast
from sklearn.linear_model import ElasticNetCV
from decorators import upload_file
from flask import Response, json


app = Flask(__name__)
app.config.from_object(config)

def onehot(df):
    le = LabelEncoder()
    le_count = 0
    y_train = df.iloc[:,-1]
    # Iterate through the columns
    for col in df[:-1]:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                # Train on the training data
                le.fit(df[col])
                # Transform both training and testing data
                df[col] = le.transform(df[col])

                # Keep track of how many columns were label encoded
                le_count += 1
    # print('%d columns were label encoded.' % le_count)
    # one-hot encoding of categorical variables
    x_train = pd.get_dummies(df.iloc[:,:-1])
    return x_train,y_train

@app.route('/')
def index():
    #session['u']
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/algor/<string:algorithm>', methods=['POST'])
def algor(algorithm):
    if algorithm == "linearRegression":
        pass

@app.route("/upload",methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files["file"]
        base_path = path.abspath(path.dirname(__file__))
        upload_path = path.join(base_path,'static/uploads/')
        file_name = upload_path + secure_filename(f.filename)
        session['file'] = file_name
        f.save(file_name)
        return file_name
    else:
        return 'false'
  #      return redirect(url_for('index'))
  #  return render_template('index.html')

@app.route('/all', methods=['POST'])
@upload_file
def all():
    request_content=request.form.to_dict()
    # print(session.get('file'))
    df = pd.read_csv(session.get('file'))
    # print(df.head())
    # print(df.columns.values[-1])
    X_train,Y_train = onehot(df)
    LL=[]
    result_dict = {}
    # result_dict['因变量'] = df.columns.values
    content = "因变量共有%s项，分别为%s，自变量为%s" % (len(df.columns.values[:-1]), df.columns.values[:-1], df.columns.values[-1])
    result_dict['content'] = content

    for i in request_content.keys():
        if i == 'algor_regression':
            reg = linear_model.LinearRegression()
            reg.fit(X_train,Y_train)
            acc_reg = round(reg.score(X_train,Y_train) * 100, 2)
            LL.append({'algor_regression':acc_reg})
            result_dict["algor_regression"] = acc_reg
            # print(reg)

        if i == 'log_regression':
            logreg = LogisticRegression()
            logreg.fit(X_train, Y_train)
            acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
            LL.append({'log_regression': acc_log})
            result_dict["log_regression"] = acc_log

        if i == 'DTree':
            decision_tree = DecisionTreeClassifier(max_depth = 4)
            decision_tree.fit(X_train, Y_train)
            acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
            LL.append({'DTree':acc_decision_tree})
            result_dict["DTree"] = acc_decision_tree

        if i == 'Random_Forest':
            random_forest = RandomForestClassifier(n_estimators=100,max_depth=3)
            random_forest.fit(X_train, Y_train)
            acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
            LL.append({'Random_Forest':acc_random_forest})
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
            xgb_model = XGBClassifier()
            xgb_model.fit(X_train,Y_train)
            acc_xgb = round(xgb_model.score(X_train,Y_train) * 100, 2)
            result_dict["XGBoostClassifier"] = acc_xgb

        if i == "LightGBM":
            lgb_model = lightgbm.LGBMClassifier()
            lgb_model.fit(X_train,Y_train)
            acc_lgb = round(lgb_model.score(X_train,Y_train) * 100,2)
            result_dict["LGBMClassifier"] = acc_lgb


    return Response(json.dumps(result_dict),content_type='application/json')
    # print(str(result_dict).replace('\"','\''))
    # return str(result_dict).replace('\'','\"')

@app.route('/algor/xgboost', methods=['POST'])
@upload_file
def algor_xgboost():
    request_content = request.form.to_dict()
    # print(session.get('file'))
    df = pd.read_csv(session.get('file'))
    X_train,Y_train = onehot(df)
    params = request_content
    if params['num_class'] == "None":
        params.pop('num_class')
    model=XGBClassifier(learning_rate=float(params['learning_rate']),
                        n_estimators=int(params['n_estimators']),
                        max_depth=int(params['max_depth']),
                        subsample=float(params['subsample']),
                        colsample_bytree=float(params['colsample_btree']),
                        reg_alpha=float(params['reg_alpha']),
                        reg_lambda=float(params['reg_lambda']),
                        gamma=float(params['gamma']),
                        min_child_weight=float(params['min_child_weight']),
                        eta=float(params['eta']),
                        objective=params['objective'],
                        num_class=int(params['num_class']))
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_train)


    context = {
        'algor' : 'xgboost',
        'roc_AUC': round(metrics.roc_auc_score(Y_train, y_pred),2),
        'ACC': round(metrics.accuracy_score(Y_train, y_pred),2),
        'Recall': round(metrics.recall_score(Y_train, y_pred),2),
        'F1_score': round(metrics.f1_score(Y_train, y_pred),2),
        'Precesion': round(metrics.precision_score(Y_train, y_pred),2),
    }
    return render_template('xgboost.html',**context)
    # return "roc_AUC: %.4f || ACC: %.4f || Recall: %.4f || F1-score: %.4f || Precesion: %.4f || confusion_matrix: %s"%(metrics.roc_auc_score(Y_train, y_pred),
    #                                                                                                                                        metrics.accuracy_score(Y_train, y_pred),
    #                                                                                                                                        metrics.recall_score(Y_train, y_pred),
    #                                                                                                                                        metrics.f1_score(Y_train, y_pred),
    #                                                                                                                                        metrics.precision_score(Y_train, y_pred),
    #                                                                                                                                        metrics.confusion_matrix(Y_train, y_pred))

@app.route('/algor/RandomForestClassifier', methods=['POST'])
@upload_file
def algor_randomclassifier():
    request_content = request.form.to_dict()
    # print(session.get('file'))
    df = pd.read_csv(session.get('file'))
    X_train,Y_train = onehot(df)
    params = request_content
    if params['max_depth']!='None':
        params['max_depth'] = int(params['max_depth'])
    else:
        params['max_depth'] = None
    if params['class_weight']!='None':
        params['class_weight'] = ast.literal_eval(params['class_weight'])
    else:
        params['class_weight'] = None
    if float(params['min_samples_split'])>=1:
        params['min_samples_split'] = int(params['min_samples_split'])
    else:
        params['min_samples_split'] = float(params['min_samples_split'])
    if float(params['min_samples_leaf'])>=1:
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
    else:
        params['min_samples_leaf'] = float(params['min_samples_leaf'])
    # print(type(params['max_depth']))
    model = RandomForestClassifier(criterion=params['criterion'],
                          n_estimators=int(params['n_estimators']),
                          max_depth=params['max_depth'],
                          min_samples_split=params['min_samples_split'],
                          min_samples_leaf=params['min_samples_leaf'],
                          class_weight=params['class_weight']
                                   )
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_train)

    context = {
        'algor': 'RandomForestClassifier',
        'roc_AUC': round(metrics.roc_auc_score(Y_train, y_pred), 2),
        'ACC': round(metrics.accuracy_score(Y_train, y_pred), 2),
        'Recall': round(metrics.recall_score(Y_train, y_pred), 2),
        'F1_score': round(metrics.f1_score(Y_train, y_pred), 2),
        'Precesion': round(metrics.precision_score(Y_train, y_pred), 2),
    }
    return render_template('randomclassifier.html', **context)

@app.route('/algor/DecisionTreeClassifier', methods=['POST'])
@upload_file
def algor_dtree():
    request_content = request.form.to_dict()
    # print(session.get('file'))
    df = pd.read_csv(session.get('file'))
    X_train,Y_train = onehot(df)
    params = request_content
    if params['max_depth'] != 'None':
        params['max_depth'] = int(params['max_depth'])
    else:
        params['max_depth'] = None
    if params['class_weight'] != 'None':
        params['class_weight'] = ast.literal_eval(params['class_weight'])
    else:
        params['class_weight'] = None
    if float(params['min_samples_split']) >= 1:
        params['min_samples_split'] = int(params['min_samples_split'])
    else:
        params['min_samples_split'] = float(params['min_samples_split'])
    if float(params['min_samples_leaf']) >= 1:
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
    else:
        params['min_samples_leaf'] = float(params['min_samples_leaf'])
    # print(type(params['max_depth']))
    model = DecisionTreeClassifier(criterion=params['criterion'],
                                   splitter=params['splitter'],
                                   max_depth=params['max_depth'],
                                   min_samples_split=params['min_samples_split'],
                                   min_samples_leaf=params['min_samples_leaf'],
                                   class_weight=params['class_weight']
                                   )
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_train)

    context = {
        'algor': '决策树',
        'roc_AUC': round(metrics.roc_auc_score(Y_train, y_pred), 2),
        'ACC': round(metrics.accuracy_score(Y_train, y_pred), 2),
        'Recall': round(metrics.recall_score(Y_train, y_pred), 2),
        'F1_score': round(metrics.f1_score(Y_train, y_pred), 2),
        'Precesion': round(metrics.precision_score(Y_train, y_pred), 2),
    }
    return render_template('dtree.html', **context)

@app.route('/algor/logisticRegression', methods=['POST'])
@upload_file
def algor_logisticRegression():
    request_content = request.form.to_dict()
    # print(session.get('file'))
    df = pd.read_csv(session.get('file'))
    X_train,Y_train = onehot(df)
    params = request_content
    if params['class_weight'] != 'None':
        if params['class_weight'] == 'balanced':
            pass
        else:
            params['class_weight'] = ast.literal_eval(params['class_weight'])
    else:
        params['class_weight'] = None
    # print(type(params['max_depth']))
    if params['penalty'] in ['elasticnet','l1']:
        solver = 'saga'
    else:
        solver = 'lbfgs'
    if params['penalty'] == 'elasticnet':
        if params['l1_ratio'] == "None":
            return '请指定l1_ratio的值'
        else:
            params['l1_ratio'] = float(params['l1_ratio'])
    else:
        params['l1_ratio'] = None

        # elif params['class_weight'] == 'l1':
    # max_iter=int(round(float(params['max_iter'])))
    model = LogisticRegression(penalty=params['penalty'],
                                   class_weight=params['class_weight'],
                                   fit_intercept=bool(params['fit_intercept']),
                                   C=float(params['C']),
                                   max_iter=int(round(float(params['max_iter']))),
                                   tol=float(params['tol']),solver=solver,l1_ratio=params['l1_ratio']
                                   )
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_train)
    if len(Y_train.unique())>2:
        context = {
            'algor': '逻辑回归',
            'roc_AUC': 'None(分类目标大于2，无法适用roc_Auc曲线）',
            'ACC': round(metrics.accuracy_score(Y_train, y_pred), 2),
            'Recall': round(metrics.recall_score(Y_train, y_pred), 2),
            'F1_score': round(metrics.f1_score(Y_train, y_pred), 2),
            'Precesion': round(metrics.precision_score(Y_train, y_pred), 2),
        }
    else:
        context = {
            'algor': '逻辑回归',
            'roc_AUC': round(metrics.roc_auc_score(Y_train, y_pred), 2),
            'ACC': round(metrics.accuracy_score(Y_train, y_pred), 2),
            'Recall': round(metrics.recall_score(Y_train, y_pred), 2),
            'F1_score': round(metrics.f1_score(Y_train, y_pred), 2),
            'Precesion': round(metrics.precision_score(Y_train, y_pred), 2),
        }
    return render_template('logisticRegression.html', **context)

@app.route('/algor/ElasticNetCV', methods=['POST'])
@upload_file
def algor_ElasticNetCV():
    request_content = request.form.to_dict()
    df = pd.read_csv(session.get('file'))
    X_train,Y_train = onehot(df)
    params = request_content
    if params['alpha'] != 'None':
        params['alpha'] = [float(params['alpha'])]
    else:
        params['alpha'] = None
    # print(type(params['max_depth']))
    # print(params['max_iter'])
    # elif params['class_weight'] == 'l1':
    # max_iter = int(round(float(params['max_iter'])))
    model = ElasticNetCV(alphas=params['alpha'],
                               l1_ratio=float(params['l1_rotio']),
                               fit_intercept=bool(params['fit_intercept']),
                               normalize=bool(params['normalize']),
                               max_iter=int(params['max_iter']),
                               tol=float(params['tol'],)
                               )
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_train)

    context = {
        'algor': '弹性网回归',
        'roc_AUC': 'None(仅用于分类器)',
        'ACC': 'None(仅用于分类器)',
        'Recall': 'None(仅用于分类器)',
        'F1_score': 'None(仅用于分类器)',
        'Precesion':  'None(仅用于分类器)',
        'R_2' : round(metrics.r2_score(Y_train,y_pred),2)
    }
    return render_template('ElasticNetCV.html', **context)

if __name__ == '__main__':
    app.run(debug=True)


