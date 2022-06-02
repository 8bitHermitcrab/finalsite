import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from category_encoders.ordinal import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier, Pool


def zerone(df):
    for li in ['car','reality']:
        if df[li].values=='있음':
            df[li]='Y'
        else:
            df[li]='N'
        # df[df[li]=='있음']='Y'
        # df[df[li]=='없음']='N'

    for li in ['email','phone','work_phone']:
        if df[li].values=='있음':
            df[li]=1
        else:
            df[li]=0
        # df[df[li]=='있음']=1
        # df[df[li]=='없음']=0
    for li in ['gender']:
    # df[df['gender'] == '여자'] = 'F'
    # df[df['gender'] == '남자'] = 'M'
        if df[li].values=='여자':
            df['gender']='F'
        else:
            df['gender']='M'
    return df

def preprocessing(train, test):
    # 파생변수 생성
    for df in [train, test]:
        # before_EMPLOYED: 고용되기 전까지의 일수
        df['before_EMPLOYED'] = df['DAYS_BIRTH']*365 - df['DAYS_EMPLOYED_r']*365
        df['income_total_befofeEMP_ratio'] = df['income_total'] / df['before_EMPLOYED']
        df['before_EMPLOYED_m'] = np.floor(df['before_EMPLOYED'] / 30) - ((np.floor(df['before_EMPLOYED'] / 30) / 12).astype(int) * 12)
        df['before_EMPLOYED_w'] = np.floor(df['before_EMPLOYED'] / 7) - ((np.floor(df['before_EMPLOYED'] / 7) / 4).astype(int) * 4)

        #DAYS_BIRTH 파생변수- Age(나이), 태어난 월, 태어난 주(출생연도의 n주차)
        # df['Age'] = df['DAYS_BIRTH'] // 365
        df['DAYS_BIRTH_m'] = np.floor(df['DAYS_BIRTH']*365 / 30) - ((np.floor(df['DAYS_BIRTH']*365 / 30) / 12).astype(int) * 12)
        df['DAYS_BIRTH_w'] = np.floor(df['DAYS_BIRTH']*365 / 7) - ((np.floor(df['DAYS_BIRTH']*365 / 7) / 4).astype(int) * 4)


        #DAYS_EMPLOYED_m 파생변수- EMPLOYED(근속연수), DAYS_EMPLOYED_m(고용된 달) ,DAYS_EMPLOYED_w(고용된 주(고용연도의 n주차))  
        # df['EMPLOYED'] = df['DAYS_EMPLOYED'] // 365
        df['DAYS_EMPLOYED_m'] = np.floor(df['DAYS_EMPLOYED_r']*365 / 30) - ((np.floor(df['DAYS_EMPLOYED_r']*365 / 30) / 12).astype(int) * 12)
        df['DAYS_EMPLOYED_w'] = np.floor(df['DAYS_EMPLOYED_r']*365 / 7) - ((np.floor(df['DAYS_EMPLOYED_r']*365 / 7) / 4).astype(int) * 4)

        #ability: 소득/(살아온 일수+ 근무일수)
        df['ability'] = df['income_total'] / (df['DAYS_BIRTH']*365 + df['DAYS_EMPLOYED_r']*365)

        #income_mean: 소득/ 가족 수
        df['income_mean'] = df['income_total'] / df['family_size']

        #ID 생성: 각 컬럼의 값들을 더해서 고유한 사람을 파악(*한 사람이 여러 개 카드를 만들 가능성을 고려해 begin_month는 제외함)
        df['ID'] = \
        df['child_num'].astype(str) + '_' + df['income_total'].astype(str) + '_' +\
        df['DAYS_BIRTH'].astype(str) + '_' + df['DAYS_EMPLOYED_r'].astype(str) + '_' +\
        df['work_phone'].astype(str) + '_' + df['phone'].astype(str) + '_' +\
        df['email'].astype(str) + '_' + df['family_size'].astype(str) + '_' +\
        df['gender'].astype(str) + '_' + df['car'].astype(str) + '_' +\
        df['reality'].astype(str) + '_' + df['income_type'].astype(str) + '_' +\
        df['edu_type'].astype(str) + '_' + df['family_type'].astype(str) + '_' +\
        df['house_type'].astype(str) + '_' + df['occyp_type'].astype(str)

    cols = ['child_num', 'DAYS_BIRTH', 'DAYS_EMPLOYED_r']
    train.drop(cols, axis=1, inplace=True)
    test.drop(cols, axis=1, inplace=True)

    numerical_feats = df.dtypes[df.dtypes != "object"].index.tolist()
    
    try:
        numerical_feats.remove('credit')
    except:
        pass
    categorical_feats = df.dtypes[df.dtypes == "object"].index.tolist()

    for df in [train, test]:
        df['income_total'] = np.log1p(1+df['income_total'])

    encoder = OrdinalEncoder(categorical_feats)

    train[categorical_feats] = encoder.fit_transform(train[categorical_feats], train['credit'])
    test[categorical_feats] = encoder.transform(test[categorical_feats])

    train['ID'] = train['ID'].astype('int64')
    test['ID'] = test['ID'].astype('int64')

    numerical_feats.remove('income_total')
    scaler = StandardScaler()
    train[numerical_feats] = scaler.fit_transform(train[numerical_feats])
    test[numerical_feats] = scaler.transform(test[numerical_feats])
    print('categorical_feats',categorical_feats)
    print(test[categorical_feats])
    print(train[categorical_feats])
    print('numerical_feats',numerical_feats)
    print(train[numerical_feats])
    print(test[numerical_feats])

    print(train.isnull().sum())
    print(test.isnull().sum())
    return train, test



def train_model(train, test):
    n_est = 2000
    seed = 42
    n_class = 3
    n_fold = 18
    # n_fold = 3

    target = 'credit'
    X = train.drop(target, axis=1)
    y = train[target]
    X_test = test

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    skfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    folds=[]
    for train_idx, valid_idx in skfold.split(X, y):
        folds.append((train_idx, valid_idx))

    cat_pred = np.zeros((X.shape[0], n_class))
    cat_pred_test = np.zeros((X_test.shape[0], n_class))
    cat_cols = ['income_type', 'edu_type', 'family_type', 'house_type', 'occyp_type', 'ID']

    for fold in range(n_fold):
        # print(f'\n----------------- Fold {fold} -----------------\n')
        train_idx, valid_idx = folds[fold]
        X_train, X_valid, y_train, y_valid = X.iloc[train_idx], X.iloc[valid_idx], y[train_idx], y[valid_idx]
        train_data = Pool(data=X_train, label=y_train, cat_features=cat_cols)
        valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_cols)

        model_cat = CatBoostClassifier()
        model_cat.fit(train_data, eval_set=valid_data, use_best_model=True, early_stopping_rounds=100, verbose=100)

        cat_pred[valid_idx] = model_cat.predict_proba(X_valid)
        cat_pred_test += model_cat.predict_proba(X_test) / n_fold
        # print(f'CV Log Loss Score: {log_loss(y_valid, cat_pred[valid_idx]):.6f}')
    # model_cat.save_model('/Users/jiho/Desktop/mulcam_final/data/model.bin') #5/28수정
    return model_cat, X_train



def result(model_cat, X_train):
    y_predict= model_cat.predict(X_train)
    # print(f'\tacc: {accuracy_score(y_train, y_predict):.6f}')     
    # print(f'\tLog Loss: {log_loss(y, cat_pred):.6f}')
    # print('='*60)

    return y_predict