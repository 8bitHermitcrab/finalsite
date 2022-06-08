import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from category_encoders.ordinal import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier, Pool


# def zerone(df):
#     for li in ['car','reality']:
#         if df[li].values=='있음':
#             df[li]='Y'
#         else:
#             df[li]='N'
#         # df[df[li]=='있음']='Y'
#         # df[df[li]=='없음']='N'

#     for li in ['email','phone','work_phone']:
#         if df[li].values=='있음':
#             df[li]=1
#         else:
#             df[li]=0
#         # df[df[li]=='있음']=1
#         # df[df[li]=='없음']=0
#     for li in ['gender']:
#     # df[df['gender'] == '여자'] = 'F'
#     # df[df['gender'] == '남자'] = 'M'
#         if df[li].values=='여자':
#             df['gender']='F'
#         else:
#             df['gender']='M'
#     return df

def preprocessing(train, test):
    # 파생변수 생성
    for df in [train, test]:
        # before_EMPLOYED: 고용되기 전까지의 일수
        df['고용되기_전까지의_일수'] = df['나이']*365 - df['고용연수_r']*365
        df['소득_고용되기전일수_비율'] = df['소득'] / df['고용되기_전까지의_일수']
        df['고용되기전일수_달'] = np.floor(df['고용되기_전까지의_일수'] / 30) - ((np.floor(df['고용되기_전까지의_일수'] / 30) / 12).astype(int) * 12)
        df['고용되기전일수_주'] = np.floor(df['고용되기_전까지의_일수'] / 7) - ((np.floor(df['고용되기_전까지의_일수'] / 7) / 4).astype(int) * 4)

        #DAYS_BIRTH 파생변수- Age(나이), 태어난 월, 태어난 주(출생연도의 n주차)
        # df['Age'] = df['DAYS_BIRTH'] // 365
        df['나이_월'] = np.floor(df['나이']*365 / 30) - ((np.floor(df['나이']*365 / 30) / 12).astype(int) * 12)
        df['나이_주'] = np.floor(df['나이']*365 / 7) - ((np.floor(df['나이']*365 / 7) / 4).astype(int) * 4)


        #DAYS_EMPLOYED_m 파생변수- EMPLOYED(근속연수), DAYS_EMPLOYED_m(고용된 달) ,DAYS_EMPLOYED_w(고용된 주(고용연도의 n주차))  
        # df['EMPLOYED'] = df['DAYS_EMPLOYED'] // 365
        df['고용연수_달'] = np.floor(df['고용연수_r']*365 / 30) - ((np.floor(df['고용연수_r']*365 / 30) / 12).astype(int) * 12)
        df['고용연수_주'] = np.floor(df['고용연수_r']*365 / 7) - ((np.floor(df['고용연수_r']*365 / 7) / 4).astype(int) * 4)

        #ability: 소득/(살아온 일수+ 근무일수)
        df['능력'] = df['소득'] / (df['나이']*365 + df['고용연수_r']*365)

        #income_mean: 소득/ 가족 수
        df['평균_소득'] = df['소득'] / df['가족_규모']

        #ID 생성: 각 컬럼의 값들을 더해서 고유한 사람을 파악(*한 사람이 여러 개 카드를 만들 가능성을 고려해 begin_month는 제외함)
        df['ID'] = \
        df['자녀_수'].astype(str) + '_' + df['소득'].astype(str) + '_' +\
        df['나이'].astype(str) + '_' + df['고용연수_r'].astype(str) + '_' +\
        df['직장_전화'].astype(str) + '_' + df['집_전화'].astype(str) + '_' +\
        df['이메일'].astype(str) + '_' + df['가족_규모'].astype(str) + '_' +\
        df['성별'].astype(str) + '_' + df['자동차'].astype(str) + '_' +\
        df['부동산'].astype(str) + '_' + df['소득_형태'].astype(str) + '_' +\
        df['학력'].astype(str) + '_' + df['가족_형태'].astype(str) + '_' +\
        df['주거_형태'].astype(str) + '_' + df['직업'].astype(str)

    # cols = ['자녀_수', '나이', '고용연수_r']
    cols = ['자녀_수', '나이', '고용연수_r', '연령대', '소득_4분위', '소득_5분위', '소득_10분위']
    train.drop(cols, axis=1, inplace=True)
    cols_t = ['자녀_수', '나이', '고용연수_r']
    test.drop(cols_t, axis=1, inplace=True)

    numerical_feats = df.dtypes[df.dtypes != "object"].index.tolist()
    
    try:
        numerical_feats.remove('신용도')
    except:
        pass
    categorical_feats = df.dtypes[df.dtypes == "object"].index.tolist()

    for df in [train, test]:
        df['소득'] = np.log1p(1+df['소득'])

    encoder = OrdinalEncoder(categorical_feats)

    train[categorical_feats] = encoder.fit_transform(train[categorical_feats], train['신용도'])
    test[categorical_feats] = encoder.transform(test[categorical_feats])

    train['ID'] = train['ID'].astype('int64')
    test['ID'] = test['ID'].astype('int64')

    numerical_feats.remove('소득')
    scaler = StandardScaler()
    train[numerical_feats] = scaler.fit_transform(train[numerical_feats])
    test[numerical_feats] = scaler.transform(test[numerical_feats])
    # print('categorical_feats',categorical_feats)
    # print(test[categorical_feats])
    # print(train[categorical_feats])
    # print('numerical_feats',numerical_feats)
    # print(train[numerical_feats])
    # print(test[numerical_feats])

    # print(train.isnull().sum())
    # print(test.isnull().sum())
    return train, test



def train_model(train, test):
    n_est = 2000
    seed = 42
    n_class = 3
    n_fold = 18
    # n_fold = 3

    target = '신용도'
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
    cat_cols = ['소득_형태', '학력', '가족_형태', '주거_형태', '직업', 'ID']

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
    model_cat.save_model('./data/model.bin') #5/28수정
    return model_cat, X_train



def result(model_cat, X_train):
    y_predict= model_cat.predict(X_train)
    # print(f'\tacc: {accuracy_score(y_train, y_predict):.6f}')     
    # print(f'\tLog Loss: {log_loss(y, cat_pred):.6f}')
    # print('='*60)

    return y_predict