import os
import streamlit as st
# from multiapp import MultiApp
import pandas as pd
import numpy as np
import plotly.express as px
from persist import persist, load_widget_state
from catboost_model_sample import preprocessing, train_model, result, zerone
from catboost import CatBoostClassifier # 05/28
from io import BytesIO

# import pandas_profiling

# from streamlit_pandas_profiling import st_profile_report


def main():
    if "page" not in st.session_state:
        # Initialize session state.
        st.session_state.update({
            # Default page.
            "page": "home",

            # Radio, selectbox and multiselect options.
            "gender_options": ["여자", "남자"],

            # 기본값
            "text": "",
            "slider": 0,
            "checkbox": False,
            "gender" : "F",
            # "age" :
            "selectbox": "Hello",
            "multiselect": ["Hello", "Everyone"],
        }
        )

    page = st.sidebar.radio("페이지를 선택해주세요", tuple(PAGES.keys()), format_func=str.capitalize)

    PAGES[page]()

def total_graph():
    st.subheader('전체 표 보기')
    DATA_PATH = ('./data/')

    # 불러온 차트 보여주기
    train = pd.read_csv(DATA_PATH + 'final_df.csv')
    train.drop(['credit_r', 'DAYS_EMPLOYED'], axis=1, inplace=True)
    test = pd.read_csv(DATA_PATH + 'final_test_df.csv')
    test.drop(['DAYS_EMPLOYED'], axis=1, inplace=True)


    #income_total
    for col1, col2 in [['income_total', 'DAYS_BIRTH']]:
        df = train.copy()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col2, y=col1, color="credit", title=str('Scatter chart: '+col1+' & '+col2))
                        #  size='', hover_data=[''])
        # fig.show()
        st.plotly_chart(fig)
    
    #income_type
    for col in ['income_type']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.bar(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'income_type','Unnamed: 0':'credit count'})
        st.plotly_chart(fig)

    #days_birth
    for col in ['DAYS_BIRTH']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.bar(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'age','Unnamed: 0':'credit count'})
        st.plotly_chart(fig)

    #DAYS_EMPLOYED
    #train['DAYS_EMPLOYED_Y'] = train['DAYS_EMPLOYED_Y'].map(lambda x: 0 if x < 0 else x)
    for col in ['DAYS_EMPLOYED_r']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'DAYS_EMPLOYED_r','Unnamed: 0':'credit count'})
        st.plotly_chart(fig)

    #occyp_type
    for col in ['occyp_type']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.pie(df, values='Unnamed: 0', names=col, title=str('Pie chart: '+col), labels={col:'occyp_type','Unnamed: 0':'credit count'})
        # fig.show()
        st.plotly_chart(fig)

    #begin_month
    for col in ['begin_month']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.bar(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'begin_month','Unnamed: 0':'credit count'})
        # fig.show()
        st.plotly_chart(fig)
    
    # 데이터 전처리
    train.drop('Unnamed: 0', axis=1, inplace=True)
    # pre_train, pre_test = preprocessing(train, test)
    preprocessing(train, test)

    #ability
    for col1, col2 in [['ability', 'income_type']]:
        df = train.copy()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col2, y=col1, color="credit", title=str('Scatter chart: '+col1+' & '+col2))
                        #  size='', hover_data=[''])
        # fig.show()
        st.plotly_chart(fig)
    
    #income_mean
    for col1, col2 in [['income_mean', 'family_type']]:
        df = train.copy()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col2, y=col1, color="credit", title=str('Scatter chart: '+col1+' & '+col2))
                        #  size='', hover_data=[''])
        # fig.show()
        st.plotly_chart(fig)
    
    # sunburst chart
    # col1, col2, col3 in 범주형 변수, 범주형 변수, 범주형 변수
    for col1, col2, col3 in [['family_type','income_type', 'credit']]:
        df = train.copy()
        fig = px.sunburst(df, path=[col1,col2,col3], title = str('Sun chart: '+col1+' & '+col2+' & '+col3))
        # fig.show()
        st.plotly_chart(fig)

    train = pd.read_csv(DATA_PATH + 'final_df.csv')
    # train.drop('credit', axis=1, inplace=True)
    # train.drop('Unnamed: 0', axis=1, inplace=True)
    X_train = pd.read_csv(DATA_PATH + 'input_list.csv')
    X_train=zerone(X_train)
    # a = train.profile_report()
    # b = X_train.profile_report()
    # st_profile_report(a)
    # st_profile_report(b)
    # a = pd.read_csv(DATA_PATH + 'service.csv')
    print(X_train.info())
    # print("-"*10)
    # print(a[X_train.columns].info())
    # 1. 사용자 input 저장 ( DB vs CSV)
    # 2. model -> data를 뭘 사용? (train데이터로 학습된 모델 사용)
    # -> 


    # pre_train, pre_test = preprocessing(train, test)
    # pre_train, pre_X_train = preprocessing(train, X_train)
    preprocessing(train, X_train)
    # preprocessing(train, a)
    # 데이터 전처리 출력
    # st.write(pre_train, pre_test)
    # st.write(pre_train, pre_X_train)


    # 모델 학습
    # model_cat, X_train = train_model(pre_train, pre_test)
    # model_cat.save_model("./data/model.bin") # 5/28

    # 모델 학습 출력
    # st.write(model_cat, X_train)
    from_file = CatBoostClassifier()  # 5/28
    from_file.load_model("./data/model.bin") # 5/28

    # 학습 결과
    # y_predict = result(model_cat, X_train)
    
    # b = from_file.predict(train)
    
    print("AAAA")

    y_predict = from_file.predict(X_train) # 5/28
    # 학습 결과 출력
    st.write(y_predict)
    your_score = y_predict[0][0]
    st.write(f'당신의 신용도는 {your_score}등급입니다.')

    st.write(
        f"""
        Settings values
        ---------------
        - **성별**: {st.session_state.gender}
        """
    )



    
def my_settings():
    st.header("정보를 입력해주세요")

    gender=st.radio('성별', st.session_state["gender_options"], key=persist("gender"))
    DAYS_BIRTH=st.text_input('나이')
    edu_type=st.selectbox('학력',('Higher education' ,'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree'))
    income_total=st.text_input('소득','0')
    child_num=st.text_input('아이 수','0')
    DAYS_EMPLOYED_r=st.text_input('고용연수','0')
    work_phone=st.radio('직장 전화',('있음','없음'))
    phone=st.radio('집 전화',('있음','없음'))
    email=st.radio('이메일',('있음','없음'))
    car=st.radio('자동차',('있음','없음'))
    reality=st.radio('부동산',('있음','없음'))
    income_type=st.selectbox('소득 형태',('Commercial associate', 'Pensioner', 'State servant', 'Student', 'Working'))
    family_type=st.selectbox('가족 형태',('Civil marriage', 'Married', 'Separated', 'Single / not married', 'Widow'))
    house_type=st.selectbox('집 형태',('Co-op apartment',
        'House / apartment',
        'Municipal apartment',
        'Office apartment',
        'Rented apartment',
        'With parents'))
    occyp_type=st.selectbox('직업',('Accountants',
        'Cleaning staff',
        'Cooking staff',
        'Core staff',
        'Drivers',
        'HR staff',
        'High skill tech staff',
        'IT staff',
        'Laborers',
        'Low-skill Laborers',
        'Managers',
        'Medicine staff',
        'Private service staff',
        'Realty agents',
        'Sales staff',
        'Secretaries',
        'Security staff',
        'Waiters/barmen staff',
        'job_less'))


    family_size=st.text_input('가족 규모','0')
    begin_month=25.0
    # 7, 24, 26
    is_submit = st.button("제출", )
    if is_submit:

        # st.text_input('신용카드 발급 개월수', '0')
        input_dict = dict(
            gender=gender,
            car=car,
            reality=reality,
            child_num=child_num,
            income_total=income_total,
            income_type=income_type,
            edu_type=edu_type,
            family_type=family_type,
            house_type=house_type,
            DAYS_BIRTH=DAYS_BIRTH,
            DAYS_EMPLOYED_r=DAYS_EMPLOYED_r,
            work_phone=work_phone,
            phone=phone,
            email=email,
            occyp_type=occyp_type,
            family_size=family_size,
            begin_month=begin_month
        )
        filename = './data/input_list.csv'
        if not os.path.exists(filename):
            pd.DataFrame([], columns=input_dict.keys()).to_csv(filename, mode='w' ,header=True, index=False)
        a = pd.DataFrame([input_dict])
        a.to_csv(filename, mode='a', header=False, index=False)

        # 다른페이지로 이동
        # 결과 출력 


def my_graph():
    st.header("내 표 보기")

    st.write()


PAGES = {
    "내 정보 입력": my_settings,
    "전체 표 보기": total_graph,
    "내 표 보기" : my_graph
}


if __name__ == "__main__":
    load_widget_state()
    main()