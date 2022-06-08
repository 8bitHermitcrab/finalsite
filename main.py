import os
import streamlit as st
# from multiapp import MultiApp
import pandas as pd
import numpy as np
import plotly.express as px
from persist import persist, load_widget_state
from catboost_model import preprocessing, train_model
from catboost import CatBoostClassifier # 05/28
from io import BytesIO
from datetime import datetime
import streamlit_authenticator as stauth


def main():
    if "page" not in st.session_state:
        # Initialize session state.
        st.session_state.update({
            # Default page.
            "page": "home",

            # Radio, selectbox and multiselect options.
            "gender_options": ["여성", "남성"],

            # # 기본값
            # "text": "",
            # "slider": 0,
            # "checkbox": False,
            # "gender" : "여자",
            # # "age" :
            # "selectbox": "Hello",
            # "multiselect": ["Hello", "Everyone"],
        }
        )

    page = st.sidebar.radio("페이지를 선택해주세요", tuple(PAGES.keys()), format_func=str.capitalize)

    PAGES[page]()

# 로그인 기능
def login():
    names = ['관리자']
    usernames = ['mulcam']
    passwords = ['123']
    hashed_passwords = stauth.Hasher(passwords).generate()
    authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
    'some_cookie_name','some_signature_key',cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login('Login','main')
    if authentication_status:
        st.write('환영합니다. *%s* 님' % (name))
        authenticator.logout('Logout', 'main')
        return total_graph()
    elif authentication_status == False:
        st.error('아이디나 비밀번호가 맞지 않습니다.')
    elif authentication_status == None:
        st.warning('아이디와 비밀번호를 입력해주세요.')


def total_graph():
    st.header("마이데이터를 활용한 신용도 예측 시스템")
    st.subheader('기업용 전체 표 보기')
    
    DATA_PATH = ('./data/')

    # 불러온 차트 보여주기
    train = pd.read_csv(DATA_PATH + 'trans_final_df.csv')
    train.drop(['신용도_r', '고용연수'], axis=1, inplace=True)

    #income_total
    for col1, col2 in [['income_total', 'DAYS_BIRTH']]:
        df = train.copy()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col2, y=col1, color="credit", title=str('Scatter chart: '+col1+' & '+col2))
                        #  size='', hover_data=[''])
        fig.add_annotation(x=38,
            y=202500.0,
            # text=" ",
            # bgcolor='white',
            # font_size=15,
            showarrow=True,
            arrowsize=2, 
            arrowhead=3, 
            # ax=20, 
            # ay=0,
            # arrowhead=3,
            arrowwidth=1,
            arrowcolor='red',
            # xshift=10
            )
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
        df = df[df['DAYS_EMPLOYED_r'] > 0]
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'DAYS_EMPLOYED_r','Unnamed: 0':'credit count'})
        # fig.update_layout(legend_traceorder="normal", legend_title='credit')
        st.plotly_chart(fig)

    #occyp_type
    for col in ['occyp_type']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.pie(df, values='Unnamed: 0', names=col, title=str('Pie chart: '+col), labels={col:'occyp_type','Unnamed: 0':'credit count'})
        st.plotly_chart(fig)

    #begin_month
    for col in ['begin_month']:
        df = train.copy()
        df = df.groupby(by=[col,'credit']).count().reset_index()
        df['credit'] = df['credit'].astype(str)
        fig = px.bar(df, x=col, y="Unnamed: 0", title=str('Bar chart: '+col+' & credit count'), color="credit", labels={col:'begin_month','Unnamed: 0':'credit count'})
        # fig.show()
        st.plotly_chart(fig)
    
    for col1, col2 in [['credit','occyp_type']]:
        df = train.copy()
        fig = px.sunburst(df, path=[col1,col2], title = str('Sun chart: '+col1+' & '+col2))
        fig.update_layout(margin=dict(t=50, l=50, r=50, b=50)).update_traces(texttemplate="%{label}<br>%{percentEntry:.2%}")
        st.plotly_chart(fig)

    # sunburst chart
    # col1, col2, col3 in 범주형 변수, 범주형 변수, 범주형 변수
    for col1, col2, col3 in [['family_type','income_type', 'credit']]:
        df = train.copy()
        fig = px.sunburst(df, path=[col1,col2,col3], title = str('Sun chart: '+col1+' & '+col2+' & '+col3))
        fig.update_layout(margin=dict(t=50, l=50, r=50, b=50)).update_traces(texttemplate="%{label}<br>%{percentEntry:.2%}")
        st.plotly_chart(fig)

    #ability
    for col1, col2 in [['ability', 'income_type']]:
        df = train.copy()
        #ability: 소득/(살아온 일수+ 근무일수)
        df['ability'] = df['income_total'] / (df['DAYS_BIRTH']*365 + df['DAYS_EMPLOYED_r']*365)
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col2, y=col1, color="credit", title=str('Scatter chart: '+col1+' & '+col2))
                        #  size='', hover_data=[''])
        # 범례
        # fig.update_layout(legend_traceorder="normal")
        st.plotly_chart(fig)
    
    #income_mean
    for col1, col2 in [['income_mean', 'family_type']]:
        # df = train.copy()
        #income_mean: 소득/ 가족 수
        df['income_mean'] = df['income_total'] / df['family_size']
        df['credit'] = df['credit'].astype(str)
        fig = px.scatter(df, x=col2, y=col1, color="credit", title=str('Scatter chart: '+col1+' & '+col2))
                        #  size='', hover_data=[''])
        st.plotly_chart(fig)


    # train = pd.read_csv(DATA_PATH + 'final_df.csv')
    train = pd.read_csv(DATA_PATH + 'trans_final_df.csv')
    X_train = pd.read_csv(DATA_PATH + 'input_list.csv')
    # X_train=zerone(X_train)
    
    # 1. 사용자 input 저장 ( DB vs CSV)
    # 2. model -> data를 뭘 사용? (train데이터로 학습된 모델 사용)
    # -> 


    # 데이터 전처리
    preprocessing(train, X_train)

    # 모델 학습 출력
    from_file = CatBoostClassifier()  # 5/28
    from_file.load_model("./data/model.bin") # 5/28


    y_predict = from_file.predict(X_train) # 5/28
    # 학습 결과 출력
    st.write(y_predict)
    your_score = y_predict[0][0]
    st.write(f'당신의 신용도는 {your_score}등급입니다.')



    
def my_settings():
    st.header("마이데이터를 활용한 신용도 예측 시스템")
    st.subheader("소비자용 내 정보 입력")
    st.text("정보를 입력해주세요.")

    gender=st.radio('성별', st.session_state["gender_options"], key=persist("gender"))
    DAYS_BIRTH=st.text_input('나이', placeholder='숫자만 입력. 예) 20세 = 20', max_chars=3)
    try:
        int(DAYS_BIRTH) 
    except:
        st.error("형식에 맞게 입력해 주세요.")
    edu_type=st.selectbox('학력',('중학교 졸업', '고등학교 졸업', '고등학교 중퇴', '초등학교 졸업', '대학교 졸업'))
    income_total=st.text_input('소득 (만 단위)', placeholder='숫자만 입력(만 단위). 예) 4000만원 = 4000', max_chars=6)
    try:
        int(income_total)
    except:
        st.error("형식에 맞게 입력해 주세요.")
    child_num=st.text_input('자녀 수', placeholder='숫자만 입력. 예) 2명 = 2', max_chars=1)
    try:
        int(child_num)
    except:
        st.error("형식에 맞게 입력해 주세요.")
    DAYS_EMPLOYED_r=st.text_input('고용연수', placeholder='숫자만 입력(연 단위). 예) 5년 = 5', max_chars=2)
    try:
        int(DAYS_EMPLOYED_r)
    except:
        st.error("형식에 맞게 입력해 주세요.")
    work_phone=st.radio('직장 전화',('있음','없음'))
    phone=st.radio('집 전화',('있음','없음'))
    email=st.radio('이메일',('있음','없음'))
    car=st.radio('자동차',('있음','없음'))
    reality=st.radio('부동산',('있음','없음'))
    income_type=st.selectbox('소득 형태',('근로자', '사업가', '연금수령자', '공무원', '학생'))
    family_type=st.selectbox('가족 형태',('미혼', '기혼', '사실혼', '이혼', '미망인'))
    house_type=st.selectbox('주거 형태',('주택조합 아파트', '주택 / 아파트', '시립 아파트', '사옥', '임대 아파트', '부모님과 거주'))
    occyp_type=st.selectbox('직업',('회계사', '청소부원', '요리사', '사무직', '운전기사', '인사직', '전문직',
        'IT직', '일용직 노동자', '제조업 노동자', '관리자', '의료인', '민간 서비스직', '부동산업자', '영업직', '비서', '경비원', '웨이터 / 바텐더', '무직'))

    family_size=st.text_input('가족 규모', placeholder='본인 포함, 숫자만 입력. 예) 3명 = 3', max_chars=2)
    try:
        int(family_size)
    except:
        st.error("형식에 맞게 입력해 주세요.")
    begin_month=st.text_input('신용카드 유효기간 (MM/YY) ', placeholder='MM/YY', max_chars=5, help='본인이 소유하고 있는 신용 카드 중 유효기간이 제일 긴 카드 입력하세요. 신용카드가 없으면 00/00을 입력하세요.')

    # 제출 버튼
    is_submit = st.button("제출", )
    if is_submit:

        if begin_month== '00/00':
            begin_month = 0
        else:
            begin_month=int(begin_month[-2:])*12+int(begin_month[:2])-(datetime.today().year%100*12+datetime.today().month)

        input_dict = dict(
            성별=gender,
            자동차=car,
            부동산=reality,
            자녀_수=child_num,
            소득=income_total,
            소득_형태=income_type,
            학력=edu_type,
            가족_형태=family_type,
            주거_형태=house_type,
            나이=DAYS_BIRTH,
            고용연수_r=DAYS_EMPLOYED_r,
            직장_전화=work_phone,
            집_전화=phone,
            이메일=email,
            직업=occyp_type,
            가족_규모=family_size,
            신용카드_유효기간=begin_month
        )
        filename = './data/input_list.csv'
        if not os.path.exists(filename):
            pd.DataFrame([], columns=input_dict.keys()).to_csv(filename, mode='w' ,header=True, index=False)
        a = pd.DataFrame([input_dict])
        a.to_csv(filename, mode='a', header=False, index=False)

        # 다른페이지로 이동
        # 결과 출력
        DATA_PATH = ('./data/')
        train = pd.read_csv(DATA_PATH + 'trans_final_df.csv')
        train.drop(['Unnamed: 0', '신용도_r', '고용연수'], axis=1, inplace=True)

        X_train = pd.read_csv(DATA_PATH + 'input_list.csv')

        preprocessing(train, X_train)
        from_file = CatBoostClassifier()  # 5/28
        from_file.load_model("./data/model.bin") # 5/28
        y_predict = from_file.predict(X_train) # 5/28

        # # 모델 학습
        # model_cat, X_train = train_model(pre_train, pre_test)
        # model_cat.save_model("./data/model.bin") # 5/28

        your_score = y_predict[0][0]
        st.info(f'당신의 신용도는 {your_score}등급입니다.')

    # 초기화 버튼
    is_reset = st.button("초기화", )
    if is_reset:
        os.unlink('./data/input_list.csv')
        st.success('데이터가 초기화되었습니다.')



PAGES = {
    "소비자용 내 정보 입력": my_settings,
    "기업용 전체 표 보기": total_graph,
}


if __name__ == "__main__":
    load_widget_state()
    main()