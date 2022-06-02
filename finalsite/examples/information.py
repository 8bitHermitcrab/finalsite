import streamlit as st
import pandas as pd
import numpy as np

st.title('정보 입력')

st.selectbox('성별',('남','여'))
st.text_input('나이')
st.selectbox('학력',('초졸 이하','중졸','고졸 이상'))
st.text_input('소득','0')
st.text_input('아이 수','0')
st.text_input('고용일수','0')
st.selectbox('직장 전화',('있음','없음'))
st.selectbox('휴대폰',('있음','없음'))
st.selectbox('이메일',('있음','없음'))
st.selectbox('자동차',('있음','없음'))
st.selectbox('부동산',('있음','없음'))
st.selectbox('소득 형태',('Commercial associate', 'Pensioner', 'State servant', 'Student', 'Working'))
st.selectbox('가족 형태',('Civil marriage', 'Married', 'Separated', 'Single / not married', 'Widow'))
st.selectbox('집 형태',('Co-op apartment',
 'House / apartment',
 'Municipal apartment',
 'Office apartment',
 'Rented apartment',
 'With parents'))
st.selectbox('직업',('Accountants',
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
 '없음'))

st.sidebar()