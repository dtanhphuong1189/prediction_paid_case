# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import date
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
import base64
import time

#Build model
filename = 'finalized_model_sf.sav'
model = joblib.load(filename)


#Part 2: Hiển thị kết quả trên Streamlit
st.title("Data Science")
st.write("Phân loại khách hàng trả nợ vay")
menu=["New Prediction","Upload file"]
choice=st.sidebar.selectbox("Menu",menu)
if choice == "New Prediction":
    st.subheader("Make New Prediction")
    st.write("#### Input/ Select data")
    name=st.text_input("Name: ")
    #sex=st.radio("Sex",options=["Male","Female"])
    #today=date.today()
    #ngay_sinh=st.date_input("Ngày sinh khách hàng",datetime.date(1921,1,1))
    #tuoi=today.year-ngay_sinh.year
    age=st.slider("tuoi",min_value=18,max_value=80,step=1)

    gia_san_pham=st.number_input("ĐIỀN GIÁ SẢN PHẨM",0)
    tien_ky=st.number_input("ĐIỀN SỐ TIỀN KỲ",100000)
    tong_so_ky=st.slider("TỔNG SỐ KỲ KHÁCH HÀNG PHẢI TRẢ",1,100,1)
    so_lan_da_tra=st.slider("SỐ LẦN KHÁCH HÀNG ĐÃ TRẢ",0,300,1)
    dpd=st.number_input("ĐIỀN SỐ DPD HỢP ĐỒNG",1)

    phone=st.slider("SỐ ĐIỆN THOẠI CUNG CẤP LÀ BAO NHIÊU SỐ",1,20,1)

    khach_hang_duoc_mien_giam=st.selectbox(label="Lựa chọn 0 nếu không được giảm - 1 nếu được giảm",options=[0,1])
    so_cong_ty_vay=st.slider("SỐ CÔNG TY VAY",0,10,1)
    so_lan_xuat_hien=st.slider("SỐ LẦN XUẤT HIỆN",1,10,1)



        #make new prediction
        #Sex, Age, Pclass, Sibsb, Parch, Fare
        #data=data[["Sex","Pclass","SibSp","Parch","Fare","Survived"]]

    new_data=([[dpd,so_lan_da_tra,gia_san_pham,age,phone,khach_hang_duoc_mien_giam,tien_ky,so_cong_ty_vay,
    so_lan_xuat_hien,tong_so_ky]])
    prediction=model.predict(new_data)
    predict_probability=model.predict_proba(new_data)

    if prediction[0] == 1:

        st.subheader('Khách hàng {} sẽ có xác xuất trả tiền là {}%'.format(name , 
                                                        round(predict_probability[0][1]*100 , 2)))
    else:
        st.subheader('Khách hàng {} sẽ có xác xuất không trả tiền là {}%'.format(name, 
                                                        round(predict_probability[0][0]*100 , 2)))
    

elif choice=="Upload file":
    st.subheader("Upload file")
    data_file = st.file_uploader("Upload CSV",type=['csv'])
    if st.button("Process"):
        if data_file is not None:
            df = pd.read_csv(data_file)
            
            prediction=model.predict(df)
            #print(prediction)
            predict_proba=[]
            for i in range(df.shape[0]):
                if model.predict_proba(df)[i][0] >=0.5:
                    predict_proba.append(model.predict_proba(df)[i][0]*100)
                else:
                    predict_proba.append(model.predict_proba(df)[i][1]*100)
            #print("predict_proba",predict_proba)
            df["Prediction"]=prediction
            df["Prediction"]=df["Prediction"].map(lambda x:"Kha nang tra cao" if x==1 else "Kha nang tra thap")
            df["Predict_proba"]=predict_proba
            st.dataframe(df)
            results = df.to_csv(index=False,encoding='utf-8')
            b64 = base64.b64encode(results.encode()).decode()  # some strings <-> bytes conversions necessary here
            timestr=time.strftime("%d%m%y - %H%M%S")
            new_file="new file {} .csv" .format(timestr)
            st.markdown("DOWNLOAD HERE")
            href = f'<a href="data:file/csv;base64,{b64}" download="{new_file}">Click Here!!</a>'
            st.markdown(href, unsafe_allow_html=True)