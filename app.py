import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are Your Mushrooms edible Or Poisonous 🍄")
    st.sidebar.markdown("Are Your Mushrooms edible Or Poisonous 🍄")
    @st.cache(persist=True) 
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label= LabelEncoder()
        for col in data.columns:
            data[col]= label.fit_transform(data[col])
        return data



    @st.cache(persist=True)
    def split(df):
        y=df.type
        x=df.drop(columns=['type'])
        x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=0)
        return x_train,x_test,y_train,y_test
    def plot_metrics(metrics_list):
        if 'Confusion Matix' in metrics_list:
            st.subheader("Confusion Matix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=class_names)
            st.pyplot()
    
        if 'ROC Curve'in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model,x_test,y_test)
            st.pyplot()
        if 'Precision-Recall-curve' in metrics_list:
            st.subheader("Precision-Recall-curve")
            plot_precision_recall_curve(model,x_test,y_test)
            st.pyplot()


    df = load_data()

    x_train,x_test,y_train,y_test = split(df)
    class_names=['edible','poisonous']
    st.sidebar.subheader("Chosse Classifier")
    classifier=st.sidebar.selectbox("Classifier",("Support Vector Machine(SVM)","Logistic regression(LR)","RandomForestClassifier"))
    if classifier == 'Support Vector Machine(SVM)':
         st.sidebar.subheader("Model Hyperparameters")
         C= st.sidebar.number_input("C (Regularization Parameter)",0.01,10.0,step=0.01,key='C')
         kernel =st.sidebar.radio("Kernel",("rbf","linear","sigmoid"),key='kernel')
         gamma =st.sidebar.radio("Gamma" "(Kernal Coeffiecient)",("scale","auto"),key='gamma')
         metrics =st.sidebar.multiselect("What metrics to plot:",("Confusion Matix","ROC Curve","Precision-Recall-curve"))
         
         if  st.sidebar.button("classify",key='classsify'):
             st.subheader("Support Vactor Machines Subheader (SVM) Result....")
             model=SVC(C=C,kernel=kernel,gamma=gamma)
             model.fit(x_train,y_train)
             accuracy=model.score(x_test,y_test)
             y_pred=model.predict(x_test)
             st.write("accuracy :",accuracy.round(2)) 
             st.write("preccisons Score :",precision_score(y_test,y_pred,labels=class_names).round(2)) 
             st.write(" Recall :",recall_score(y_test,y_pred,labels=class_names).round(2)) 
             plot_metrics(metrics)

    if classifier == 'Logistic regression(LR)':
         st.sidebar.subheader("Model Hyperparameters")
         C= st.sidebar.number_input("C (Regularization Parameter)",0.01,10.0,step=0.01,key='C')
         max_iter= st.sidebar.slider("Maximum No Of Iterations",100,500,key='max_iter')

         metrics =st.sidebar.multiselect("What metrics to plot:",("Confusion Matix","ROC Curve","Precision-Recall-curve"))

         if  st.sidebar.button("classify",key='classsify'):
             st.subheader(" Logistic regression(LR) Result....")
             model=LogisticRegression(C=C,max_iter=max_iter)
             model.fit(x_train,y_train)
             accuracy=model.score(x_test,y_test)
             y_pred=model.predict(x_test)
             st.write("accuracy :",accuracy.round(2)) 
             st.write("preccisons Score :",precision_score(y_test,y_pred,labels=class_names).round(2)) 
             st.write(" Recall :",recall_score(y_test,y_pred,labels=class_names).round(2)) 
             plot_metrics(metrics)

    if classifier == 'RandomForestClassifier':
         st.sidebar.subheader("Model Hyperparameters")
         max_depth= st.sidebar.number_input("The Maximum Depth Of The Tree",1,20,step=1,key='max_depth')
         n_estimators= st.sidebar.number_input("The No. Of tress in the Forests",100,5000,step=10,key='n_estimators')
         bootstrap= st.sidebar.radio("Bootstrap sample the no of trees:",('True','False'),key='bootstrap')
         metrics =st.sidebar.multiselect("What metrics to plot:",("Confusion Matix","ROC Curve","Precision-Recall-curve"))

         if  st.sidebar.button("classify",key='classsify'):
             st.subheader(" RandomForestClassifier Model  Result....")
             model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,bootstrap = bootstrap,n_jobs=-1)
             model.fit(x_train,y_train)
             accuracy=model.score(x_test,y_test)
             y_pred=model.predict(x_test)
             st.write("accuracy :",accuracy.round(2)) 
             st.write("preccisons Score :",precision_score(y_test,y_pred,labels=class_names).round(2)) 
             st.write(" Recall :",recall_score(y_test,y_pred,labels=class_names).round(2)) 
             plot_metrics(metrics)
        



    if st.sidebar.checkbox("Show Raw data",False):
        st.subheader("Mushrooms data (Classsification)")
        st.write(df)
         


    






if __name__ == '__main__':
    main()


