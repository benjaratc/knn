#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# 1.โหลด csv เข้าไปใน Python Pandas

# In[2]:


colnames=['class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315', 'Proline']
df = pd.read_csv('../Desktop/DataCamp/wine.csv',names=colnames)
df


# 2. เขียนโค้ดแสดง หัว10แถว ท้าย10แถว และสุ่ม10แถว
# 

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.sample(10)


# 3. เช็คว่ามีข้อมูลที่หายไปไหม สามารถจัดการได้ตามความเหมาะสม

# In[6]:


df.isnull().any()


# 4. ใช้ info และ describe อธิบายข้อมูลเบื้องต้น

# In[7]:


df.info()


# In[8]:


df.describe()


# 5. ใช้ pairplot ดูความสัมพันธ์เบื้องต้นของ features ที่สนใจ

# In[9]:


sns.pairplot(df[['Alcalinity_of_ash','Flavanoids','OD280/OD315','Total_phenols']])


# 6. ใช้ displot เพื่อดูการกระจายของแต่ละคอลัมน์

# In[10]:


sns.distplot(df['Alcohol'])


# In[11]:


sns.distplot(df['Alcalinity_of_ash'])


# In[12]:


sns.distplot(df['Proline'])


# 7. ใช้ heatmap ดูความสัมพันธ์ของคอลัมน์ที่สนใจ

# In[13]:


fig = plt.figure(figsize = (12,8))
sns.heatmap(df.corr(), annot = df.corr())


# 8. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation สูงสุด

# In[14]:


sns.scatterplot(data = df, y = 'class', x ='Alcalinity_of_ash')


# 9. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation ต่ำสุด

# In[15]:


sns.scatterplot(data = df, y = 'Total_phenols', x ='Flavanoids')


# 10. สร้าง histogram ของ feature ที่สนใจ

# In[16]:


plt.hist(df['Flavanoids'], bins = 20)


# In[17]:


plt.hist(df['Total_phenols'], bins = 30)


# 11. สร้าง box plot ของ features ที่สนใจ

# In[18]:


sns.boxplot(data = df, x = 'class', y = 'Flavanoids', orient = 'v')


# In[19]:


sns.boxplot(data = df, x = 'class', y = 'Proanthocyanins', orient = 'v')


# In[20]:


sns.boxplot(data = df, x = 'class', y = 'Alcalinity_of_ash', orient = 'v')


# In[21]:


sns.boxplot(data = df, x = 'class', y = 'Magnesium', orient = 'v')


# In[22]:


sns.boxplot(data = df, x = 'class', y = 'Color_intensity', orient = 'v')


# 13. ทำ Data Visualization อื่นๆ (แล้วแต่เลือก)

# In[23]:


fig = plt.figure(figsize = [12,8])
sns.stripplot(x='class',y= 'Color_intensity',data =df)


# In[24]:


fig = plt.figure(figsize = [12,8])
sns.swarmplot(x='class',y= 'Proanthocyanins',data =df)


# In[25]:


fig = plt.figure(figsize = [12,8])
sns.swarmplot(x='class',y= 'Flavanoids',data =df)


# 14. พิจารณาว่าควรทำ Normalization หรือ Standardization หรือไม่ควร พร้อมให้เหตุผล 

# ควรทำ Normalization เพราะ x ไม่เป็น normal distribution

# In[26]:


from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score


# In[27]:


min_max_scaler = MinMaxScaler()


# In[28]:


y = df['class']
y


# In[29]:


X_minmax = min_max_scaler.fit_transform(df.drop('class', axis = 1))
X_minmax


# In[30]:


print(X_minmax.shape)


# 12. สร้าง train/test split ของข้อมูล สามารถลองทดสอบ 70:30, 80:20, 90:10 ratio ได้ตามใจ

# In[31]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax,y,test_size =0.3, random_state = 20)


# In[32]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[33]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)


# In[34]:


predicted = knn.predict(X_test)
predicted


# 16. วัดผลโมเดล โดยใช้ confusion matrix และ ประเมินผลด้วยคะแนน Accuracy, F1 score, Recall, Precision

# In[35]:


confusion_matrix(predicted, y_test)


# In[36]:


print('accuracy score',accuracy_score(y_test,predicted))
print('precision score',precision_score(y_test,predicted,average = 'micro'))
print('recall_score',recall_score(y_test,predicted,average = 'micro'))
print('f1 score',f1_score(y_test,predicted,average = 'micro'))


# 17. หาค่า K ที่ดีที่สุด สำหรับ Dataset นี้

# In[37]:


accuracy_lst = []

for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,y_train)
    predicted_i = knn.predict(X_test)
    accuracy_lst.append(accuracy_score(y_test,predicted_i))


# In[38]:


accuracy_lst


# In[39]:


plt.figure(figsize = (12,8))
plt.plot(range(1,30), accuracy_lst,color = 'black',linestyle = 'dashed', marker = 'o', 
         markerfacecolor = 'blue', markersize = 7)
plt.xlabel('Number of K')
plt.ylabel('Accuracy')


# # Standardization

# In[40]:


from sklearn.preprocessing import StandardScaler 


# In[41]:


X = df.drop('class', axis = 1)
y = df['class']


# In[42]:


sc_X =  StandardScaler()
X1 = sc_X.fit_transform(X)


# In[43]:


X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size =0.3, random_state = 20)


# In[44]:


knn3 = KNeighborsClassifier(n_neighbors = 5)
knn3.fit(X_train,y_train)


# In[45]:


predicted3 = knn3.predict(X_test)
predicted3 


# In[46]:


confusion_matrix(predicted3, y_test)


# In[47]:


print('accuracy score',accuracy_score(y_test,predicted3))
print('precision score',precision_score(y_test,predicted3,average = 'micro'))
print('recall_score',recall_score(y_test,predicted3,average = 'micro'))
print('f1 score',f1_score(y_test,predicted3,average = 'micro'))


# In[48]:


accuracy_lst3 = []

for i in range(1,30):
    knn3 = KNeighborsClassifier(n_neighbors = i)
    knn3.fit(X_train,y_train)
    predicted_i = knn3.predict(X_test)
    accuracy_lst3.append(accuracy_score(y_test,predicted_i))


# In[49]:


accuracy_lst3


# In[50]:


plt.figure(figsize = (12,8))
plt.plot(range(1,30), accuracy_lst3,color = 'black',linestyle = 'dashed', marker = 'o', 
         markerfacecolor = 'blue', markersize = 7)
plt.xlabel('Number of K')
plt.ylabel('Accuracy')


# 18. เลือกเฉพาะ features ที่สนใจมาเทรนโมเดล และวัดผลเปรียบเทียบกับแบบ all-features

# In[51]:


X_minmax = min_max_scaler.fit_transform(df[['Alcalinity_of_ash','Flavanoids','Hue','Proline']])
X_minmax


# In[52]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax,y,test_size =0.3, random_state = 20)


# In[53]:


knn2 = KNeighborsClassifier(n_neighbors = 5)
knn2.fit(X_train,y_train)


# In[54]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[55]:


predicted2 = knn2.predict(X_test)
predicted2


# In[56]:


confusion_matrix(predicted2, y_test)


# In[57]:


print('accuracy score',accuracy_score(y_test,predicted2))
print('precision score',precision_score(y_test,predicted2,average = 'micro'))
print('recall_score',recall_score(y_test,predicted2,average = 'micro'))
print('f1 score',f1_score(y_test,predicted2,average = 'micro'))


# In[58]:


accuracy_lst2 = []

for i in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(X_train,y_train)
    predicted_i = knn2.predict(X_test)
    accuracy_lst2.append(accuracy_score(y_test,predicted_i))


# In[59]:


accuracy_lst2


# 19. ทำ Visualization ของค่า K ค่าต่างๆ

# In[60]:


plt.figure(figsize = (12,8))
plt.plot(range(1,30), accuracy_lst2,color = 'black',linestyle = 'dashed', marker = 'o', 
         markerfacecolor = 'blue', markersize = 7)
plt.xlabel('Number of K')
plt.ylabel('Accuracy')


# 20. สามารถใช้เทคนิคใดก็ได้ตามที่สอนมา แล้วให้ผลลัพธ์ที่ดีที่สุดที่เป็นไปได้

# In[61]:


X = df.drop('class', axis = 1)
y = df['class']


# In[62]:


sc_X =  StandardScaler()
X2 = sc_X.fit_transform(X)


# In[63]:


X_train,X_test,y_train,y_test = train_test_split(X1,y,test_size =0.3, random_state = 20)


# In[64]:


knn4 = KNeighborsClassifier(n_neighbors = 27)
knn4.fit(X_train,y_train)


# In[65]:


predicted4 = knn4.predict(X_test)
predicted4


# In[66]:


confusion_matrix(predicted4, y_test)


# In[67]:


print('accuracy score',accuracy_score(y_test,predicted4))
print('precision score',precision_score(y_test,predicted4,average = 'micro'))
print('recall_score',recall_score(y_test,predicted4,average = 'micro'))
print('f1 score',f1_score(y_test,predicted4,average = 'micro'))


# In[68]:


accuracy_lst4 = []

for i in range(1,30):
    knn4 = KNeighborsClassifier(n_neighbors = i)
    knn4.fit(X_train,y_train)
    predicted_i = knn4.predict(X_test)
    accuracy_lst4.append(accuracy_score(y_test,predicted_i))


# In[69]:


accuracy_lst4


# In[70]:


plt.figure(figsize = (12,8))
plt.plot(range(1,30), accuracy_lst4,color = 'black',linestyle = 'dashed', marker = 'o', 
         markerfacecolor = 'blue', markersize = 7)
plt.xlabel('Number of K')
plt.ylabel('Accuracy')

