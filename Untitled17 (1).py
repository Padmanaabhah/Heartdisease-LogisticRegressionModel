#!/usr/bin/env python
# coding: utf-8

# In[1180]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
# Importing Pandas and NumPy
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[1181]:


heart_data = pd.read_csv('heart.csv',index_col=None)


# In[1182]:


heart_data.head()


# In[1183]:


heart_data.describe()


# In[1184]:


heart_data.shape


# In[1185]:


heart_data.isnull().sum()


# In[1186]:


heart_data.info()


# In[1187]:


heart_data.corr()


# In[1188]:


heart_data


# In[1189]:


def convert_trestbps(x):
    if x < 120:
        return 'Normal'
    elif x >= 120 and x <=129:
        return 'Elevated'
    elif x >= 130 and x <= 139:
        return 'HighBP1'
    elif x >= 140 and x < 180:
        return 'HighBP2'
    else:
        return 'HypertensiveCrisis'


# In[1190]:


heart_data['trestbps'] = heart_data['trestbps'].apply(convert_trestbps)


# In[1191]:


def convert_cholestral(x):
    if x < 200:
        return 'Normal'
    elif x >= 200 and x <=239:
        return 'Intermediate'
    else:
        return 'High'    


# In[1192]:


heart_data['chol'] = heart_data['chol'].apply(convert_cholestral)


# In[1193]:


heart_data['chol'] 


# In[1194]:


def convert_restecg(x):
    if x == 0:
        return 'probable left ventricular hypertrophy'
    elif x == 1:
        return 'normal'
    else:
        return 'abnormalities in the T wave or ST segment'    


# In[1195]:


heart_data['restecg'] = heart_data['restecg'].apply(convert_restecg)


# In[1196]:


def convert_heartrate_acheived(a):    
    x = a['age']
    y = a['thalach'] 
    if x <= 20:
        if y >= 200:
            return "High"
        else:
            return 'Normal'
    if x <= 30:
        if y >= 190:
            return "High"
        else:
            return 'Normal'
    if x <= 35:
        if y >= 185:
            return "High"
        else:
            return 'Normal'
    if x <= 40:
        if y >= 180:
            return "High"
        else:
            return 'Normal'
    if x <= 45:
        if y >= 175:
            return "High"
        else:
            return 'Normal'
    if x <= 50:
        if y >= 170:
            return "High"
        else:
            return 'Normal'
    if x <= 55:
        if y >= 165:
            return "High"
        else:
            return 'Normal'
    if x <= 60:
        if y >= 160:
            return "High"
        else:
            return 'Normal'
    if x <= 65:
        if y >= 155:
            return "High"
        else:
            return 'Normal'
    if x > 65:
        if y >= 150:
            return "High"
        else:
            return 'Normal'
    
    
        


# In[1197]:


heart_data


# In[1198]:


heart_data.apply(lambda x: convert_heartrate_acheived(x), axis=1)


# In[1199]:


def gender_map(x):
    if x == 1:
        return 'Male'
    elif x == 0:
        return 'Female' 


# In[1200]:


heart_data['sex'] = heart_data['sex'].apply(gender_map)


# In[1201]:


def cp_map(x):
    if x == 0:
        return 'typical angina'
    elif x == 1:
        return 'atypical angina'
    elif x == 2:
        return 'non-anginal pain'
    elif x == 3:
        return 'asymptomatic'


# In[1202]:


heart_data['cp'] = heart_data['cp'].apply(cp_map)


# In[1203]:


def fbs_map(x):
    if x == 1:
        return 'Yes'
    elif x == 0:
        return 'No' 


# In[1204]:


heart_data['fbs'] = heart_data['fbs'].apply(fbs_map)


# In[1205]:


def exang_map(x):
    if x == 1:
        return 'Yes'
    elif x == 0:
        return 'No' 


# In[1206]:


heart_data['exang'] = heart_data['exang'].apply(exang_map)


# In[1207]:


def slope_map(x):
    if x == 0:
        return 'descending'
    elif x == 1:
        return 'flat'
    elif x == 2:
        return 'ascending' 


# In[1208]:


heart_data['slope'] = heart_data['slope'].apply(slope_map)


# In[1209]:


def thal_map(x):
    if x == 0:
        return 'None'
    elif x == 1:
        return 'Fixed defect'
    elif x == 2:
        return 'Normal blood flow'
    elif x == 3:
        return 'reversible defect'


# In[1210]:


heart_data['thal'] = heart_data['thal'].apply(thal_map)


# In[1211]:


def ca_map(x):
    if x == 0:
        return 'Low'
    elif x == 1:
        return 'Medium'
    elif x == 2:
        return 'High'
    elif x == 3:
        return 'Very High'


# In[1212]:


heart_data['ca'] = heart_data['ca'].apply(ca_map)


# In[1213]:


heart_data['ca'].unique()


# In[1214]:


heart_data['thal'].unique()


# In[1215]:


heart_data


# In[1216]:


ax = sns.countplot(x="target",hue="sex", data=heart_data)


# In[1217]:


sns.displot(heart_data, x="age", hue="target", bins=10,multiple="dodge")
plt.show()


# In[1218]:


dummy1 = pd.get_dummies(heart_data[['sex','cp','trestbps','chol','fbs','restecg','exang','slope','ca','thal']],drop_first=True)

heart_data = pd.concat([heart_data, dummy1], axis=1)


# In[1219]:


heart_data.head()


# In[1220]:


heart_data = heart_data.drop(['sex','cp','trestbps','chol','fbs','restecg','exang','slope','ca','thal'],1)


# In[1221]:


heart_data.head()


# In[1222]:


heart_data.info()


# In[1223]:


heart_data.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# In[1224]:


sns.boxplot(heart_data['oldpeak'])


# In[1225]:


heart_data.isnull().sum()


# In[1226]:


from sklearn.model_selection import train_test_split


# In[1227]:


X = heart_data.drop(['target'], axis=1)
X.head()


# In[1228]:


y = heart_data['target']
y.head()


# In[1229]:


X_train,X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=100)


# In[1230]:


from sklearn.preprocessing import StandardScaler


# In[1231]:


heart_data


# In[1232]:


scaler = StandardScaler()
X_train[['age','thalach','oldpeak']] = scaler.fit_transform(X_train[['age','thalach','oldpeak']])


# In[1233]:


X_train


# In[1234]:


is_heart_attack = (sum(heart_data['target'])/ len(heart_data['target']))
is_heart_attack


# In[1235]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1236]:


plt.figure(figsize = (20,10))
sns.heatmap(heart_data.corr(), annot = True)
plt.show()


# In[1237]:


import statsmodels.api as sm


# In[1238]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1239]:


X_train = X_train.drop(['trestbps_HypertensiveCrisis'], axis=1)


# In[1240]:


logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1241]:


X_train = X_train.drop(['restecg_normal'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1242]:


X_train = X_train.drop(['thal_None'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1243]:


X_train = X_train.drop(['exang_Yes'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1244]:


X_train = X_train.drop(['fbs_Yes'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1245]:


X_train = X_train.drop(['trestbps_HighBP1'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1246]:


X_train = X_train.drop(['trestbps_HighBP2'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1247]:


X_train = X_train.drop(['thal_Normal blood flow'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1248]:


X_train = X_train.drop(['ca_Medium'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1249]:


X_train = X_train.drop(['restecg_probable left ventricular hypertrophy'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1250]:


X_train = X_train.drop(['slope_descending'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1251]:


X_train = X_train.drop(['ca_Very High'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1252]:


X_train = X_train.drop(['chol_Normal'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1253]:


X_train = X_train.drop(['cp_non-anginal pain'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1254]:


X_train = X_train.drop(['trestbps_Normal'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1255]:


X_train = X_train.drop(['chol_Intermediate'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1256]:


X_train = X_train.drop(['thalach'], axis=1)
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family= sm.families.Binomial())
logm1.fit().summary()


# In[1257]:


X_train = X_train.drop(['age'], axis=1)
X_train_sm = (sm.add_constant(X_train))
logm1 = sm.GLM(y_train,X_train_sm, family= sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[1258]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[1259]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[1260]:


X_train_sm.columns


# In[1261]:


y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[1262]:


y_train_pre_final = pd.DataFrame({"target" : y_train.values, "traget_prob": y_train_pred})


# In[1263]:


y_train_pre_final['predicted'] = y_train_pre_final.traget_prob.map(lambda x:1 if x > 0.5 else 0)
y_train_pre_final.head()


# In[1264]:


from sklearn import metrics


# In[1265]:


confusion = metrics.confusion_matrix(y_train_pre_final.target, y_train_pre_final.predicted)
confusion


# In[1266]:


print(metrics.accuracy_score(y_train_pre_final.target, y_train_pre_final.predicted))


# In[1267]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[1268]:


# Sensitivity
TP/float(TP+FN)


# In[1269]:


# Specificity

TN/(TN+FP)


# In[1270]:


# False Positive Rate
FP/ float(TN+FP)


# In[1271]:


# Positive predictive rate
TP/ float(TP+FP)


# In[1272]:


# Negative predictive rate
TN/ float(TN+FN)


# In[1273]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[1274]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pre_final.target, y_train_pre_final.traget_prob, drop_intermediate = False )


# In[1275]:


draw_roc(y_train_pre_final.target, y_train_pre_final.traget_prob)


# In[1276]:


numbers = [float (x) /10 for x in range(10)]
for i in numbers:
    y_train_pre_final[i]=y_train_pre_final.traget_prob.map(lambda x: 1 if x > i else 0)

y_train_pre_final.head(20)


# In[1277]:


cutoff_df = pd.DataFrame(columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pre_final.target, y_train_pre_final[i])
    total1 = sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] = [ i, accuracy,sensi,speci]
print(cutoff_df)


# In[1278]:


cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[1279]:


y_train_pre_final['final_predicted'] = y_train_pre_final.traget_prob.map(lambda x:1 if x > 0.5 else 0)


# In[1280]:


y_train_pre_final.head(20)


# In[1281]:


print(metrics.accuracy_score(y_train_pre_final.target, y_train_pre_final.predicted))


# In[1282]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[1283]:


# Sensitivity
TP/float(TP+FN)


# In[1284]:


# Specificity

TN/(TN+FP)


# In[1285]:


# False Positive Rate
FP/ float(TN+FP)


# In[1286]:


# Positive predictive rate
TP/ float(TP+FP)


# In[1287]:


# Negative predictive rate
TN/ float(TN+FN)


# In[1288]:


confusion = metrics.confusion_matrix(y_train_pre_final.target, y_train_pre_final.predicted)


# In[1289]:


confusion


# In[1290]:


# Precision
confusion[1,1]/(confusion[0,1] + confusion[1,1])


# In[1291]:


# Recall
confusion[1,1]/ (confusion[1,0] + confusion[1,1])


# In[1292]:


from sklearn.metrics import precision_score, recall_score


# In[1293]:


precision_score(y_train_pre_final.target, y_train_pre_final.predicted)


# In[1294]:


recall_score(y_train_pre_final.target, y_train_pre_final.predicted)


# In[1295]:


from sklearn.metrics import precision_recall_curve


# In[1296]:


p, r , threshold = precision_recall_curve(y_train_pre_final.target, y_train_pre_final.traget_prob)


# In[1297]:


plt.plot(threshold, p[:-1], "g-")
plt.plot(threshold, r[:-1], "r-")
plt.show()


# In[1298]:


X_test[['age','thalach','oldpeak']] = scaler.fit_transform(X_test[['age','thalach','oldpeak']])


# In[1299]:


X_test


# In[1300]:


col = X_train_sm.columns[1:]
col


# In[1301]:


X_test_sm = sm.add_constant(X_test[col]) 


# In[1302]:


X_test_sm


# In[1303]:


y_test_pred = res.predict(X_test_sm[X_train_sm.columns])


# In[1304]:


y_test_pred[:10]


# In[1305]:


y_pred_1 = pd.DataFrame(y_test_pred)


# In[1306]:


y_pred_1['Id'] = y_pred_1.index


# In[1307]:


y_pred_1.drop(0, axis=1)


# In[1308]:


y_test_df = pd.DataFrame(y_test)


# In[1309]:


y_test_df['Id'] = y_test_df.index


# In[1310]:


y_test_df


# In[1311]:


y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[1312]:


y_test_df


# In[1313]:


y_pred_1


# In[1314]:


y_pred_final = y_test_df.merge(y_pred_1, how='inner',on='Id')


# In[1315]:


y_pred_final.head()


# In[1316]:


y_pred_final.index = y_pred_final['Id']


# In[1317]:


y_pred_final = y_pred_final.drop('Id', axis=1)


# In[1318]:


y_pred_final


# In[1319]:


y_pred_final = y_pred_final.rename(columns={0:'target_prob'})


# In[1320]:


y_pred_final.head()


# In[1321]:


y_pred_final['final_predicted'] = y_pred_final.target_prob.map(lambda x: 1 if x > 0.5 else 0)


# In[1322]:


metrics.accuracy_score(y_pred_final.target, y_pred_final.final_predicted)


# In[1323]:


confusion2 = metrics.confusion_matrix(y_pred_final.target, y_pred_final.final_predicted)
confusion2


# In[1324]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[1325]:


#Sensitivty
TP / float(TP+FN)


# In[1326]:


# Let us calculate specificity
TN / float(TN+FP)

