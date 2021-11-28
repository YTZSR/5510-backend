#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:18:52 2021

@author: zhangjiawei
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from collections import Counter

# read in data
final_all = pd.read_csv("final_all.csv")

#  =============undersample majority class manuallly ==========================
np.random.seed(245)
chosen_idx = np.random.choice(final_all.loc[final_all.y == 0].shape[0], replace=False, size=2000)
majotiry = final_all.loc[final_all.y == 0].iloc[chosen_idx]
#majotiry = final_all.loc[final_all.y == 0].sample(n=2000)
minority  = final_all.loc[final_all.y == 1]
final_al = pd.concat([majotiry,minority])

#split train and test
X_ful = final_al.drop(columns = ["period",'cust_id','rpt_clb_cd_x','aud_ind_x','dft_time',"y"]).replace(np.nan,0).replace(np.inf,0).replace(-np.inf,0)
y_ful = final_al.loc[:,"y"]
X_train, X_test, y_train, y_test = train_test_split(X_ful, y_ful, test_size=0.2, random_state=233)

#=====================feature engineering======================================
final_all["to_pay_exp"] = (final_all.ITM_1040+0.01)/(final_all.ITM_1280+0.01)
final_all["inventory_vs_expense"] = (final_all.ITM_1090+0.01)/(final_all.ITM_2020+0.01)
final_all["sales_profit"] = (final_all.ITM_2170+0.01)/(final_all.ITM_2010+0.01)
final_all["gross_profit"] = (final_all.ITM_1090+0.01)/(final_all.ITM_2040+0.01)
final_all["roa"] = (final_all.ITM_2170+0.01)/(final_all.ITM_1370+0.01)
final_all["profit_net_asset"] = (final_all.ITM_2170+0.01)/(final_all.ITM_1370 - final_all.ITM_1620+0.01)
final_all["liab_ratio"] = (final_all.ITM_1620+0.01)/(final_all.ITM_1370+0.01)
final_all["liab_equity"] = (final_all.ITM_1620+0.01)/(final_all.ITM_1720+0.01)
final_all["tangible_asset_liab"] = (final_all.ITM_1620+0.01)/(final_all.ITM_1720 - final_all.ITM_1250+0.01)
final_all["secure_liab"] = (final_all.ITM_2150+0.01)/(final_all.ITM_2160+0.01)
final_all["liquidity"] = (final_all.ITM_1210+0.01)/(final_all.ITM_1680+0.01)
final_all["subsidiary"] = (final_all.ITM_2170-final_all.ITM_2172+0.01)/(final_all.ITM_2170+0.01)

# =========================== Train test split ===============================
#ubdersample majority class manuallly
np.random.seed(245)
chosen_idx = np.random.choice(final_all.loc[final_all.y == 0].shape[0], replace=False, size=2000)
majotiry = final_all.loc[final_all.y == 0].iloc[chosen_idx]
#majotiry = final_all.loc[final_all.y == 0].sample(n=2000)
minority  = final_all.loc[final_all.y == 1]
final_al = pd.concat([majotiry,minority])

#split train and test
X_ful = final_al.drop(columns = ["period",'cust_id','rpt_clb_cd_x','aud_ind_x','dft_time',"y"]).replace(np.nan,0).replace(np.inf,0).replace(-np.inf,0)
y_ful = final_al.loc[:,"y"]
X_train, X_test, y_train, y_test = train_test_split(X_ful, y_ful, test_size=0.2, random_state=233)

#==============================SMOTE===========================================
y_train.value_counts() # 0:61588, 1:495
y_test.value_counts()

smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(X_train, y_train)

print(Counter(y_ful)) #Counter({0: 2000, 1: 609})
print(Counter(y_smo)) #Counter({0: 1582, 1: 1582})

# =============================== LR Models ===================================
# Logistics regression 

lr = LogisticRegression(random_state=0).fit(X_smo, y_smo)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)
#print(metrics.recall_score(y_test, y_pred_lr, average='micro')) #0.725339862122286
#print(metrics.precision_score(y_test, y_pred_lr, average='micro')) #
confusion_matrix(y_test, y_pred_lr) #
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred_lr, target_names=target_names))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_lr)
print(metrics.auc(fpr, tpr)) #0.7266746411483254, np.random.seed(245)

# =========================== Boosting ========================================
#xgb
from xgboost import XGBClassifier
xgb = XGBClassifier(objective= 'binary:logistic',learning_rate =0.15, 
                    n_estimators=5000, max_depth=9,
                    min_child_weight=1, gamma=0, subsample=0.9,
                    use_label_encoder=False)
X_train_xgb = pd.DataFrame(X_smo,columns = X_ful.columns.tolist())
y_train_xgb = pd.Series(y_smo)
xgb.fit(X_train_xgb, y_train_xgb,eval_set=[(X_train_xgb, y_train_xgb), (X_test, y_test)],
        eval_metric = "logloss",early_stopping_rounds=20,verbose=True)
res_xgb = xgb.predict(X_test)
res = xgb.predict_proba(X_test)
print(classification_report(y_test, res_xgb, target_names=target_names))
fpr, tpr, thresholds = metrics.roc_curve(y_test, res_xgb)
print(metrics.auc(fpr, tpr)) #0.7600294442399705

#feature importance XGBoost
importance_xgb = pd.DataFrame(list(zip(X_ful.columns.tolist(),xgb.feature_importances_.tolist())),columns = ["feature","importance"])
unimt_xgb = importance_xgb.loc[importance_xgb.importance == 0].feature.tolist()

# =============================================================================
#catboost
from catboost import CatBoostClassifier,Pool
catboost_model = CatBoostClassifier(
    iterations=500,max_ctr_complexity=3,learning_rate=0.03,depth=9,
    random_seed= 123,od_type='Iter',loss_function = "Logloss", eval_metric='Logloss',
    od_wait=25,verbose=0) #,class_weights = [0.8,0.2]
catboost_model.fit(X_train_xgb, y_train_xgb,eval_set=(X_test, y_test))

##predict
pred_catboost = catboost_model.predict(X_test)
print(classification_report(y_test, pred_catboost, target_names=target_names))
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_catboost)
print(metrics.auc(fpr, tpr)) #0.7612486198012514

#feature importance
imp_df_cat = pd.DataFrame()
imp_df_cat["feature"] = X_ful.columns.tolist()
imp_df_cat["importance"] = catboost_model.get_feature_importance(Pool(X_test, label=y_test))

#==============================================================================
#lightgbm
import lightgbm as lgb
train_data_lgb = lgb.Dataset(data=X_train_xgb.replace(np.nan,0),label=y_train_xgb)
test_data_lgb = lgb.Dataset(data=X_test.replace(np.nan,0),label=y_test)
lgb_params = {
    'objective':"binary",
    'subsample': 0.623,
    'colsample_bytree': 0.7,
    'num_leaves':55,
    'max_depth': 7,
    'seed': 233,
    'bagging_freq': 1,
    'n_jobs': 4,
    'metric': {"auc"},
}

lgb_m = lgb.train(params=lgb_params, train_set=train_data_lgb, num_boost_round=200)
y_pred_lgb = lgb_m.predict(X_test)
y_pred_dummy_lgb = np.select(condlist = [y_pred_lgb > 0.5],choicelist = [1],default = 0)
print(classification_report(y_test, y_pred_dummy_lgb, target_names=target_names))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_dummy_lgb)
print(metrics.auc(fpr, tpr)) #0.7744295178505706


#

# ============================ Stacking ========================================
#modify on train data(drop unimportant columns, add xgb result into catboost)
#X_train_drop = X_train_xgb.drop(columns = unimt_xgb)
#X_test_drop = X_test.drop(columns = unimt_xgb)

res_train = xgb.predict_proba(X_train_xgb)
res_test = xgb.predict_proba(X_test)
res_train_lr = lr.predict_proba(X_smo)
res_test_lr = lr.predict_proba(X_test)
res_train_cat = catboost_model.predict_proba(X_train_xgb)
res_test_cat = catboost_model.predict_proba(X_test)

res_train_df = pd.DataFrame(res_train,columns = ["class0xgb","class1xgb"])
res_test_df = pd.DataFrame(res_test,columns = ["class0xgb","class1xgb"])
res_train_lr_df = pd.DataFrame(res_train_lr,columns = ["class0lr","class1lr"])
res_test_lr_df = pd.DataFrame(res_test_lr,columns = ["class0lr","class1lr"])
res_train_cat_df = pd.DataFrame(res_train_cat,columns = ["class0cat","class1cat"])
res_test_cat_df = pd.DataFrame(res_test_cat,columns = ["class0cat","class1cat"])

X_train_drop_f = pd.concat([X_train_xgb,res_train_df,res_train_lr_df,res_train_cat_df],axis = 1)
X_test_drop_f= pd.concat([X_test.reset_index(),res_test_df.reset_index(),res_test_lr_df,res_test_cat_df],axis = 1).drop(columns = ["index"])

#stack cat
catboost_model_2 = CatBoostClassifier(iterations=500,max_ctr_complexity=3,
                                      learning_rate=0.03,depth=9,random_seed= 123,
                                      od_type='Iter',loss_function = "Logloss", 
                                      eval_metric='Logloss',od_wait=25,verbose=0) #,class_weights = [0.8,0.2]
catboost_model_2.fit(X_train_drop_f, y_train_xgb,eval_set=(X_test_drop_f, y_test))

##predict
pred_catboost2 = catboost_model_2.predict(X_test_drop_f)
print(classification_report(y_test, pred_catboost2, target_names=target_names))
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_catboost2)
print(metrics.auc(fpr, tpr)) #0.785195068089805
df_imp_cat = catboost_model_2.get_feature_importance(prettified=True)

#stack xgb
xgb2 = XGBClassifier(objective= 'binary:logistic',nthread = 10)
xgb2.fit(X_train_drop_f, y_train_xgb,eval_set=[(X_train_drop_f, y_train_xgb), (X_test_drop_f, y_test)],
        eval_metric = "logloss",early_stopping_rounds=50,verbose=False)
res_xgb2 = xgb2.predict(X_test_drop_f)
res2 = xgb2.predict_proba(X_test_drop_f)
print(classification_report(y_test, res_xgb2, target_names=target_names))
fpr, tpr, thresholds = metrics.roc_curve(y_test, res_xgb2)
print(metrics.auc(fpr, tpr)) #0.7587182554287817
importance_xgb2 = pd.DataFrame(list(zip(X_test_drop_f.columns.tolist(),xgb2.feature_importances_.tolist())),columns = ["feature","importance"])

# =================== Simple by three models's outcome ========================
X_train_stack_f = pd.concat([res_train_df,res_train_lr_df,res_train_cat_df],axis = 1)
X_test_stack_f= pd.concat([res_test_df.reset_index(),res_test_lr_df,res_test_cat_df],axis = 1).drop(columns = ["index"])

catboost_model_2 = CatBoostClassifier(iterations=500,max_ctr_complexity=3,
                                      learning_rate=0.03,depth=9,random_seed= 123,
                                      od_type='Iter',loss_function = "Logloss", 
                                      eval_metric='Logloss',od_wait=25,verbose=0) #,class_weights = [0.8,0.2]
catboost_model_2.fit(X_train_stack_f, y_train_xgb,eval_set=(X_test_stack_f, y_test))
pred_catboost2 = catboost_model_2.predict(X_test_drop_f)
print(classification_report(y_test, pred_catboost2, target_names=target_names)) #0.7791682002208318

xgb2 = XGBClassifier(objective= 'binary:logistic',nthread = 10)
xgb2.fit(X_train_stack_f, y_train_xgb,eval_set=[(X_train_stack_f, y_train_xgb), (X_test_stack_f, y_test)],
        eval_metric = "logloss",early_stopping_rounds=50,verbose=False)
res_xgb2 = xgb2.predict(X_test_stack_f)
res2 = xgb2.predict_proba(X_test_stack_f)
print(classification_report(y_test, res_xgb2, target_names=target_names))
fpr, tpr, thresholds = metrics.roc_curve(y_test, res_xgb2)
print(metrics.auc(fpr, tpr)) #0.7612026131762973

