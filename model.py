#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:18:52 2021

@author: zhangjiawei
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier, Booster
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn import metrics
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from collections import Counter
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import pickle
import joblib
from sklearn.metrics import classification_report

# read in data
final_all = pd.read_csv("/Users/zhangjiawei/Desktop/T1/FTEC5510/final_all.csv")


def feature_engineering_train_test_split(final_all):
    #  ============= undersample majority class manuallly ====================
    np.random.seed(245)
    chosen_idx = np.random.choice(final_all.loc[final_all.y == 0].shape[0], replace=False, size=2000)
    majotiry = final_all.loc[final_all.y == 0].iloc[chosen_idx]
    minority = final_all.loc[final_all.y == 1]
    final_al = pd.concat([majotiry, minority])

    # ===================== split train and test ==============================
    X_ful = final_al.drop(columns=["period", 'cust_id', 'rpt_clb_cd_x', 'aud_ind_x', 'dft_time', "y"]).replace(np.nan,
                                                                                                               0).replace(
        np.inf, 0).replace(-np.inf, 0)
    y_ful = final_al.loc[:, "y"]
    X_train, X_test, y_train, y_test = train_test_split(X_ful, y_ful, test_size=0.2, random_state=233)

    final_all["to_pay_exp"] = (final_all.ITM_1040 + 0.01) / (final_all.ITM_1280 + 0.01)
    final_all["inventory_vs_expense"] = (final_all.ITM_1090 + 0.01) / (final_all.ITM_2020 + 0.01)
    final_all["sales_profit"] = (final_all.ITM_2170 + 0.01) / (final_all.ITM_2010 + 0.01)
    final_all["gross_profit"] = (final_all.ITM_1090 + 0.01) / (final_all.ITM_2040 + 0.01)
    final_all["roa"] = (final_all.ITM_2170 + 0.01) / (final_all.ITM_1370 + 0.01)
    final_all["profit_net_asset"] = (final_all.ITM_2170 + 0.01) / (final_all.ITM_1370 - final_all.ITM_1620 + 0.01)
    final_all["liab_ratio"] = (final_all.ITM_1620 + 0.01) / (final_all.ITM_1370 + 0.01)
    final_all["liab_equity"] = (final_all.ITM_1620 + 0.01) / (final_all.ITM_1720 + 0.01)
    final_all["tangible_asset_liab"] = (final_all.ITM_1620 + 0.01) / (final_all.ITM_1720 - final_all.ITM_1250 + 0.01)
    final_all["secure_liab"] = (final_all.ITM_2150 + 0.01) / (final_all.ITM_2160 + 0.01)
    final_all["liquidity"] = (final_all.ITM_1210 + 0.01) / (final_all.ITM_1680 + 0.01)
    final_all["subsidiary"] = (final_all.ITM_2170 - final_all.ITM_2172 + 0.01) / (final_all.ITM_2170 + 0.01)

    # ==============================SMOTE=======================================
    # ubdersample majority class manuallly
    np.random.seed(245)
    chosen_idx = np.random.choice(final_all.loc[final_all.y == 0].shape[0], replace=False, size=2000)
    majotiry = final_all.loc[final_all.y == 0].iloc[chosen_idx]
    # majotiry = final_all.loc[final_all.y == 0].sample(n=2000)
    minority = final_all.loc[final_all.y == 1]
    final_al = pd.concat([majotiry, minority])

    # split train and test
    X_ful = final_al.drop(columns=["period", 'cust_id', 'rpt_clb_cd_x', 'aud_ind_x', 'dft_time', "y"]).replace(np.nan,
                                                                                                               0).replace(
        np.inf, 0).replace(-np.inf, 0)
    y_ful = final_al.loc[:, "y"]
    X_train, X_test, y_train, y_test = train_test_split(X_ful, y_ful, test_size=0.2, random_state=233)

    # y_train.value_counts() # 0:61588, 1:495
    # y_test.value_counts()
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_resample(X_train, y_train)
    col_lst = X_ful.columns.tolist()

    return X_smo, X_test, y_smo, y_test, col_lst


def LR_model_train_save(path, X_smo, y_smo, X_test, y_test):
    # train
    lr = LogisticRegression(random_state=0).fit(X_smo, y_smo)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_lr)
    # print(metrics.auc(fpr, tpr)) #0.7266746411483254, np.random.seed(245)
    # save
    with open(path + "LR_model.pkl", 'wb') as file:
        pickle.dump(lr, file)
    return y_pred_lr, y_prob_lr


def XGB_classifier_train_save(path, X_smo, y_smo, X_test, y_test, col_lst):
    # train
    xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.15,
                        n_estimators=5000, max_depth=9,
                        min_child_weight=1, gamma=0, subsample=0.9,
                        use_label_encoder=False)
    # X_train_xgb = pd.DataFrame(X_smo,columns = col_lst)
    # y_train_xgb = pd.Series(y_smo)
    xgb.fit(X_smo, y_smo, eval_set=[(X_smo, y_smo), (X_test, y_test)],
            eval_metric="logloss", early_stopping_rounds=20, verbose=True)
    res_xgb = xgb.predict(X_test)
    res_prob = xgb.predict_proba(X_test)
    # important features
    importance_xgb = pd.DataFrame(list(zip(col_lst, xgb.feature_importances_.tolist())),
                                  columns=["feature", "importance"])
    importance_xgb_lst = importance_xgb.loc[importance_xgb.importance >= 0.01].feature.tolist()
    # save
    pickle.dump(xgb, open(path + "xgb_model.json", "wb"))
    return res_prob, res_xgb, importance_xgb_lst


def Catboost_classifier_train_save(path, X_smo, y_smo, X_test, y_test, col_lst):
    # train
    catboost_model = CatBoostClassifier(
        iterations=500, max_ctr_complexity=3, learning_rate=0.03, depth=9,
        random_seed=123, od_type='Iter', loss_function="Logloss", eval_metric='Logloss',
        od_wait=25, verbose=0)  # ,class_weights = [0.8,0.2]
    catboost_model.fit(X_smo, y_smo, eval_set=(X_test, y_test))
    pred_catboost = catboost_model.predict(X_test)
    pred_cat_prob = catboost_model.predict_proba(X_test)
    # important features
    imp_df_cat = pd.DataFrame()
    imp_df_cat["feature"] = col_lst
    imp_df_cat["importance"] = catboost_model.get_feature_importance(Pool(X_test, label=y_test))
    imp_cat = imp_df_cat.loc[imp_df_cat.importance >= 0.01].feature.tolist()
    # save
    catboost_model.save_model(path + "catboost_model.json")
    return pred_catboost, pred_cat_prob, imp_cat


def LightGBM_classifier_train_save(path, X_smo, y_smo, X_test, y_test, col_lst):
    # train
    X_train_xgb = pd.DataFrame(X_smo, columns=col_lst)
    y_train_xgb = pd.Series(y_smo)

    train_data_lgb = lgb.Dataset(data=X_train_xgb.replace(np.nan, 0), label=y_train_xgb)
    test_data_lgb = lgb.Dataset(data=X_test.replace(np.nan, 0), label=y_test)
    lgb_params = {
        'objective': "binary", 'subsample': 0.623, 'colsample_bytree': 0.7,
        'num_leaves': 55, 'max_depth': 7, 'seed': 233, 'bagging_freq': 1,
        'n_jobs': 4, 'metric': {"auc"}
    }

    lgb_m = lgb.train(params=lgb_params, train_set=train_data_lgb, num_boost_round=200,
                      valid_sets=[train_data_lgb, test_data_lgb])
    y_pred_lgb = lgb_m.predict(X_test)
    y_pred_dummy_lgb = np.select(condlist=[y_pred_lgb > 0.5], choicelist=[1], default=0)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_dummy_lgb)
    # light gbm imp
    imp_df_a = pd.DataFrame()
    imp_df_a["feature"] = X_train_xgb.columns.tolist()
    imp_df_a["importance_gain"] = lgb_m.feature_importance(importance_type='gain')
    imp_df_a["importance_split"] = lgb_m.feature_importance(importance_type='split')
    imp_lgb = imp_df_a.loc[imp_df_a.importance_gain >= 0.01].feature.tolist()
    # save
    joblib.dump(lgb_m, path + 'lgb_model.pkl')
    return y_pred_lgb, y_pred_dummy_lgb, imp_lgb


def Stacking_model_train_valid_save(path, X_smo, y_smo, X_test, y_test, col_lst):
    # load dataset
    X_train_xgb = pd.DataFrame(X_smo, columns=col_lst)
    y_train_xgb = pd.Series(y_smo)

    # reload trained model
    xgb1 = pickle.load(open(path + "xgb_model.json", "rb"))
    LR_model = pickle.load(open(path + "LR_model.pkl", "rb"))
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(path + "catboost_model.json")
    lgb_m = pickle.load(open(path + 'lgb_model.pkl', "rb"))
    lgb_m = joblib.load('lgb_model.pkl')

    # feature engineering
    res_train = xgb1.predict_proba(X_train_xgb)
    res_test = xgb1.predict_proba(X_test)
    res_train_lr = LR_model.predict_proba(X_smo)
    res_test_lr = LR_model.predict_proba(X_test)
    res_train_cat = catboost_model.predict_proba(X_train_xgb)
    res_test_cat = catboost_model.predict_proba(X_test)
    temp1 = lgb_m.predict(X_train_xgb)
    res_train_lgb = np.select(condlist=[temp1 > 0.5], choicelist=[1], default=0)
    temp2 = lgb_m.predict(X_test)
    res_test_lgb = np.select(condlist=[temp2 > 0.5], choicelist=[1], default=0)

    res_train_df = pd.DataFrame(res_train, columns=["class0xgb", "class1xgb"])
    res_test_df = pd.DataFrame(res_test, columns=["class0xgb", "class1xgb"])
    res_train_lr_df = pd.DataFrame(res_train_lr, columns=["class0lr", "class1lr"])
    res_test_lr_df = pd.DataFrame(res_test_lr, columns=["class0lr", "class1lr"])
    res_train_cat_df = pd.DataFrame(res_train_cat, columns=["class0cat", "class1cat"])
    res_test_cat_df = pd.DataFrame(res_test_cat, columns=["class0cat", "class1cat"])
    res_train_lgb_df = pd.DataFrame(res_train_lgb, columns=["class1lgb"])
    res_test_lgb_df = pd.DataFrame(res_test_lgb, columns=["class1lgb"])

    X_train_drop_f = pd.concat([X_train_xgb, res_train_df, res_train_lr_df,
                                res_train_cat_df, res_train_lgb_df], axis=1)
    X_test_drop_f = pd.concat([X_test.reset_index(), res_test_df.reset_index(), res_test_lr_df,
                               res_test_cat_df, res_test_lgb_df], axis=1).drop(columns=["index"])
    # stack cat
    catboost_model_2 = CatBoostClassifier(iterations=500, max_ctr_complexity=3,
                                          learning_rate=0.03, depth=9, random_seed=123,
                                          od_type='Iter', loss_function="Logloss",
                                          eval_metric='Logloss', od_wait=25, verbose=0)
    catboost_model_2.fit(X_train_drop_f, y_train_xgb, eval_set=(X_test_drop_f, y_test))
    ##predict
    pred_catboost2 = catboost_model_2.predict(X_test_drop_f)
    pred_prob_catboost2 = catboost_model_2.predict_proba(X_test_drop_f)

    # save stacking model
    catboost_model_2.save_model(path + "catboost_model_stacking.json")

    # finalprecision
    target_names = ['class 0', 'class 1']
    report = classification_report(y_test, pred_catboost2, target_names=target_names)
    return pred_catboost2, pred_prob_catboost2, report


final_all = pd.read_csv("/Users/zhangjiawei/Desktop/T1/FTEC5510/final_all.csv")
X_smo, X_test, y_smo, y_test, col_lst = feature_engineering_train_test_split(final_all)
path = "/Users/zhangjiawei/Desktop/T1/FTEC5510/code/"

y_pred_lr, y_prob_lr = LR_model_train_save(path, X_smo, y_smo, X_test, y_test)
res_prob, res_xgb, importance_xgb_lst = XGB_classifier_train_save(path, X_smo, y_smo, X_test, y_test, col_lst)
pred_catboost, pred_cat_prob, imp_cat = Catboost_classifier_train_save(path, X_smo, y_smo, X_test, y_test, col_lst)
y_pred_lgb, y_pred_dummy_lgb, imp_lgb = LightGBM_classifier_train_save(path, X_smo, y_smo, X_test, y_test, col_lst)
pred_catboost2, pred_prob_catboost2, report = Stacking_model_train_valid_save(path, X_smo, y_smo, X_test, y_test,
                                                                              col_lst)
