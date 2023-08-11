import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
# from quant_data_util import load_data_from_db
import numpy as np
import pickle
import sys
import lightgbm as lgb
import math

params_lgb = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    # 'subsample': 0.95,
    'learning_rate': 0.001,
    # "max_depth": 3,
    # "feature_fraction": 0.8,
    # "bagging_fraction": 0.8,
    # "min_child_samples": 20,
    # "reg_alpha": 1,
    # "reg_lambda": 10,
    # "max_bin": 255,
    # "min_data_per_group": 100,
    # "bagging_freq": 1,
    # "cat_smooth": 10,
    # "cat_l2": 10,
    # "verbosity": 1,
    # 'random_state': 42,
    'n_estimators': 2000,
    # 'colsample_bytree': 1.0
}


def load_data(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    return data



def train_xgboost():
    random_seed = 10
    data_file_name = "quant_reg_data.pkl"
    use_offline_data = True

    if use_offline_data:
        print("loading data from local files.")
        dataset = load_data(data_file_name)
#    else:
#        print("loading data from database.")
#        dataset = load_data_from_db()
#        pickle.dump(dataset, open(data_file_name, 'wb'))
#        print("data saved to {}".format(data_file_name))

    train_x, train_y, test_x, test_y, pred_x, append_x = dataset
    print("data loaded")
    print("train x shape {}, train y shape {}, train y mean {}, variance {}".format(train_x.shape, train_y.shape, train_y.mean(), train_y.var()))
    print("test x shape {}, test y shape {}, test y mean {}, variance {}".format(test_x.shape, test_y.shape, train_y.mean(), test_y.var()))
    print("pred x shape {}, append x shape {}".format(pred_x.shape, append_x.shape))


    # ## train lgbm
    lgbm = lgb.LGBMRegressor(**params_lgb)
    print("lgbm model created, model {}".format(lgbm))
    print("training started")
    lgbm.fit(train_x, train_y)
    # make predictions for test data
    print("evaluation started")
    pred = lgbm.predict(test_x)

    mse = mean_squared_error(test_y, pred)
    print("mse {}, rmse {}".format(mse, math.sqrt(mse)))

    print("now make predictions")
    pred = lgbm.predict(pred_x)
    append_x["score"] = pred
    print("now find the ups")
    append_x.sort_values(by=["score"], inplace=True)
    print(append_x.tail(20).to_string())

    #
    #
    # model = XGBRegressor(n_estimators=1000,  learning_rate=0.1)
    # print("xgboost model created, model {}".format(model))
    # print("training started")
    # model.fit(train_x, train_y)
    # # make predictions for test data
    # print("evaluation started")
    # pred = model.predict(test_x)
    #
    # mse = mean_squared_error(test_y, pred)
    #
    # print("rmse {}".format(math.sqrt(mse)))
    #
    # print("now make predictions")
    # pred = model.predict(pred_x)
    # append_x["score"] = pred
    # print("now find the ups")
    # append_x.sort_values(by=["score"], inplace=True)
    # # print(pred[pred > 0.5])
    # # print(append_x[append_x.score > 0.01].to_string())
    # print(append_x.tail(20).to_string())
    #
    #




if __name__ == '__main__':
    train_xgboost()
