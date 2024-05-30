
import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from sklearn.metrics import mean_squared_error
from utils import quant_data_util
import pickle
from lightgbm import LGBMRegressor
import lightgbm as lgb
import numpy as np

import math
from datetime import timedelta

import os

import matplotlib.pyplot as plt
from utils.log_util import get_logger
from utils.stock_zh_a_util import is_trade_date

logger = get_logger(__name__)

import optuna

#
# params_lgb = {
#     "task": "train",
#     "boosting_type": "gbdt",
#     "objective": "regression",
#     # 'subsample': 0.95,
#     'learning_rate': 0.01,
#     "max_depth": 3,
#     "min_data_in_leaf": 10,
#     # "num_leaves": 16,
#     # "feature_fraction": 0.5,
#     # "bagging_fraction": 0.5,
#     # "min_child_samples": 20,
#     # "reg_alpha": 1,
#     # "reg_lambda": 10,
#     # "max_bin": 20,
#     # "min_data_per_group": 100,
#     # "bagging_freq": 1,
#     # "cat_smooth": 10,
#     # "cat_l2": 10,
#     "verbosity": -1,
#     # 'random_state': 42,
#     'n_estimators': 2000,
#     # 'colsample_bytree': 1.0
# }

# num_iterations 🔗︎, default = 100, type = int, aliases: num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, nrounds, num_boost_round, n_estimators, max_iter, constraints: num_iterations >= 0


# These global variables are for auto_tune, whose objective function doesnot accept arguments.
train_x, train_y, test_x, test_y, test_ext, pred_x, pred_ext = None, None, None, None, None, None, None
feature_names, numerical_feature_names, categorical_feature_names = None, None, None
boosting_type = None
params_file_name = None


def get_feature_names(use_categorical_features):
    numerical_features = ["ma_5", "ma_10", "ma_20", "ma_60", "ma_120", "ma_240", "hhv_5", "hhv_10", "hhv_20",
                          "hhv_60", "hhv_120", "hhv_240", "llv_5", "llv_10", "llv_20", "llv_60", "llv_120",
                          "llv_240", "bias_6", "bias_12", "bias_24", "boll_upper_20", "boll_mid_20",
                          "boll_lower_20", "rsi_6", "rsi_12", "rsi_24", "wr_10", "wr_6", "mtm_12", "mtm_12_ma_6",
                          "k_9", "d_9", "j_9", "macd_dif", "macd_dea", "macd", "dmi_pdi", "dmi_mdi", "dmi_adx",
                          "dmi_adxr", "obv", "cci", "roc_12", "ma_6_roc_12", "bbi", "expma_12", "expma_50", "ar",
                          "br", "atr", "dma_dif", "dma", "emv", "maemv", "psy", "psyma", "asi", "asit", "mfi",
                          "mass", "mamass", "dpo", "madpo", "vr", "trix", "trma", "kc_upper", "kc_mid",
                          "kc_lower", "dc_upper", "dc_mid", "dc_lower", "open", "close", "high", "low", "roc",
                          "volume", "turnover_rate", "roc_m1d", "roc_m2d", "roc_m3d"]
    if use_categorical_features:
        categorical_features = ["sym", "date_mon", "date_dow", "industry", "is_index_000001", "is_index_000016",
                                "is_index_000300"]
    else:
        categorical_features = []

    features = []
    features.extend(numerical_features)
    features.extend(categorical_features)
    logger.info("numerical_features {}, categorical features {}".format(numerical_features, categorical_features))
    return features, numerical_features, categorical_features


def get_model_name(args):
    return "lightgbm_{}_cat_{}".format(args.boosting_type, args.use_categorical_features)


def get_categorical_feature_map(df, feature_names):
    if feature_names is None or len(feature_names) == 0:
        return None
    categorical_feature_map = dict()
    for name in feature_names:
        tmp_dict = dict()
        for i, val in enumerate(np.sort(df[name].unique())):
            tmp_dict[val] = i
        categorical_feature_map[name] = tmp_dict
    return categorical_feature_map


def convert_categorical_variables_to_integers(df, feature_map):
    if feature_map is None:
        return df

    for name in feature_map.keys():
        tmp_df = df[name]
        tmp_dict = feature_map[name]
        unknown_val_set = set(tmp_df.unique()).difference(tmp_dict.keys())
        for k, v in tmp_dict.items():
            df.loc[tmp_df == k, name] = v
        for val in unknown_val_set:
            df.loc[tmp_df == val, name] = -1
    return df


def load_params():
    if os.path.exists(params_file_name):
        logger.info("load params from file {}".format(params_file_name))
        params = pickle.load(open(params_file_name, "rb"))
    else:
        logger.info("specified file {} does not exist, using default params".format(params_file_name))
        params = {"metric": "mse", "verbosity": -1,
                  'learning_rate': 0.05814374665204413,
                  'max_depth': 12,
                  'lambda_l1': 0.2788050348029096,
                  'lambda_l2': 8.486058444555043e-07,
                  'num_leaves': 464,
                  'feature_fraction': 0.8882445798293603,
                  'bagging_fraction': 0.9529766897517215,
                  'bagging_freq': 4,
                  'min_child_samples': 53,
                  'max_bin': 4005,
                  'min_data_in_leaf': 45,
                  'num_iterations': 285,
                  "num_threads": 8
                  }
    params["boosting_type"] = boosting_type
    params["objective"] = "regression"
    # params["objective"] = "huber"
    params["verbosity"] = -1
    return params


def load_data(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def get_most_important_features(fi, fn):
    f = np.array([fi, fn])
    if f.shape[0] == 2:
        f = np.transpose(f)
    return np.flipud(f[f[:, 0].astype("int").argsort()])


def prepare_data(train_file_name, pred_file_name, use_roc_label, use_log_close, use_sqrt_roc, numerical_features,
                 categorical_features, label_name=""):

    if label_name is None or len(label_name) == 0:
        label_name = "label_roc" if use_roc_label else "label_close"

    dataset = quant_data_util.load_data(train_file_name, pred_file_name, categorical_column_names=categorical_features,
                                        label_column_names=[label_name])
    train_data, test_data, pred_data, scaler, encoder, enc_column_names = dataset


    ext_column_names = ["日期", "代码", "roc", "close", "label_roc", "label_close", "label_roi_1d", "label_roi_2d",
                        "label_roi_3d", "label_roi_5d", "label_roi_10d"]

    train_data = train_data[train_data["日期"] > train_data["日期"].max() - timedelta(days=730)]
    train_x, train_y, feature_map = get_x_and_y(train_data, label_name, numerical_features, categorical_features)
    test_x, test_y, _ = get_x_and_y(test_data, label_name, numerical_features, categorical_features, feature_map)
    test_ext = test_data.loc[:, ext_column_names]

    pred_data = pred_data[pred_data["日期"] == pred_data["日期"].max()]
    pred_x, pred_y, _ = get_x_and_y(pred_data, label_name=[], numerical_features=numerical_features,
                                    categorical_features=categorical_features, feature_map=feature_map)
    pred_ext = pred_data.loc[:, ext_column_names]

    if use_log_close:
        train_y = np.array([math.log(y) if y > 0 else 0 for y in train_y])
        test_y = np.array([math.log(y) if y > 0 else 0 for y in test_y])

    if use_sqrt_roc:
        train_y = np.array([math.sqrt(y) if y > 0 else -math.sqrt(-y) for y in train_y])
        test_y = np.array([math.sqrt(y) if y > 0 else -math.sqrt(-y) for y in test_y])

    logger.info("data loaded")
    logger.info("numerical features {}, categorical features {}".format(numerical_features, categorical_features))
    logger.info("label {}, ext columns {}".format(label_name, ext_column_names))
    logger.info("train data min date {}, max date {}".format(train_data["日期"].min(), train_data["日期"].max()))
    logger.info("test data min date {}, max date {}".format(test_data["日期"].min(), test_data["日期"].max()))
    logger.info("pred data min date {}, max date {}".format(pred_data["日期"].min(), pred_data["日期"].max()))

    logger.info("train x shape {}, train y shape {}, train y mean {}, variance {}".format(train_x.shape, train_y.shape,
                                                                                          train_y.mean(),
                                                                                          train_y.var()))
    logger.info("test x shape {}, test y shape {}, test y mean {}, variance {}".format(test_x.shape, test_y.shape,
                                                                                       test_y.mean(), test_y.var()))
    logger.info("pred x shape {}, append x shape {}".format(pred_x.shape, pred_ext.shape))

    return train_x, train_y, test_x, test_y, test_ext, pred_x, pred_ext


def get_x_and_y(data, label_name, numerical_features, categorical_features, feature_map=None):
    if feature_map is None:
        feature_map = get_categorical_feature_map(data, categorical_features)
    data = convert_categorical_variables_to_integers(data, feature_map)
    x_num = data.loc[:, numerical_features].fillna(0).values.astype(float)
    x_cat = data.loc[:, categorical_features].values.astype(int)
    x = np.concatenate((x_num, x_cat), axis=1, dtype="object")
    y = data.loc[:, label_name].values.astype(float)
    return x, y, feature_map


def train(args):
    use_roc_label = args.use_roc_label
    model_name = get_model_name(args)
    run_id = quant_data_util.get_run_id(prefix=model_name)

    params_lgb = load_params()
    lgbm = LGBMRegressor(**params_lgb)
    logger.info("lgbm model created, model {}".format(lgbm))
    print("training started")

    lgbm.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], feature_name=feature_names,
             categorical_feature=categorical_feature_names)
    # print("evals result {}".format(lgbm.evals_result_))
    logger.info("feature importance {}".format(lgbm.feature_importances_))

    f = get_most_important_features(lgbm.feature_importances_, feature_names)
    logger.info("sorted features with importance \n {} \n without importance \n ['{}']".format(f, "', '".join(f[:, 1])))

    lgb.plot_metric(lgbm)
    plt.savefig('metric_{}.png'.format(run_id))
    lgb.plot_importance(lgbm)
    plt.savefig('importance_{}.png'.format(run_id))
    print("evaluation started")
    pred = lgbm.predict(test_x)
    mse = mean_squared_error(test_y, pred)
    print("mse {}, rmse {}".format(mse, math.sqrt(mse)))
    # quant_data_util.fill_ext_with_predictions(test_ext, pred, use_roc_label)
    # quant_data_util.compute_precision_recall(test_ext)

    label_name = args.label_name
    label_name_pred = "{}_pred".format(label_name)
    test_ext[label_name_pred] = pred
    label = test_ext[label_name].values.astype(float)
    quant_data_util.compute_precision_recall_updated(label, pred)

    ds = args.ds
    save_prediction_result = args.save_prediction_result
    if save_prediction_result:
        logger.info("saving validation results")
        quant_data_util.save_prediction_to_db(test_ext, ds=ds, model_name=model_name, stage="validation",
                                              pred_name=label_name, pred_val_col_name=label_name_pred,
                                              label_col_name=label_name,
                                              run_id=run_id)

    # #
    print("now make predictions")
    pred = lgbm.predict(pred_x)
    # quant_data_util.fill_ext_with_predictions(pred_ext, pred, use_roc_label)
    pred_ext[label_name_pred] = pred
    print("now find the ups")
    pred_ext.sort_values(by=[label_name_pred], inplace=True, ascending=False)
    print(pred_ext.head(20).to_string())
    # #
    if args.save_prediction_result:
        quant_data_util.save_prediction_to_db(pred_ext, ds=ds, model_name=model_name, stage="prediction",
                                              pred_name=label_name, pred_val_col_name=label_name_pred,
                                              label_col_name=label_name,
                                              run_id=run_id)

    return 0


def train_optuna(trial):
    param = {
        "task": "train",
        "boosting_type": boosting_type,
        "objective": "regression",
        # "objective": "huber",
        # "alpha": 5.0,
        "verbosity": -1,
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 512),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 14),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 1000),
        "max_bin": trial.suggest_int("max_bin", 5, 5000),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 10000, log=True),
        "num_iterations": trial.suggest_int("num_iterations", 50, 200),
        "num_threads": 8
    }

    dtrain = lgb.Dataset(train_x, label=train_y, feature_name=feature_names,
                         categorical_feature=categorical_feature_names)
    dvalid = lgb.Dataset(test_x, label=test_y, feature_name=feature_names,
                         categorical_feature=categorical_feature_names)

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l2")
    # pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "huber")

    lgbm = lgb.train(param, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])
    # make predictions for test data
    print("evaluation started")
    pred = lgbm.predict(test_x)
    mse = mean_squared_error(test_y, pred)
    print("mse {}, rmse {}".format(mse, math.sqrt(mse)))
    return mse


def auto_tune():
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize"
    )
    study.optimize(train_optuna, n_trials=30)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    '{}': {},".format(key, value))

    pickle.dump(trial.params, open(params_file_name, "wb"))
    logger.info("best params saved to {}".format(params_file_name))


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto_tune', type=int, default=0, help='auto tune 0 no, 1 yes, default 0')
    parser.add_argument('--ds', type=str, default="", help='biz date')
    parser.add_argument('--load_from_db', type=int, default=0, help='load_from_db, 0 no, 1 yes, default 0')
    parser.add_argument('--data_file_name', type=str, default="quant_reg_data.pkl",
                        help='data_file_name, default quant_reg_data.pkl')

    parser.add_argument('--boosting_type', type=str, default="gbdt", help='lgbm boosting type, gbdt, rf')

    parser.add_argument('--use_roc_label', type=int, default=1, help='use_roc_label 0 no, 1 yes, default 1')
    parser.add_argument('--use_log_close', type=int, default=0, help='use log for close price, 0 no, 1 yes, default 0')
    parser.add_argument('--use_sqrt_roc', type=int, default=0, help='use sqrt for roc, 0 no, 1 yes, default 0')
    parser.add_argument('--use_categorical_features', type=int, default=1,
                        help='use_categorical_features, 0 no, 1 yes, default 1')

    parser.add_argument('--label_name', type=str, default="label_roi_5d",
                        help='if used, use_roc_label,use_sqrt_roc,use_log_close will be disabled.')

    parser.add_argument('--save_prediction_result', type=int, default=1,
                        help='save prediction result to db, 0 no, 1 yes, default 1')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    logger.info("execute task on ds {}".format(args.ds))
    logger.info("arguments {}".format(args))

    if not is_trade_date(args.ds):
        logger.info(f"{args.ds} is not trade date. task exits.")
        exit(os.EX_OK)

    # set global variables
    feature_names, numerical_feature_names, categorical_feature_names = get_feature_names(args.use_categorical_features)

    train_file_name, pred_file_name = quant_data_util.get_train_pred_file_names(args.ds)

    train_x, train_y, test_x, test_y, test_ext, pred_x, pred_ext = prepare_data(train_file_name, pred_file_name,
                                                                                args.use_roc_label, args.use_log_close,
                                                                                args.use_sqrt_roc,
                                                                                numerical_feature_names,
                                                                                categorical_feature_names,
                                                                                args.label_name)
    boosting_type = args.boosting_type
    params_file_name = "params_{}.pkl".format(get_model_name(args))
    logger.info("params file name {}".format(params_file_name))

    if args.auto_tune == 1:
        logger.info("start auto tuning.")
        auto_tune()
    logger.info("auto tune finished, train model using best parameters.")
    train(args)
