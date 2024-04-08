# https://www.wuzao.com/document/pyportfolioopt/ExpectedReturns.html
# https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/4-Black-Litterman-Allocation.ipynb
import uuid

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
import sys, os

from utils.log_util import get_logger
logger = get_logger(__name__)

from utils.stock_zh_a_util import get_stock_data, get_benchmark_data, get_stock_position


class MeanVarianceOptimizer:
    def __init__(self, ds, start_date):
        self.ds = ds
        self.start_date = start_date

    def get_stock_data(self):
        logger.info("get stock data")
        symbols = get_stock_position()
        df_list = []
        for stock in symbols:
            df = get_stock_data(stock, ds=self.ds, start_date=self.start_date, yf_compatible=True)
            levels = [df.index, [stock]*len(df)]
            df.index = pd.MultiIndex.from_arrays(levels, names=["date", "ticker"])
            df_list.append(df)
        data = pd.concat(df_list)
        data = data.unstack()
        return data

    def get_market_data(self):
        logger.info("get benchmark data")
        symbol = "sh000300"
        df = get_benchmark_data(symbol="sh000300", ds=self.ds, start_date=self.start_date,  yf_compatible=True)
        levels = [df.index, [symbol]*len(df)]
        df.index = pd.MultiIndex.from_arrays(levels, names=["date", "benchmark"])
        data = df.unstack()
        return data



    def opt(self):
        from pypfopt import risk_models
        from pypfopt import EfficientFrontier
        from pypfopt import objective_functions
        from pypfopt import expected_returns

        run_id = str(uuid.uuid1())
        data = self.get_stock_data()
        market_data = self.get_market_data()
        close = data["close"]
        market_close = market_data["close"]

        df_list = []

        sample_cov = risk_models.sample_cov(close, frequency=252)
        mu = expected_returns.capm_return(prices=close, market_prices=market_close)
        S = risk_models.CovarianceShrinkage(close).ledoit_wolf()
        ef = EfficientFrontier(None, S, weight_bounds=(None, None))
        ef.min_volatility()
        df = self.get_metrics_output(close, ef, return_model="capm_return", risk_model="covariance_shrinkage",
                                     optimizer="efficient_frontier", objective="min_volatility", run_id=run_id)
        df_list.append(df)


        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # gamma is the tuning parameter
        ef.efficient_risk(target_volatility=0.15)
        df = self.get_metrics_output(close, ef, return_model="capm_return", risk_model="covariance_shrinkage",
                                     optimizer="efficient_frontier", objective="efficient_risk_l2_reg", run_id=run_id)
        df_list.append(df)


        semicov = risk_models.semicovariance(close, benchmark=0)
        ef = EfficientFrontier(mu, semicov)
        ef.min_volatility()
        df = self.get_metrics_output(close, ef, return_model="capm_return", risk_model="semicov",
                                     optimizer="efficient_frontier", objective="min_volatility", run_id=run_id)
        df_list.append(df)


        logger.info("construct the portfolio with the minimum CVaR")
        from pypfopt import EfficientCVaR
        returns = expected_returns.returns_from_prices(close).dropna()
        ec = EfficientCVaR(mu, returns)
        ec.min_cvar()
        df = self.get_metrics_output(close, ef, return_model="capm_return", risk_model="returns_from_prices",
                                     optimizer="efficient_cvar", objective="min_cvar", run_id=run_id)
        df_list.append(df)



        ec = EfficientCVaR(mu, returns)
        ec.efficient_risk(target_cvar=0.025)
        df = self.get_metrics_output(close, ef, return_model="capm_return", risk_model="returns_from_prices",
                                     optimizer="efficient_cvar", objective="efficient_risk", run_id=run_id)
        df_list.append(df)



        from pypfopt import CLA
        cla = CLA(mu, S)
        cla.min_volatility()
        df = self.get_metrics_output(close, ef, return_model="capm_return", risk_model="covariance_shrinkage",
                                     optimizer="cla", objective="min_volatility", run_id=run_id)
        df_list.append(df)
        df_all = pd.concat(df_list)
        print(df_all)
        return df_all

        #
        # from pypfopt import plotting
        # import matplotlib.pyplot as plt

        # # sample_cov
        # close.plot(figsize=(15,10))
        # # plt.show()
        # plotting.plot_covariance(sample_cov, plot_correlation=True)
        # # plt.show()



        # plotting.plot_covariance(S, plot_correlation=True)
        # plt.show()

        # You don't have to provide expected returns in this case
        # ef = EfficientFrontier(None, S, weight_bounds=(None, None))
        # ef.min_volatility()
        # weights = ef.clean_weights()
        # print(weights)
        # pd.Series(weights).plot.barh()
        # plt.show()

        # ef.portfolio_performance(verbose=True)

        # ef.max_sharpe()
        # weights = ef.clean_weights()
        # pd.Series(weights).plot.pie(figsize=(10, 10))
        # plt.show()


        #
        # ef = EfficientFrontier(mu, S)
        # ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # gamma is the tuning parameter
        #
        # ef.efficient_risk(target_volatility=0.15)
        # weights = ef.clean_weights()
        # print(weights)
        # pd.Series(weights).plot.barh()
        # # plt.show()
        # ef.portfolio_performance(verbose=True)

        #
        # semicov = risk_models.semicovariance(close, benchmark=0)
        # plt.figure()
        # plotting.plot_covariance(semicov)
        # ef = EfficientFrontier(mu, semicov)
        # ef.min_volatility()
        # weights = ef.clean_weights()
        # print(weights)
        # ef.portfolio_performance(verbose=True)

        #
        # logger.info("compute var and cvar")
        # returns = expected_returns.returns_from_prices(close).dropna()
        # ef = EfficientFrontier(mu, S)
        # ef.min_volatility()
        # weight_arr = ef.weights
        # # Compute CVaR
        # portfolio_rets = (returns * weight_arr).sum(axis=1)
        # plt.figure()
        # portfolio_rets.hist(bins=50)
        #
        #
        # # VaR
        # var = portfolio_rets.quantile(0.05)
        # cvar = portfolio_rets[portfolio_rets <= var].mean()
        # print("VaR: {:.2f}%".format(100 * var))
        # print("CVaR: {:.2f}%".format(100 * cvar))

        # construct the portfolio with the minimum CVaR
        # logger.info("construct the portfolio with the minimum CVaR")
        # from pypfopt import EfficientCVaR
        #
        # ec = EfficientCVaR(mu, returns)
        # ec.min_cvar()
        # ec.portfolio_performance(verbose=True)
        #
        # from pypfopt import EfficientCVaR
        #
        # ec = EfficientCVaR(mu, returns)
        # ec.efficient_risk(target_cvar=0.025)
        # ec.portfolio_performance(verbose=True)
        #
        #
        # from pypfopt import CLA, plotting
        #
        # cla = CLA(mu, S)
        # cla.min_volatility()
        # cla.portfolio_performance(verbose=True)
        # plt.figure()
        # ax = plotting.plot_efficient_frontier(cla, showfig=False)
        #
        # # complex plots
        # n_samples = 10000
        # w = np.random.dirichlet(np.ones(len(mu)), n_samples)
        # rets = w.dot(mu)
        # stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
        # sharpes = rets / stds
        #
        # print("Sample portfolio returns:", rets)
        # print("Sample portfolio volatilities:", stds)
        # # Plot efficient frontier with Monte Carlo sim
        # ef = EfficientFrontier(mu, S)
        #
        # fig, ax = plt.subplots()
        # plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
        #
        # # Find and plot the tangency portfolio
        # ef2 = EfficientFrontier(mu, S)
        # ef2.min_volatility()
        # ret_tangent, std_tangent, _ = ef2.portfolio_performance()
        #
        # # Plot random portfolios
        # ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
        #
        # # Format
        # ax.set_title("Efficient Frontier with random portfolios")
        # ax.legend()
        # plt.tight_layout()
        #
        # plt.show()

    def get_metrics_output(self, close, ef, return_model, risk_model, optimizer, objective, run_id):
        from pypfopt import expected_returns
        import json

        weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=True)
        returns = expected_returns.returns_from_prices(close).dropna()
        weight_arr = ef.weights
        portfolio_rets = (returns * weight_arr).sum(axis=1)
        var = portfolio_rets.quantile(0.05)
        cvar = portfolio_rets[portfolio_rets <= var].mean()
        print(f"weight {weights}, perf {perf}, var {var}, cvar {cvar}, portfolio_rets {portfolio_rets}")
        df = pd.DataFrame(data={"weights": json.dumps(weights),
                                "expected_annual_return": perf[0],
                                "annual_volatility": perf[1],
                                "sharpe_ratio": perf[2],
                                "var": var,
                                "cvar": cvar,
                                "return_model": return_model,
                                "risk_model": risk_model,
                                "optimizer": optimizer,
                                "objective": objective,
                                "run_id": run_id
                                }, index=[0])
        return df


if __name__ == '__main__':
    # ds = sys.argv[1]
    # logger.info("execute task on ds {}".format(ds))
    # if not is_trade_date(ds):
    #     logger.info(f"{ds} is not trade date. task exits.")
    #     exit(os.EX_OK)
    pd.set_option('display.max_columns', None)

    ds = "20240401"

    start_date = (datetime.strptime(ds, '%Y%m%d') - timedelta(days=712)).strftime("%Y-%m-%d")

    opt = MeanVarianceOptimizer(ds, start_date)
    opt.opt()