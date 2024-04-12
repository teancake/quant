import numpy as np
import pandas as pd
import talib as ta
import MyTT as tt


def get_ta_indicator_map(data_df: pd.DataFrame):
    data_df.sort_values(by="日期", ascending=True, inplace=True)
    open_ = data_df['开盘'].fillna(0).to_numpy()
    close = data_df['收盘'].fillna(0).to_numpy()
    high = data_df['最高'].fillna(0).to_numpy()
    low = data_df['最低'].fillna(0).to_numpy()
    vol = data_df['成交量'].fillna(0).to_numpy()

    ti_map = {}
    ti_map["ma_5"] = tt.MA(close, 5)
    ti_map["ma_10"] = tt.MA(close, 10)
    ti_map["ma_20"] = tt.MA(close, 20)
    ti_map["ma_60"] = tt.MA(close, 60)
    ti_map["ma_120"] = tt.MA(close, 120)
    ti_map["ma_240"] = tt.MA(close, 240)

    ti_map["hhv_5"] = tt.HHV(close, 5)
    ti_map["hhv_10"] = tt.HHV(close, 10)
    ti_map["hhv_20"] = tt.HHV(close, 20)
    ti_map["hhv_60"] = tt.HHV(close, 60)
    ti_map["hhv_120"] = tt.HHV(close, 120)
    ti_map["hhv_240"] = tt.HHV(close, 240)

    ti_map["llv_5"] = tt.LLV(close, 5)
    ti_map["llv_10"] = tt.LLV(close, 10)
    ti_map["llv_20"] = tt.LLV(close, 20)
    ti_map["llv_60"] = tt.LLV(close, 60)
    ti_map["llv_120"] = tt.LLV(close, 120)
    ti_map["llv_240"] = tt.LLV(close, 240)

    ti_map["bias_6"], ti_map["bias_12"], ti_map["bias_24"] = tt.BIAS(close, 6, 12, 24)
    ti_map["boll_upper_20"], ti_map["boll_mid_20"], ti_map["boll_lower_20"] = tt.BOLL(close, N=20, P=2)
    ti_map["rsi_6"] = tt.RSI(close, N=6)
    ti_map["rsi_12"] = tt.RSI(close, N=12)
    ti_map["rsi_24"] = tt.RSI(close, N=24)
    ti_map['wr_10'], ti_map['wr_6'] = tt.WR(close, high, low, N=10, N1=6)
    ti_map["mtm_12"], ti_map["mtm_12_ma_6"] = tt.MTM(close, N=12, M=6)
    ti_map["k_9"], ti_map["d_9"], ti_map["j_9"] = tt.KDJ(close, high, low, N=9, M1=3, M2=3)
    ti_map["macd_dif"], ti_map["macd_dea"], ti_map["macd"] = tt.MACD(close, SHORT=12, LONG=26, M=9)
    ti_map["dmi_pdi"], ti_map["dmi_mdi"], ti_map["dmi_adx"], ti_map["dmi_adxr"] = tt.DMI(close, high, low, M1=14, M2=6)
    ti_map["obv"] = tt.OBV(close, vol)
    ti_map["cci"] = tt.CCI(close, high, low, N=14)
    ti_map["roc_12"], ti_map["ma_6_roc_12"] = tt.ROC(close, N=12, M=6)
    ti_map["bbi"] = tt.BBI(close, M1=3, M2=6, M3=12, M4=24)
    ti_map["expma_12"], ti_map["expma_50"] = tt.EXPMA(close, N1=12, N2=50)
    # print("expma_12 {}, expma_50 {}".format(len(ti_map["expma_12"]), len(ti_map["expma_50"])))


    # BRAR
    ti_map["ar"], ti_map["br"] = tt.BRAR(open_, close, high, low, M1=26)
    # ATR - 真实波动N日平均值
    ti_map["atr"] = tt.ATR(close, high, low, N=20)
    # DMA
    ti_map["dma_dif"], ti_map["dma"] = tt.DFMA(close, N1=10, N2=50, M=10)
    # EMV
    ti_map["emv"], ti_map["maemv"] = tt.EMV(high, low, vol, N=14, M=9)
    # PSY
    ti_map["psy"], ti_map["psyma"] = tt.PSY(close, N=12, M=6)
    # ASI
    ti_map["asi"], ti_map["asit"] = tt.ASI(open_, close, high, low, M1=26, M2=10)
    # MFI
    ti_map["mfi"] = tt.MFI(close, high, low, vol, N=14)
    # MASS
    ti_map["mass"], ti_map["mamass"] = tt.MASS(high, low, N1=9, N2=25, M=6)
    # DPO
    ti_map["dpo"], ti_map["madpo"] = tt.DPO(close, M1=20, M2=10, M3=6)
    # # CR 能量指标
    # ti_map["cr"] = tt.CR(close, high, low, N=20)
    # VR
    ti_map["vr"] = tt.VR(close, vol, M1=26)
    #TRIX
    ti_map["trix"], ti_map["trma"] = tt.TRIX(close, M1=12, M2=20)
    # 肯特纳通道（Keltner Channel，KC）肯特纳通道（KC）是一个移动平均通道
    ti_map["kc_upper"], ti_map["kc_mid"], ti_map["kc_lower"], = tt.KTN(close, high, low, N=20, M=10)
    # 唐奇安通道（DC）指标是由著名商品交易员Richard Donchian创建的。Donchian被称为“趋势交易之父”。
    ti_map["dc_upper"], ti_map["dc_mid"], ti_map["dc_lower"], = tt.TAQ(high, low, N=20)

    ti_df = pd.DataFrame(ti_map)
    ti_df.insert(0, "日期", data_df["日期"])
    return ti_df
