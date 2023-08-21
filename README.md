# A股量化交易学习
## 数据
### 原始数据：
通过akshare能拿到每只股票的所有历史行情数据（名称、日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率），分三个时间粒度统计（天、周、月），复权方式有3种，量化交易常采用后复权，数据总量在1000多万。 


从akshare也可以拿到实时数据、指数数据等，目前暂且未使用。
还有一些非结构数据，如新闻、消息等也可从akshare拿到，目前也未使用。

### 技术指标数据：
从akshare无法直接拿到技术指标数据，但是可以从开盘、收盘、最高、最低数据计算，
myTT和talib都行，myTT更方便一些，目前使用myTT提取了大约30个技术指标，共约80个值。 
https://github.com/mpquant/MyTT/blob/main/README.md
https://github.com/HuaRongSAO/talib-document 

关于技术指标的说明和使用，可以参考 
https://www.joinquant.com/help/api/help#name:technicalanalysis 


## 模型
由于A股是T+1的交易，即当天买的股票，最早只能第二天卖出，所以模型暂且都只做以天为粒度的预测。



### 标签：
如果是分类模型，比如预测某支股票第二天的涨跌，可以把第二天的涨跌幅做处理，大于3%的记1，小于3%的记0。
如果是回归模型，可以把第二天的涨跌幅做为标签，也可以直接拿收盘价做标签做股价预测。

上面第二天可以改为：第3天，或者未来5天以内，如果是未来几天以内，就不能用那天的涨跌幅直接做标签，需要计算那天的收益，即把中间几天的涨跌都算进去。


### 特征：
原始特征就是每支股票每天的开盘、收盘等数据，衍生特征目前只使用了myTT计算出来的技术指标。
#### 衍生特征
##### 收盘价的N日简单移动平均值
ma_5
ma_10
ma_20
ma_60
ma_120
ma_240

##### 最近N天收盘最高价
hhv_5
hhv_10
hhv_20
hhv_60
hhv_120
hhv_240

##### 最近N天最低收盘价 
llv_5
llv_10
llv_20
llv_60
llv_120
llv_240

##### BIAS乖离率
bias_6
bias_12
bias_24

##### BOLL指标，布林带  
boll_upper_20
boll_mid_20
boll_lower_20

##### RSI指标 
rsi_6
rsi_12
rsi_24

##### W&R威廉指标 
wr_10
wr_6

##### 动量指标 
mtm_12
mtm_12_ma_6

##### KDJ指标 
k_9
d_9
j_9

##### MACD指标
macd_dif
macd_dea
macd

##### 动向指标 
dmi_pdi
dmi_mdi
dmi_adx
dmi_adxr

##### 能量潮指标 
obv

##### CCI指标
cci

##### 变动率指标 
roc_12
ma_6_roc_12

##### BBI多空指标  
bbi

##### EMA指数平均数指标 
expma_12
expma_50

##### ARBR情绪指标 
ar
br

##### 真实波动N日平均值 
atr

##### 平行线差指标 
dma_dif
dma

##### 简易波动指标 
emv
maemv

##### PSY
psy
psyma

##### 振动升降指标 
asi
asit

##### MFI指标 
mfi

##### 梅斯线 
mass
mamass

##### 区间震荡线 
dpo
madpo

##### VR容量比率 
vr

##### 三重指数平滑平均线 
trix
trma

##### 肯特纳通道（Keltner Channel，KC）
kc_upper
kc_mid
kc_lower

##### 唐奇安通道（DC）
dc_upper
dc_mid
dc_lower

### 模型：

经典策略可以参考 https://github.com/zhy0313/ea-python ， 这些策略基本上是通过一个或者多个技术指标的值或者趋势判断。

2022年九坤Kaggle量化大赛的一些模型 
https://www.vzkoo.com/read/202301318245ba57351bdd7c187391ba.html 

九坤量化模型的数据条数约300万，共1211个time_id（推测是天数，大约是3年多）,3579个investment_id(推测是这么多个股票)


#### 经典模型
* xgboost/lightgbm reg 
* xgboost cls 

#### 神经网络
* mlp reg 
* lstm: 将特征的时间序列引进来 

## 代码
代码在code文件夹里，包含lightgbm和mlp两个模型。分别运行 `quant_reg_mlp_train.py` 和 `quant_reg_gbdt_train.py` 即可。

数据已经存到 `quant_reg_data.pkl` 文件里了，
每支股票每天一条数据，只使用了过去1年的数据，数据总量约30万。特征是按天统计的73个技术指标，技术指标的名称跟dataframe的列名一致。标签是第二天的涨跌幅/10.





