## 时序分析Py工具包

### prophet

`github` ⭐17.1k

- 仅适用单变量的时序预测模型
- 仅支持 prophet 这一个模型。

### tsfresh

`github` ⭐7.8k

- 用于时序特征工程，包括对时间序列自动提取特征、特征选择、特征转换等
- 不能用于实现时序预测或时序分类等任务。

### sktime

`github` ⭐7k

- 具有时序预测、时序分类、时序回归、时序聚类等
- 可与 scikit-learn 进行互操作
- 提供经典的统计学模型系列（ARIMA、ETS、prophet等）
- 提供模型Ensemble能力和AutoML功能（带有模型选择和自动调参功能的时序建模）
- 也提供一些深度学习的模型，如Transformer等

#### 功能表

| Module | Status | Links |
|---|---|---|
| **[Forecasting]** | stable | [Tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html) · [API Reference](https://www.sktime.net/en/latest/api_reference/forecasting.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py)  |
| **[Time Series Classification]** | stable | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/02_classification.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/classification.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/classification.py) |
| **[Time Series Regression]** | stable | [API Reference](https://www.sktime.net/en/latest/api_reference/regression.html) |
| **[Transformations]** | stable | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/transformations.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **[Parameter fitting]** | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/param_est.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py)  |
| **[Time Series Clustering]** | maturing | [API Reference](https://www.sktime.net/en/latest/api_reference/clustering.html) ·  [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/clustering.py) |
| **[Time Series Distances/Kernels]** | maturing | [Tutorial](https://github.com/sktime/sktime/blob/main/examples/03_transformers.ipynb) · [API Reference](https://www.sktime.net/en/latest/api_reference/dists_kernels.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/dist_kern_panel.py) |
| **[Time Series Alignment]** | experimental | [API Reference](https://www.sktime.net/en/latest/api_reference/alignment.html) · [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/alignment.py) |
| **[Annotation]** | experimental | [Extension Template](https://github.com/sktime/sktime/blob/main/extension_templates/annotation.py) |
| **[Distributions and simulation]** | experimental |  |

### darts 

`github` ⭐6.6k

- 时序分析工具的集大成者，支持单变量和多变量预测
- 聚焦于时序预测问题，支持从ARIMA到深度学习模型，如Transformer、TCN等新的时序建模方法。
- 该库还可以方便地对模型进行回溯测试，并将多个模型的预测和外部回归组合起来。

#### 可用模型

| Model                                                                                                                                                                                                                                                                                                   | Sources                                                                                                                                                                                                                           | Target Series Support:<br/><br/>Univariate/<br/>Multivariate | Covariates Support:<br/><br/>Past-observed/<br/>Future-known/<br/>Static | Probabilistic Forecasting:<br/><br/>Sampled/<br/>Distribution Parameters | Training & Forecasting on Multiple Series |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|-------------------------------------------|
| **Baseline Models**<br/>([LocalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms))                                                                                                                                                              |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [NaiveMean](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveMean)                                                                                                                                                         |                                                                                                                                                                                                                                   | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| [NaiveSeasonal](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveSeasonal)                                                                                                                                                 |                                                                                                                                                                                                                                   | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| [NaiveDrift](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveDrift)                                                                                                                                                       |                                                                                                                                                                                                                                   | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| [NaiveMovingAverage](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveMovingAverage)                                                                                                                                       |                                                                                                                                                                                                                                   | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| **Statistical / Classic Models**<br/>([LocalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms))                                                                                                                                                 |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [ARIMA](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html#darts.models.forecasting.arima.ARIMA)                                                                                                                                                                         |                                                                                                                                                                                                                                   | 🟩 🟥                                                        | 🟥 🟩 🟥                                                                 | 🟩 🟥                                                                    | 🟥                                        |
| [VARIMA](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.varima.html#darts.models.forecasting.varima.VARIMA)                                                                                                                                                                     |                                                                                                                                                                                                                                   | 🟥 🟩                                                        | 🟥 🟩 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| [AutoARIMA](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html#darts.models.forecasting.auto_arima.AutoARIMA)                                                                                                                                                       |                                                                                                                                                                                                                                   | 🟩 🟥                                                        | 🟥 🟩 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| [StatsForecastAutoArima](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_arima.html#darts.models.forecasting.sf_auto_arima.StatsForecastAutoARIMA) (faster AutoARIMA)                                                                                                    | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | 🟩 🟥                                                        | 🟥 🟩 🟥                                                                 | 🟩 🟥                                                                    | 🟥                                        |
| [ExponentialSmoothing](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.exponential_smoothing.html#darts.models.forecasting.exponential_smoothing.ExponentialSmoothing)                                                                                                           |                                                                                                                                                                                                                                   | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟩 🟥                                                                    | 🟥                                        |
| [StatsforecastAutoETS](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_ets.html#darts.models.forecasting.sf_auto_ets.StatsForecastAutoETS)                                                                                                                               | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | 🟩 🟥                                                        | 🟥 🟩 🟥                                                                 | 🟩 🟥                                                                    | 🟥                                        |
| [StatsforecastAutoCES](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_ces.html#darts.models.forecasting.sf_auto_ces.StatsForecastAutoCES)                                                                                                                               | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| [BATS](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tbats_model.html#darts.models.forecasting.tbats_model.BATS) and [TBATS](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tbats_model.html#darts.models.forecasting.tbats_model.TBATS)               | [TBATS paper](https://robjhyndman.com/papers/ComplexSeasonality.pdf)                                                                                                                                                              | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟩 🟥                                                                    | 🟥                                        |
| [Theta](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html#darts.models.forecasting.theta.Theta) and [FourTheta](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html#darts.models.forecasting.theta.FourTheta)                             | [Theta](https://robjhyndman.com/papers/Theta.pdf) & [4 Theta](https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R)                                                                                          | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| [StatsForecastAutoTheta](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_theta.html#darts.models.forecasting.sf_auto_theta.StatsForecastAutoTheta)                                                                                                                       | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟩 🟥                                                                    | 🟥                                        |
| [Prophet](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html#darts.models.forecasting.prophet_model.Prophet) (see [install notes](https://github.com/unit8co/darts/blob/master/INSTALL.md#enabling-support-for-facebook-prophet)) | [Prophet repo](https://github.com/facebook/prophet)                                                                                                                                                                               | 🟩 🟥                                                        | 🟥 🟩 🟥                                                                 | 🟩 🟥                                                                    | 🟥                                        |
| [FFT](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.fft.html#darts.models.forecasting.fft.FFT) (Fast Fourier Transform)                                                                                                                         |                                                                                                                                                                                                                                   | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| [KalmanForecaster](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.kalman_forecaster.html#darts.models.forecasting.kalman_forecaster.KalmanForecaster) using the Kalman filter and N4SID for system identification                                                               | [N4SID paper](https://people.duke.edu/~hpgavin/SystemID/References/VanOverschee-Automatica-1994.pdf)                                                                                                                              | 🟩 🟩                                                        | 🟥 🟩 🟥                                                                 | 🟩 🟥                                                                    | 🟥                                        |
| [Croston](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.croston.html#darts.models.forecasting.croston.Croston) method                                                                                                                                                          |                                                                                                                                                                                                                                   | 🟩 🟥                                                        | 🟥 🟥 🟥                                                                 | 🟥 🟥                                                                    | 🟥                                        |
| **Regression Models**<br/>([GlobalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms))                                                                                                                                                          |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [RegressionModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_model.html#darts.models.forecasting.regression_model.RegressionModel): generic wrapper around any sklearn regression model                                                                          |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟥 🟥                                                                    | 🟩                                        |
| [LinearRegressionModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html#darts.models.forecasting.linear_regression_model.LinearRegressionModel)                                                                                                     |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [RandomForest](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html#darts.models.forecasting.random_forest.RandomForest)                                                                                                                                           |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟥 🟥                                                                    | 🟩                                        |
| [LightGBMModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html#darts.models.forecasting.lgbm.LightGBMModel),                                                                                                                                                          |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [XGBModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html#darts.models.forecasting.xgboost.XGBModel)                                                                                                                                                               |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [CatBoostModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.catboost_model.html#darts.models.forecasting.catboost_model.CatBoostModel)                                                                                                                                       |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| **PyTorch (Lightning)-based Models**<br/>([GlobalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms))                                                                                                                                           |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [RNNModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html#darts.models.forecasting.rnn_model.RNNModel) (incl. LSTM and GRU); equivalent to DeepAR in its probabilistic version                                                                                   | [DeepAR paper](https://arxiv.org/abs/1704.04110)                                                                                                                                                                                  | 🟩 🟩                                                        | 🟥 🟩 🟥                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [BlockRNNModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.block_rnn_model.html#darts.models.forecasting.block_rnn_model.BlockRNNModel) (incl. LSTM and GRU)                                                                                                                |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟥 🟥                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [NBEATSModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html#darts.models.forecasting.nbeats.NBEATSModel)                                                                                                                                                           | [N-BEATS paper](https://arxiv.org/abs/1905.10437)                                                                                                                                                                                 | 🟩 🟩                                                        | 🟩 🟥 🟥                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [NHiTSModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nhits.html#darts.models.forecasting.nhits.NHiTSModel)                                                                                                                                                               | [N-HiTS paper](https://arxiv.org/abs/2201.12886)                                                                                                                                                                                  | 🟩 🟩                                                        | 🟩 🟥 🟥                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [TCNModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html#darts.models.forecasting.tcn_model.TCNModel)                                                                                                                                                           | [TCN paper](https://arxiv.org/abs/1803.01271), [DeepTCN paper](https://arxiv.org/abs/1906.04397), [blog post](https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4) | 🟩 🟩                                                        | 🟩 🟥 🟥                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [TransformerModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.transformer_model.html#darts.models.forecasting.transformer_model.TransformerModel)                                                                                                                           |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟥 🟥                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [TFTModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html#darts.models.forecasting.tft_model.TFTModel) (Temporal Fusion Transformer)                                                                                                                             | [TFT paper](https://arxiv.org/pdf/1912.09363.pdf), [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/latest/models.html)                                                                                        | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [DLinearModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.dlinear.html#darts.models.forecasting.dlinear.DLinearModel)                                                                                                                                                       | [DLinear paper](https://arxiv.org/pdf/2205.13504.pdf)                                                                                                                                                                             | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [NLinearModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nlinear.html#darts.models.forecasting.nlinear.NLinearModel)                                                                                                                                                       | [NLinear paper](https://arxiv.org/pdf/2205.13504.pdf)                                                                                                                                                                             | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [TiDEModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tide_model.html#darts.models.forecasting.tide_model.TiDEModel)                                                                                                                                                       | [TiDE paper](https://arxiv.org/pdf/2304.08424.pdf)                                                                                                                                                                                | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| **Ensemble Models**<br/>([GlobalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms)): Model support is dependent on ensembled forecasting models and the ensemble model itself                                                                  |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [NaiveEnsembleModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveEnsembleModel)                                                                                                                                       |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |
| [RegressionEnsembleModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_ensemble_model.html#darts.models.forecasting.regression_ensemble_model.RegressionEnsembleModel)                                                                                             |                                                                                                                                                                                                                                   | 🟩 🟩                                                        | 🟩 🟩 🟩                                                                 | 🟩 🟩                                                                    | 🟩                                        |


### Kats 

`github` ⭐4.6k

使用此库，可以执行以下操作：
- 时间序列分析
- 模式检测，包括季节性、异常值、趋势变化
- 产生65个特征的特征工程模块
- 对时间序列数据建立预测模型，包括Prophet、ARIMA、Holt Winters等。

### gluonts 

`github` ⭐3.9k

#### [可用模型](https://ts.gluon.ai/stable/getting_started/models.html)

<table class="colwidths-auto docutils align-default">
<thead>
<tr class="row-odd"><th class="head"><p>Model + Paper</p></th>
<th class="head"><p>Local/global</p></th>
<th class="head"><p>Data layout</p></th>
<th class="head"><p>Architecture/method</p></th>
<th class="head"><p>Implementation</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td><p>DeepAR<br><a class="reference external" href="https://doi.org/10.1016/j.ijforecast.2019.07.001">Salinas et al. 2020</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>RNN</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deepar/_estimator.py">MXNet</a>, <a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/deepar/estimator.py">PyTorch</a></p></td>
</tr>
<tr class="row-odd"><td><p>DeepState<br><a class="reference external" href="https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html">Rangapuram et al. 2018</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>RNN, state-space model</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deepstate/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-even"><td><p>DeepFactor<br><a class="reference external" href="https://proceedings.mlr.press/v97/wang19k.html">Wang et al. 2019</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>RNN, state-space model, Gaussian process</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deep_factor/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-odd"><td><p>Deep Renewal Processes<br><a class="reference external" href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764">Türkmen et al. 2021</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>RNN</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/renewal/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-even"><td><p>GPForecaster</p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>MLP, Gaussian process</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/gp_forecaster/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-odd"><td><p>MQ-CNN<br><a class="reference external" href="https://arxiv.org/abs/1711.11053">Wen et al. 2017</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>CNN encoder, MLP decoder</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/seq2seq/_mq_dnn_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-even"><td><p>MQ-RNN<br><a class="reference external" href="https://arxiv.org/abs/1711.11053">Wen et al. 2017</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>RNN encoder, MLP encoder</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/seq2seq/_mq_dnn_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-odd"><td><p>N-BEATS<br><a class="reference external" href="https://openreview.net/forum?id=r1ecqn4YwB">Oreshkin et al. 2019</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>MLP, residual links</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/n_beats/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-even"><td><p>Rotbaum<br><a class="reference external" href="https://openreview.net/forum?id=VD3TMzyxKK">Hasson et al. 2021</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>XGBoost, Quantile Regression Forests, LightGBM, Level Set Forecaster</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ext/rotbaum/_estimator.py">Numpy</a></p></td>
</tr>
<tr class="row-odd"><td><p>Temporal Fusion Transformer<br><a class="reference external" href="https://doi.org/10.1016/j.ijforecast.2021.03.012">Lim et al. 2021</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>LSTM, self attention</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/tft/_estimator.py">MXNet</a>, <a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/tft/estimator.py">PyTorch</a></p></td>
</tr>
<tr class="row-even"><td><p>Transformer<br><a class="reference external" href="https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html">Vaswani et al. 2017</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>MLP, multi-head attention</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/transformer/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-odd"><td><p>WaveNet<br><a class="reference external" href="https://arxiv.org/abs/1609.03499">van den Oord et al. 2016</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>Dilated convolution</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/wavenet/_estimator.py">MXNet</a>, <a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/wavenet/estimator.py">PyTorch</a></p></td>
</tr>
<tr class="row-even"><td><p>SimpleFeedForward</p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>MLP</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/simple_feedforward/_estimator.py">MXNet</a>, <a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/simple_feedforward/estimator.py">PyTorch</a></p></td>
</tr>
<tr class="row-odd"><td><p>DeepNPTS</p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>MLP</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/deep_npts/_estimator.py">PyTorch</a></p></td>
</tr>
<tr class="row-even"><td><p>MQF2<br><a class="reference external" href="https://arxiv.org/abs/2202.11316">Kan et al. 2022</a></p></td>
<td><p>Global</p></td>
<td><p>Univariate</p></td>
<td><p>RNN, ICNN</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/mqf2/estimator.py">PyTorch</a></p></td>
</tr>
<tr class="row-odd"><td><p>DeepVAR<br><a class="reference external" href="https://proceedings.neurips.cc/paper/2019/hash/0b105cf1504c4e241fcc6d519ea962fb-Abstract.html">Salinas et al. 2019</a></p></td>
<td><p>Global</p></td>
<td><p>Multivariate</p></td>
<td><p>RNN</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deepvar/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-even"><td><p>GPVAR<br><a class="reference external" href="https://proceedings.neurips.cc/paper/2019/hash/0b105cf1504c4e241fcc6d519ea962fb-Abstract.html">Salinas et al. 2019</a></p></td>
<td><p>Global</p></td>
<td><p>Multivariate</p></td>
<td><p>RNN, Gaussian process</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/gpvar/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-odd"><td><p>LSTNet<br><a class="reference external" href="https://doi.org/10.1145/3209978.3210006">Lai et al. 2018</a></p></td>
<td><p>Global</p></td>
<td><p>Multivariate</p></td>
<td><p>LSTM</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/lstnet/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-even"><td><p>DeepTPP<br><a class="reference external" href="https://arxiv.org/pdf/1909.12127">Shchur et al. 2020</a></p></td>
<td><p>Global</p></td>
<td><p>Multivariate events</p></td>
<td><p>RNN, temporal point process</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/tpp/deeptpp/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-odd"><td><p>DeepVARHierarchical<br><a class="reference external" href="https://proceedings.mlr.press/v139/rangapuram21a.html">Rangapuram et al. 2021</a></p></td>
<td><p>Global</p></td>
<td><p>Hierarchical</p></td>
<td><p>RNN</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/mx/model/deepvar_hierarchical/_estimator.py">MXNet</a></p></td>
</tr>
<tr class="row-even"><td><p>RForecast<br><a class="reference external" href="https://www.jstatsoft.org/article/view/v027i03">Hyndman et al. 2008</a></p></td>
<td><p>Local</p></td>
<td><p>Univariate</p></td>
<td><p>ARIMA, ETS, Croston, TBATS</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ext/r_forecast/_predictor.py">Wrapped R package</a></p></td>
</tr>
<tr class="row-odd"><td><p>Prophet<br><a class="reference external" href="https://doi.org/10.1080/00031305.2017.1380080">Taylor et al. 2017</a></p></td>
<td><p>Local</p></td>
<td><p>Univariate</p></td>
<td><p>-</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ext/prophet/_predictor.py">Wrapped Python package</a></p></td>
</tr>
<tr class="row-even"><td><p>NaiveSeasonal<br><a class="reference external" href="https://otexts.com/fpp2/simple-methods.html#seasonal-na%C3%AFve-method">Hyndman et al. 2018</a></p></td>
<td><p>Local</p></td>
<td><p>Univariate</p></td>
<td><p>-</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/model/seasonal_naive/_predictor.py">Numpy</a></p></td>
</tr>
<tr class="row-odd"><td><p>Naive2<br><a class="reference external" href="https://www.wiley.com/en-ie/Forecasting:+Methods+and+Applications,+3rd+Edition-p-9780471532330">Makridakis et al. 1998</a></p></td>
<td><p>Local</p></td>
<td><p>Univariate</p></td>
<td><p>-</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/ext/naive_2/_predictor.py">Numpy</a></p></td>
</tr>
<tr class="row-even"><td><p>NPTS</p></td>
<td><p>Local</p></td>
<td><p>Univariate</p></td>
<td><p>-</p></td>
<td><p><a class="reference external" href="https://github.com/awslabs/gluonts/blob/dev/src/gluonts/model/npts/_predictor.py">Numpy</a></p></td>
</tr>
</tbody>
</table>

### Merlion 

`github` ⭐3.1k

- 支持时序预测（Forecasting）和异常检测（Anomaly Detection）
- 支持单变量和多变量的时序分析
- 支持了模型融合（Ensemble）以及AutoML能力（带有模型选择和自动调参功能的时序建模）
- 对于时序预测任务，支持统计学模型和机器学习模型，其中统计学模型包括ARIMA、ETS、Prophet等；而机器学习模型则主要是基于决策树的集成模型，例如RF和GB等。
- 支持自动绘制真实值和预测结果及置信区间的对比曲线

### tslearn 

`github` ⭐2.6k

- 针对时序数据（time series）的机器学习相关工具包
- 可与 scikit-learn 进行互操作
- 具有数据预处理、分类、回归（预测是一种特殊形式的回归任务）以及聚类等功能
- tslearn可以与其他时序工具包进行整合使用，如scikit-learn、tsfresh、sktime及pyts等

#### 功能表

| data                                                                                                                                                                                         | processing                                                                                                              | clustering                                                                                                                                                       | classification                                                                                                                                                                          | regression                                                                                                                                                                           | metrics                                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [UCR Datasets](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.datasets.html#module-tslearn.datasets)                                                                           | [Scaling](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.preprocessing.html#module-tslearn.preprocessing) | [TimeSeriesKMeans](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html#tslearn.clustering.TimeSeriesKMeans) | [KNN Classifier](https://tslearn.readthedocs.io/en/stable/gen_modules/neighbors/tslearn.neighbors.KNeighborsTimeSeriesClassifier.html#tslearn.neighbors.KNeighborsTimeSeriesClassifier) | [KNN Regressor](https://tslearn.readthedocs.io/en/stable/gen_modules/neighbors/tslearn.neighbors.KNeighborsTimeSeriesRegressor.html#tslearn.neighbors.KNeighborsTimeSeriesRegressor) | [Dynamic Time Warping](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw.html#tslearn.metrics.dtw)    |
| [Generators](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.generators.html#module-tslearn.generators)                                                                         | [Piecewise](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.piecewise.html#module-tslearn.piecewise)       | [KShape](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KShape.html#tslearn.clustering.KShape)                               | [TimeSeriesSVC](https://tslearn.readthedocs.io/en/stable/gen_modules/svm/tslearn.svm.TimeSeriesSVC.html#tslearn.svm.TimeSeriesSVC)                                                      | [TimeSeriesSVR](https://tslearn.readthedocs.io/en/stable/gen_modules/svm/tslearn.svm.TimeSeriesSVR.html#tslearn.svm.TimeSeriesSVR)                                                   | [Global Alignment Kernel](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.gak.html#tslearn.metrics.gak) |
| Conversion([1](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.utils.html#module-tslearn.utils), [2](https://tslearn.readthedocs.io/en/stable/integration_other_software.html)) |                                                                                                                         | [KernelKmeans](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KernelKMeans.html#tslearn.clustering.KernelKMeans)             | [LearningShapelets](https://tslearn.readthedocs.io/en/stable/gen_modules/shapelets/tslearn.shapelets.LearningShapelets.html)                                    | [MLP](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.neural_network.html#module-tslearn.neural_network)                                                                | [Barycenters](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.barycenters.html#module-tslearn.barycenters)              |
|                                                                                                                                                                                              |                                                                                                                         |                                                                                                                                                                  | [Early Classification](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.early_classification.html#module-tslearn.early_classification)                                      |                                                                                                                                                                                      | [Matrix Profile](https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.matrix_profile.html#module-tslearn.matrix_profile)     |


### AutoTS 

`github` ⭐883

AutoTS 是一个自动化的时间序列预测库，可以使用简单的代码训练多个时间序列模型，此库的一些最佳功能包括：

- 利用遗传规划优化方法寻找最优时间序列预测模型。
- 提供置信区间预测值的下限和上限。
- 它训练各种各样的模型，如统计的，机器学习以及深度学习模型
- 它还可以执行最佳模型的自动集成
- 它还可以通过学习最优NaN插补和异常值去除来处理混乱的数据
- 它可以运行单变量和多变量时间序列

### atspy 

`github` ⭐496

- 自动时间序列模型
- 该库的目标是预测单变量时间序列
- 可以加载数据并指定要运行的模型

### PaddleTS 

`github` ⭐405

*基于飞浆深度学习框架PaddlePaddle的开源时序建模算法库*
