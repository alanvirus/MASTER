# 复现 MASTER (AAAI-24) Table 1 全部指标

本仓库在WSL下运行，运行先请全局搜索项目中的“MASTER路径”全部替换为实际路径（eg./home/user/MASTER）

- 第一节介绍如何仅基于MASTER（不安装Qlib）完成模型训练和测试（测试不包含AR/IR指标）
- 第二节介绍如何安装Qlib，如何在Qlib框架下测试论文中的基线方法，以及如何在Qlib框架下测试自己训练的模型
> 本仓库fork于 `https://github.com/SJTU-DMTai/MASTER`(修复了一些小bug)，是一个**不依赖 Qlib 也能用 MASTER** 的轻量包，它**不能算 AR/IR指标**——AR/IR 属于组合回测层面指标，必须走 Qlib 的 `PortAnaRecord`。要复现完整 Table 1，整个流程都要在 Qlib 框架里跑。

# 一.仅基于MASTER（不安装Qlib）完成模型训练和测试

## 1. 下载数据

任选一个镜像（三份是同一个 zip）：

- OneDrive: https://1drv.ms/f/c/652674690cc447e6/Eu8Kxv4xxTFMtDQqTW0IU0UB8rnpjACA5twMi8BA_PfbSA
- MEGA: https://mega.nz/folder/MS8mUTbL#qeVz3KR1-MyXc_uLPtkvTg
- 百度网盘: https://pan.baidu.com/s/1qmDIepmGY1DVBTGGiipxfA?pwd=pm49

训练既可用 opensource 也可用 original，valid/test 只能用 opensource。最终目录：

```
data/
├── csi_market_information.csv          # 仓库已有
├── original/                           
│   ├── csi300_dl_train.pkl
│   └── csi800_dl_train.pkl
└── opensource/
    ├── csi300_dl_train.pkl
    ├── csi300_dl_valid.pkl
    ├── csi300_dl_test.pkl
    ├── csi800_dl_train.pkl
    ├── csi800_dl_valid.pkl
    └── csi800_dl_test.pkl
```

## 2. 测试

model文件夹下有训好的模型，可以直接测试

```bash
# CSI300 + opensource checkpoint（默认）
python main.py test

# CSI800
python main.py test --universe csi800

# 用 original 训出来的 checkpoint
python main.py test --prefix original

# 跑指定 seed
python main.py test --seeds 0 1 2 3 4
```

`main.py test` 默认 `--universe csi300 --prefix opensource --seeds 0`，会加载 `model/csi300_opensource_0.pkl` 在 opensource test 集上算 IC/ICIR/RIC/RICIR。


## 3. 训练

打开 `base_model.py`，定位到 `train_epoch` 里的这三行（约 110–112）：

```python
mask, label = drop_extreme(label)
feature = feature[mask, :, :]
label = zscore(label)
```

- `--prefix opensource`：**保留**这三行（默认状态）。opensource train 的 label 只做了 DropNA，需要在训练循环里现做 DropExtreme + CSZscoreNorm。
- `--prefix original`：**注释掉**这三行。original train dump 时已经做过 DropNA + DropExtreme + CSZscoreNorm。

漏切的话不会报错，但训出来的指标会偏低或不收敛。


训练的配置可通过命令行参数指定（也支持改 `main.py` 默认值）：

```bash
# CSI300 + opensource（默认）
python main.py train

# CSI800
python main.py train --universe csi800

# 用 original 数据训
python main.py train --prefix original

# 单 seed 先验证流程
python main.py train --seeds 0

# 自定义 epoch 上限
python main.py train --n-epoch 60
```

默认会跑 5 个 seed (0–4)，每个 seed 触发阈值后保存 checkpoint 并测一下 test 集。

训练逻辑（`base_model.py:157-171`）：

- 每个 seed 跑最多 `n_epoch` 轮（默认 40）
- 每轮结束打印 train_loss、valid IC/ICIR/RIC/RICIR
- **当 train_loss 首次 ≤ 0.95 时**，保存权重到 `model/{universe}_{prefix}_{seed}.pkl` 并 `break`

经验值：CSI300 大约 20–40 轮能触发阈值，CSI800 略慢。如果 40 轮还没到 0.95，要么把 `--n-epoch` 调更大，要么把 `train_stop_loss_thred` 放宽。

# 二. 基于Qlib框架进行完整测试
## 1. 环境准备

### 1.1 装 Qlib

```bash
conda create -n master-qlib python=3.8 -y
conda activate master-qlib
cd MASTER路径
git clone https://github.com/microsoft/qlib.git
cd qlib
pip install -e .
# 论文锁版本（避免 pandas/torch 行为差异）
pip install "pandas==1.5.3" "torch==1.11.0" numpy 
```
（论文使用的torch版本比较老，在新显卡上会报错，我用的是2.0.1版本）


### 1.2 下载数据：必须用 chenditc 的 opensource_data

Qlib 默认 `get_data.py qlib_data --region cn` 拿到的数据**时段不对、股票池缺 CSI800**，跟论文 Table 对不上。要跟论文对齐**必须**用 chenditc 的扩展数据：

```bash
# 1. 先去 https://github.com/chenditc/investment_data/releases 下载最新 release zip
#    (qlib_bin.tar.gz)
# 2. 解压到 MASTER路径/qlib_data/cn_data
mkdir -p MASTER路径/qlib_data/cn_data
tar -xzf qlib_bin.tar.gz -C MASTER路径/qlib_data/cn_data --strip-components=1
```

解压后目录：

```
cn_data/
├── calendars
├── features
└── instruments
```

数据要点：
- 时段覆盖到 2022 年（默认 Qlib 数据只到 2020）
- 包含 CSI800 股票池
- 包含 sh000300/sh000905/sh000906 三个市场指数（构造 market 信息要用）

### 1.3 验证 Qlib 装好了

```bash
python -c "import qlib; qlib.init(provider_uri='MASTER路径/qlib_data/cn_data', region='cn'); print('ok')"
```
输出 `ok` 且无 warning 即正常。
---

## 2. 公共 yaml 模板（论文配置）

所有模型都用同一份回测设置，只换 `task.model` 那块。先把下面这份保存为 `MASTER路径/master_repro/_base.yaml`（路径自定，下面所有 yaml 都在它基础上派生）：

```yaml
qlib_init:
    provider_uri: MASTER路径/qlib_data/cn_data
    region: cn

market: &market csi800
benchmark: &benchmark sh000906

data_handler_config: &data_handler_config
    start_time: 2008-01-01
    end_time: 2022-12-31
    fit_start_time: 2008-01-01
    fit_end_time: 2020-03-31
    instruments: *market
    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSZScoreNorm
          kwargs:
              fields_group: label
    label: ["Ref($close, -5) / Ref($close, -1) - 1"]

port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 30
            n_drop: 30
    backtest:
        start_time: 2020-07-01
        end_time: 2022-12-31
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5

task:
    # 各基线在派生 yaml 里覆盖
    model: ~
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2020-03-31]
                valid: [2020-04-01, 2020-06-30]
                test: [2020-07-01, 2022-12-31]
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              model: <MODEL>
              dataset: <DATASET>
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              ana_long_short: False
              ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
              config: *port_analysis_config
```

关键字段说明：

| 字段 | 论文值 | 说明 |
|---|---|---|
| `market` | `csi800` | 论文主表用 CSI800；做 CSI300 时改成 `csi300` 并把 `benchmark` 改 `sh000300` |
| `benchmark` | `sh000906` | CSI800 对应基准指数 |
| `label` | `Ref($close, -5) / Ref($close, -1) - 1` | 5 日相对收益（T+1 进 T+6 出）|
| `train/valid/test` | 2008–2020.03 / 2020.04–2020.06 / 2020.07–2022.12 | 论文切分 |
| `topk/n_drop` | 30/30 | TopK-Drop 组合策略 |
| `open_cost/close_cost` | 0.0005 / 0.0015 | 双边手续费（含印花税）|

> Qlib 自带 yaml 默认 train 段是 2008–2014，test 是 2017+，跟论文不一致。**派生时一定要把 segments 改成上面这套。**

---

## 3. 跑 Qlib 自带基线

### 3.1 9基线一览

论文 Table 1 出现的 Qlib 基线 + 对应 Qlib 模型类：

| 论文名 | Qlib 模型类 | module_path |
|---|---|---|
| LSTM | LSTM | `qlib.contrib.model.pytorch_lstm_ts` |
| GRU | GRU | `qlib.contrib.model.pytorch_gru_ts` |
| Transformer | Transformer | `qlib.contrib.model.pytorch_transformer_ts` |
| GATs | GATs | `qlib.contrib.model.pytorch_gats_ts` |
| TCN | TCN | `qlib.contrib.model.pytorch_tcn_ts` |
| XGBoost | XGBoost | `qlib.contrib.model.xgboost` |

> 这些 `*_ts` 后缀的模型用滚动序列输入（lookback window），跟 MASTER 一致；不带 `_ts` 的按各自实现走。

### 3.2 派生 yaml 模板 （该部分无需进行，结果已经在master_repro文件夹下了）

每个基线建一个目录，复制 `_base.yaml` 后只改 `task.model` 那块。以 **LSTM** 为例（`MASTER路径/master_repro/lstm/lstm.yaml`）：

```yaml
# 顶部跟 _base.yaml 完全一致（直接整段复制），只改 task.model
task:
    model:
        class: LSTM
        module_path: qlib.contrib.model.pytorch_lstm_ts
        kwargs:
            d_feat: 158
            hidden_size: 64
            num_layers: 2
            dropout: 0.0
            n_epochs: 200
            lr: 1e-3
            early_stop: 10
            batch_size: 800
            metric: loss
            loss: mse
            n_jobs: 20
            GPU: 0
            seed: 0
    dataset:
        class: TSDatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2020-03-31]
                valid: [2020-04-01, 2020-06-30]
                test: [2020-07-01, 2022-12-31]
            step_len: 8
    record: # 跟 _base.yaml 一样
        - { class: SignalRecord, module_path: qlib.workflow.record_temp,
            kwargs: { model: <MODEL>, dataset: <DATASET> } }
        - { class: SigAnaRecord, module_path: qlib.workflow.record_temp,
            kwargs: { ana_long_short: False, ann_scaler: 252 } }
        - { class: PortAnaRecord, module_path: qlib.workflow.record_temp,
            kwargs: { config: *port_analysis_config } }
```

> `step_len: 8` 与 MASTER 的 lookback 一致。`d_feat: 158` 对应 Alpha158 全部因子。

**其它基线的 model kwargs**： Qlib 自带 yaml 抄：

```bash
# Qlib 自带模板路径
ls qlib/examples/benchmarks/
# LSTM/  GRU/  ALSTM/  Transformer/  SFM/  GATs/  TRA/  HIST/  IGMTF/  ...
cat qlib/examples/benchmarks/GATs/workflow_config_gats_Alpha158.yaml
```

把它的 `task.model` 块整段拷到你的派生 yaml 里覆盖即可。**注意只抄 model 块，segments / handler / record 全部用 `_base.yaml` 这套以保持公平比较**。

### 3.3 跑基线（单个seed）

```bash
cd MASTER路径/master_repro/lstm
qrun lstm.yaml
```

终端会打印：

```
'IC':                0.04xx
'ICIR':              0.3xxx
'Rank IC':           0.05xx
'Rank ICIR':         0.4xxx
'excess_return_without_cost.annualized_return':   0.xxxx   ← AR
'excess_return_without_cost.information_ratio':   1.xxxx   ← IR
'excess_return_without_cost.max_drawdown':       -0.0xxx
'excess_return_with_cost.annualized_return':      0.xxxx
'excess_return_with_cost.information_ratio':      1.xxxx
```

> 论文 Table 1 报的是 `with_cost` 还是 `without_cost` 没明说。一般文章报 `without_cost`（更"干净"），但鉴于 MASTER 的 yaml 里手续费配齐了，**两套都记录**，对照看哪套接近论文数字。

### 3.4 5 个 seed 跑均值±标准差

```bash
./run_seeds.sh <model_dir>      e.g. ./run_seeds.sh lstm
```

跑完用 mlflow 收：
```bash
mlflow ui --backend-store-uri MASTER路径/master_repro/lstm/mlruns
# 浏览器打开 http://localhost:5000，把 5 个 run 的指标导成 csv
```

或者命令行：
```bash
python <<'PY'
from mlflow.tracking import MlflowClient
import os, statistics
client = MlflowClient(tracking_uri=f"file://{os.path.expanduser('MASTER路径/master_repro/lstm/mlruns')}")
exps = client.search_experiments()
for exp in exps:
    runs = client.search_runs([exp.experiment_id])
    keys = ['IC', 'ICIR', 'Rank IC', 'Rank ICIR',
            'excess_return_without_cost.annualized_return',
            'excess_return_without_cost.information_ratio']
    for k in keys:
        vals = [r.data.metrics[k] for r in runs if k in r.data.metrics]
        if vals:
            print(f"{k:55s} {statistics.mean(vals):.4f} ± {statistics.pstdev(vals):.4f}")
PY
```

---

## 4. 跑 MASTER（用本仓库的 patch）

### 4.1 把 patch 应用到 Qlib

```bash
# 1. patch 模型实现
cp MASTER路径/qlib-update/pytorch_master_ts.py \
   qlib/qlib/contrib/model/pytorch_master_ts.py

# 2. 准备 yaml
mkdir -p MASTER路径/master_repro/master
cp MASTER路径/qlib-update/workflow_config_master_Alpha158.yaml \
   MASTER路径/master_repro/master/master.yaml
```

### 4.2 修 yaml 里 provider_uri

```yaml
qlib_init:
    provider_uri: MASTER路径/qlib_data/cn_data    # 改成你 chenditc 数据的路径
    region: cn
```


### 4.3 修 marketDataHandler（关键，否则 m 信息缺失）

`qlib-update/workflow_config_master_Alpha158.yaml` 引用了 `MASTERTSDatasetH`，它内部用 `marketDataHandler` 来构造 63 维 market 信息。Qlib 默认实现里 `marketDataHandler.get_feature_config()` 用的是 sh000300 + sh000905 + sh000852（CSI100/300/500 之类），**跟论文用的 sh000300 + sh000905 + sh000906（CSI300/500/800）不一致**。

打开 `qlib/qlib/contrib/data/dataset.py`，找到 `marketDataHandler` 类，把 `get_feature_config` 的返回值替换成本仓库 README 第 100–107 行那一大段（覆盖 sh000300/sh000905/sh000906 三个指数）。直接复制粘贴。

### 4.4 跑

```bash
cd MASTER路径/master_repro/master
# 5 seeds
for s in 0 1 2 3 4; do
    sed "s/seed: 0/seed: $s/" master.yaml > master_s${s}.yaml
    qrun master_s${s}.yaml 2>&1 | tee master_s${s}.log
done
```

注意 `qlib-update/pytorch_master_ts.py` 的 `MASTERModel.__init__` 在初始化时会保存 `save_prefix`，确保 `save_prefix` 拼上 seed 后不会互相覆盖；如果默认行为有问题，加 `--save-prefix master_s${s}` 之类的覆盖。

### 4.5 CSI300

yaml里目前都是基于csi800跑，跑csi300需要改yaml：

```yaml
market: &market csi300
benchmark: &benchmark sh000300
```

记得 model 的 `kwargs.beta` 论文里 CSI300 是 5、CSI800 是 2（在 yaml 里加 `beta` 字段）。

---
