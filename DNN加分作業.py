'''
1. 請上網抓取 2 15 3 2 2512310 06 0~ 0 ，下載 5 .00 0 TW、233 .0 TW 收盤價，針對每一檔
股票以滾動式窗方式估計 2 15 3 2 25 30 06 0~ 0 06 0 每一檔股票的第 t 期參數( )X :
α10 、 β10 、 σ10 、 S kew10 、 α2 0 、 β2 0 、 σ2 0 、 S kew20 、 α 3 0 、 β3 0 、 σ3 0 、 S kew30 、 α6 0 、 β6 0 、 σ 6 0
，以第 1t+ 期的當天 IRR 為輸出變數 Y，當成 DNN 的訓練期。
以 2 15 1 2 2512310 070 ~ 0 為測試期，計算 DNN 對每一檔股票未來報酬率預測的正確
率。
'''
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def R(data:pd.Series):
    r = np.log(data/data.shift()).dropna()
    return r

def ols_alpha(data:pd.Series, market_data:pd.Series):
    m = market_data.reindex(data.index)
    df = pd.concat([data,m],axis = 1).dropna(how='any')
    X = sm.add_constant(df.iloc[:,1:]).astype(float)
    Y = df.iloc[:,0].astype(float)
    return sm.OLS(Y, X).fit().params.iloc[0]


def ols_beta(data:pd.Series, market_data:pd.Series):
    m = market_data.reindex(data.index)
    df = pd.concat([data,m],axis = 1).dropna(how='any')
    X = sm.add_constant(df.iloc[:,1:]).astype(float)
    Y = df.iloc[:,0].astype(float)
    return sm.OLS(Y, X).fit().params.iloc[1]

def build_features(data:pd.Series, market_data:pd.Series, windows:list):
    R_stock = R(data)
    R_market = R(market_data)
    df = pd.concat([R_stock,R_market], axis = 1).dropna(how='any')
    df.columns = ['ri','rm']
    out = pd.DataFrame(index = df.index)
    for i in windows:
        out[f'alpha{i}'] = df['ri'].rolling(window = i).apply(lambda s:ols_alpha(s,R_market))
        out[f'beta{i}'] = df['ri'].rolling(window = i).apply(lambda s:ols_beta(s, R_market))
        out[f'std{i}'] = df['ri'].rolling(window = i).std(ddof=1)
        out[f'skew{i}'] = df['ri'].rolling(window = i).skew()
    return out

#下載資料        
start_date = '2015-06-30'
train_end = '2025-07-01'
test_start = '2025-07-01'
end_date = '2025-12-12'
stock_data = yf.download(['0050.TW','2330.TW'], start_date, end_date)['Close']
market_data = yf.download('^TWII', start_date, end_date)['Close']

#建立全features、label
windows = [10, 20, 30, 60]
stock_list = stock_data.columns.tolist()
tables = []
for i in stock_list:
    stock = stock_data[i]
    features = build_features(stock, market_data, windows)
    label = np.log(stock/stock.shift()).shift(-1).rename('IRR_t+1')
    table = features.join(label, how='inner').dropna(how='any')
    table['asset_id'] = i
    tables.append(table)
df_long_all = pd.concat(tables, axis = 0).sort_index()


#訓練期跟測試期分開，並對訓練期做embedding方便後面joint模型丟進DNN能分辨是哪一檔股票
df_train = df_long_all.loc[:train_end].copy()
df_test = df_long_all.loc[test_start:].copy()

asset_train = sorted(df_train['asset_id'].unique().tolist())
asset2idx = {a:i for i, a in enumerate(asset_train)}
unk_idx = len(asset2idx)
asset2idx['UNK'] = unk_idx
idx2asset = {i:a for a,i in asset2idx.items()}

df_train['asset_idx'] = df_train['asset_id'].map(asset2idx).fillna(unk_idx).astype('int32')
df_test['asset_idx'] = df_test['asset_id'].map(asset2idx).fillna(unk_idx).astype('int32')

#測試期跟訓練期各自分成三組要丟給DNN得資料(0050、2330、joint)
df_0050_train = df_train[df_train['asset_id'] == '0050.TW'].copy()
df_2330_train = df_train[df_train['asset_id'] == '2330.TW'].copy()
df_joint_train = df_train.copy()
df_0050_test = df_test[df_test['asset_id'] == '0050.TW'].copy()
df_2330_test = df_test[df_test['asset_id'] == '2330.TW'].copy()
df_joint_test = df_test.copy()


#把每一組的features、label都轉成array
X_0050_train = df_0050_train[[i for i in df_0050_train.columns if i not in  ['asset_id','asset_idx', 'IRR_t+1']]].to_numpy()
Y_0050_train = df_0050_train['IRR_t+1'].to_numpy()
X_2330_train = df_2330_train[[i for i in df_2330_train.columns if i not in ['asset_id','asset_idx', 'IRR_t+1']]].to_numpy()
Y_2330_train = df_2330_train['IRR_t+1'].to_numpy()
X_joint_train = df_joint_train[[i for i in df_joint_train.columns if i not in ['asset_id','asset_idx', 'IRR_t+1']]].to_numpy()
Y_joint_train = df_joint_train['IRR_t+1'].to_numpy()
A_joint_train = df_joint_train['asset_idx'].to_numpy().astype('int32').reshape(-1, 1)

X_0050_test = df_0050_test[[i for i in df_0050_test.columns if i not in  ['asset_id','asset_idx', 'IRR_t+1']]].to_numpy()
Y_0050_test = df_0050_test['IRR_t+1'].to_numpy()
X_2330_test = df_2330_test[[i for i in df_2330_test.columns if i not in ['asset_id','asset_idx', 'IRR_t+1']]].to_numpy()
Y_2330_test = df_2330_test['IRR_t+1'].to_numpy()
X_joint_test = df_joint_test[[i for i in df_joint_test.columns if i not in ['asset_id','asset_idx', 'IRR_t+1']]].to_numpy()
Y_joint_test = df_joint_test['IRR_t+1'].to_numpy()
A_joint_test  = df_joint_test['asset_idx'].to_numpy().astype('int32').reshape(-1, 1)


#標準化
scaler_0050 = StandardScaler()
scaler_2330 = StandardScaler()
scaler_joint = StandardScaler()

X_0050_train_std = scaler_0050.fit_transform(X_0050_train)
X_2330_train_std = scaler_2330.fit_transform(X_2330_train)
X_joint_train_std = scaler_joint.fit_transform(X_joint_train)

X_0050_test_std = scaler_0050.transform(X_0050_test)
X_2330_test_std = scaler_2330.transform(X_2330_test)
X_joint_test_std = scaler_joint.transform(X_joint_test)


#建立DNN模型的funtion
def direction_acc_tf(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    return tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true), tf.sign(y_pred)), tf.float32))


def make_tolerance_acc_tf(eps: float):
    eps = tf.constant(eps, dtype=tf.float32)

    def tol_acc(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1,))
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), (-1,))
        return tf.reduce_mean(tf.cast(tf.abs(y_pred - y_true) <= eps, tf.float32))

    # Keras 顯示用名字（不然會叫 tol_acc）
    tol_acc.__name__ = f"tol_acc_{str(eps.numpy()).replace('.', '_')}"
    return tol_acc
tol_acc = make_tolerance_acc_tf(0.005)

def build_DNN_model(
    n_features: int,
    *,
    use_embedding: bool = False,
    num_assets: int | None = None,
    emb_dim: int = 8,
    hidden_units: tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.2,
    l2_strength: float = 1e-4,
    learning_rate: float = 1e-3,
    loss: str | tf.keras.losses.Loss = "mse",
) -> tf.keras.Model:
    
    # Input 1: 數值特徵
    inp_x = layers.Input(shape=(n_features,), name="x_features")
    h = inp_x

    #Input 2 (optional): 資產索引 -> Embedding 
    if use_embedding:
        if num_assets is None or num_assets <= 0:
            raise ValueError("use_embedding=True 時必須提供 num_assets（例如 len(asset2idx)）。")

        inp_a = layers.Input(shape=(1,), dtype="int32", name="asset_idx")
        emb = layers.Embedding(input_dim=num_assets, output_dim=emb_dim, name="asset_emb")(inp_a)
        emb = layers.Flatten(name="asset_emb_flat")(emb)

        h = layers.Concatenate(name="concat_x_emb")([h, emb])
        inputs = [inp_x, inp_a]
    else:
        inputs = inp_x

    # --- Dense trunk ---
    reg = regularizers.l2(l2_strength) if (l2_strength and l2_strength > 0) else None

    for k, units in enumerate(hidden_units, start=1):
        h = layers.Dense(units, activation="relu", kernel_regularizer=reg, name=f"dense_{k}")(h)
        if dropout_rate and dropout_rate > 0:
            h = layers.Dropout(dropout_rate, name=f"dropout_{k}")(h)

    # --- Output (regression) ---
    out = layers.Dense(1, name="y_pred")(h)

    model = Model(inputs=inputs, outputs=out, name=("joint_emb_dnn" if use_embedding else "single_dnn"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss, metrics=[direction_acc_tf, tol_acc]
    )
    return model

    
#訓練及測驗

EPOCHS = 50
BATCH = 64

# （可選）早停，避免 overfit
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ========= 1) 0050-only =========
model_0050 = build_DNN_model(n_features=X_0050_train_std.shape[1], hidden_units=(256, 128, 64, 32))
history_0050 = model_0050.fit(
    X_0050_train_std, Y_0050_train,
    epochs=EPOCHS, batch_size=BATCH,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

loss_0050, diracc_0050, tolacc_0050 = model_0050.evaluate(X_0050_test_std, Y_0050_test, verbose=0)
print(f"[0050] test loss={loss_0050:.6f}, dir_acc={diracc_0050:.4f}, tol_acc={tolacc_0050:.4f}")

# ========= 2) 2330-only =========
model_2330 = build_DNN_model(n_features=X_2330_train_std.shape[1], hidden_units=(256, 128, 64, 32))
history_2330 = model_2330.fit(
    X_2330_train_std, Y_2330_train,
    epochs=EPOCHS, batch_size=BATCH,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

loss_2330, diracc_2330, tolacc_2330 = model_2330.evaluate(X_2330_test_std, Y_2330_test, verbose=0)
print(f"[2330] test loss={loss_2330:.6f}, dir_acc={diracc_2330:.4f}, tol_acc={tolacc_2330:.4f}")

# ========= 3) joint + embedding =========
model_joint = build_DNN_model(
    n_features=X_joint_train_std.shape[1],
    hidden_units=(256, 128, 64, 32),
    use_embedding=True,
    num_assets=len(asset2idx),   # 含 UNK
    emb_dim=8
)

history_joint = model_joint.fit(
    {"x_features": X_joint_train_std, "asset_idx": A_joint_train},
    Y_joint_train,
    epochs=EPOCHS, batch_size=BATCH,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

loss_joint, diracc_joint, tolacc_joint = model_joint.evaluate(
    {"x_features": X_joint_test_std, "asset_idx": A_joint_test},
    Y_joint_test,
    verbose=0
)
print(f"[JOINT] test loss={loss_joint:.6f}, dir_acc={diracc_joint:.4f}, tol_acc={tolacc_joint:.4f}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







