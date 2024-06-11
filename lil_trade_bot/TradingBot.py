import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import ta

# Import the data
df = yf.download("AAPL", period="120mo")[["Open", "High", "Low", "Adj Close", "Volume"]]

df.columns = ["open", "high", "low", "close", "volume"]

def feature_engineering(df):
  """ Create new variables"""

  # We copy the dataframe to avoid interferences in the data
  df_copy = df.dropna().copy()

  # Create the returns
  df_copy["returns"] = df_copy["close"].pct_change(1)

  # Create the SMAs
  df_indicators = ta.add_all_ta_features(
  df, open="open", high="high", low="low", close="close", volume="volume", fillna=True).shift(1)

  dfc = pd.concat((df_indicators, df_copy), axis=1)

  return dfc.dropna()

dfc = feature_engineering(df)

# Percentage train set
split = int(0.80*len(dfc))

# Train set creation
X_train = dfc.iloc[:split,6:dfc.shape[1]-1]
y_train = dfc[["returns"]].iloc[:split]


# Test set creation
X_test = dfc.iloc[split:,6:dfc.shape[1]-1]
y_test = dfc[["returns"]].iloc[split:]

# STANDARDISATION
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

plt.plot(X_train.values[:,0:15])
plt.title("WITHOUT Standardisation")
plt.show()

plt.plot(X_train_sc[:,0:15], alpha=0.5)
plt.title("with standardisation")
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=6)

X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

print(f"Without PCA: {np.shape(X_train)} \nWith PCA: {np.shape(X_train_pca)}")

from sklearn.svm import SVR

reg = SVR()

reg.fit(X_train_pca, y_train)

# Create predictions for the whole dataset
X = np.concatenate((X_train_pca, X_test_pca), axis=0)

dfc["prediction"] = reg.predict(X)

# We verify that the algorithm doesn't predict only way (positive or negative)
dfc["prediction"].plot()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_pred = reg.predict(X_test_pca)

# Calcul des métriques d'évaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", r2)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

def BackTest(serie, annualiazed_scalar=252):
  def drawdown_function(serie):

    # We compute Cumsum of the returns
    cum = serie.dropna().cumsum() + 1

    # We compute max of the cumsum on the period (accumulate max) # (1,3,5,3,1) --> (1,3,5,5,5)
    running_max = np.maximum.accumulate(cum)

    # We compute drawdown
    drawdown = cum/running_max - 1
    return drawdown

  # Import the benchmark
  sp500 = yf.download("^GSPC", period="120mo")["Adj Close"].pct_change(1)

  # Change the name
  sp500.name = "SP500"

  # Concat the returns and the sp500
  val = pd.concat((serie,sp500), axis=1).dropna()
  # Compute the drawdown
  drawdown = drawdown_function(serie)*100

  # Compute max drawdown
  max_drawdown = -np.min(drawdown)




  # Put a subplots
  fig, (cum, dra) = plt.subplots(1,2, figsize=(20,6))

  # Put a Suptitle
  fig.suptitle("Backtesting", size=20)

  # Returns cumsum chart
  cum.plot(serie.cumsum()*100, color="#39B3C7")

  # SP500 cumsum chart
  cum.plot(val["SP500"].cumsum()*100, color="#B85A0F")

  # Put a legend
  cum.legend(["Portfolio", "SP500"])

  # Set individual title
  cum.set_title("Cumulative Return", size=13)

  cum.set_ylabel("Cumulative Return %", size=11)

  # Put the drawdown
  dra.fill_between(drawdown.index,0,drawdown, color="#C73954", alpha=0.65)

  # Set individual title
  dra.set_title("Drawdown", size=13)

  dra.set_ylabel("drawdown in %", size=11)

  # Plot the graph
  plt.show()


  # Compute the sortino
  sortino = np.sqrt(annualiazed_scalar) * serie.mean()/serie.loc[serie<0].std()

  # Compute the beta
  beta = np.cov(val[["return", "SP500"]].values,rowvar=False)[0][1] / np.var(val["SP500"].values)

  # Compute the alpha
  alpha = annualiazed_scalar * (serie.mean() - beta*serie.mean())

  # Print the statistics
  print(f"Sortino: {np.round(sortino,3)}")
  print(f"Beta: {np.round(beta,3)}")
  print(f"Alpha: {np.round(alpha*100,3)} %")
  print(f"MaxDrawdown: {np.round(max_drawdown,3)} %")

  # Compute the position
dfc["position"] = np.sign(dfc["prediction"])

# Compute the returns
dfc["strategy"] = dfc["returns"] * dfc["position"].shift(1)

dfc["return"] = dfc["strategy"]
BackTest(dfc["return"].iloc[split:])

def svm_reg_trading(symbol):

  def feature_engineering(df):
    """ Create new variables"""

    # We copy the dataframe to avoid interferences in the data
    df_copy = df.dropna().copy()

    # Create the returns
    df_copy["returns"] = df_copy["close"].pct_change(1)

    # Create the SMAs
    df_indicators = ta.add_all_ta_features(
    df, open="open", high="high", low="low", close="close", volume="volume", fillna=True).shift(1)

    dfc = pd.concat((df_indicators, df_copy), axis=1)

    return dfc.dropna()


  # Import the data
  df = yf.download(symbol, period="120mo" )[["Open", "High", "Low", "Adj Close", "Volume"]]

  df.columns = ["open", "high", "low", "close", "volume"]

  dfc = feature_engineering(df)

  # Percentage train set
  split = int(0.80*len(dfc))

  # Train set creation
  X_train = dfc.iloc[:split,6:dfc.shape[1]-1]
  y_train = dfc[["returns"]].iloc[:split]


  # Test set creation
  X_test = dfc.iloc[split:,6:dfc.shape[1]-1]
  y_test = dfc[["returns"]].iloc[split:]


  # What you need to remind about this chapter
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()

  X_train_sc = sc.fit_transform(X_train)
  X_test_sc = sc.transform(X_test)


  from sklearn.decomposition import PCA
  pca = PCA(n_components=6)
  X_train_pca = pca.fit_transform(X_train_sc)
  X_test_pca = pca.transform(X_test_sc)

  # Import the class
  from sklearn.svm import SVR

  # Initialize the class
  reg = SVR()

  # Fit the model
  reg.fit(X_train_pca, y_train)

  # Create predictions for the whole dataset
  X = np.concatenate((X_train_pca, X_test_pca), axis=0)

  dfc["prediction"] = reg.predict(X)

  # Compute the position
  dfc["position"] = np.sign(dfc["prediction"])

  # Compute the returns
  dfc["strategy"] = dfc["returns"] * dfc["position"].shift(1)

  dfc["return"] = dfc["strategy"]
  BackTest(dfc["return"].iloc[split:])

  svm_reg_trading("IMX10603-USD")
  svm_reg_trading("NFLX")
  svm_reg_trading('JPY=X')
  svm_reg_trading('CHF=X')
  svm_reg_trading('EURCHF=X')
  svm_reg_trading('CL=F')
