# coding=utf8

############
# 線形回帰による機械学習
####

import scipy as sp
import matplotlib.pyplot as plt

### SciPyのgenfromtxt()からデータを読み込む ###
data = sp.genfromtxt("data/web_traffic.tsv",delimiter="\t")
#print(data[:10]) #データの中身
#print(data.shape) #データの行と列

### dataをベクトルに分割 ###
x = data[:,0]
y = data[:,1]

### 不適切なデータの取り除き ###
# SciPyのisnan()で数値であるかをBooleanで返す
#"~"は論理否定演算 Trueではない要素だけに
#print(sp.isnan(y))
#print(sp.sum(sp.isnan(y))) #不適切データの数
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]


### SciPyのpolyfit()による回帰分析 ###
#polyfit()は回帰分析を行う
#返り値
#parameters(array) 関数のパラメータ(θn,‥,θ1,θ0)
#residulas(array) 近似誤差値
#rank パラメータの数
#singular_values(array)
#rcond

##poly1d()はパラメータよりモデル関数を作る

# 単回帰
f1_parameters ,f1_residuals, f1_rank, f1_singular_values, f1_rcond = sp.polyfit(x,y,1,full=True)
f1 = sp.poly1d(f1_parameters)
print("f1:")
print(f1_residuals)
#print([f1_parameters,f1_residuals, f1_rank, f1_singular_values, f1_rcond])
#print(f1)

# 二次曲線
f2_parameters ,f2_residuals, f2_rank, f2_singular_values, f2_rcond = sp.polyfit(x,y,2,full=True)
f2 = sp.poly1d(f2_parameters)
print("f2:")
print(f2_residuals)
#print([f2_parameters,f2_residuals, f2_rank, f2_singular_values, f2_rcond])
#print(f2)

#三次曲線
f3_parameters ,f3_residuals, f3_rank, f3_singular_values, f3_rcond = sp.polyfit(x,y,3,full=True)
f3 = sp.poly1d(f3_parameters)
print("f3:")
print(f3_residuals)
#print([f3_parameters,f3_residuals, f3_rank, f3_singular_values, f3_rcond])
#print(f3)

#多項式曲線
f30_parameters ,f30_residuals, f30_rank, f30_singular_values, f30_rcond = sp.polyfit(x,y,30,full=True)
f30 = sp.poly1d(f30_parameters)
print("f30:")
print(f30_residuals)
print([f30_parameters,f30_residuals, f30_rank, f30_singular_values, f30_rcond])
#print(f30)


### 近似誤差値を求める ###
def getApproximationErrorValue(f, x, y):
	return (sp.sum((f(x) - y) ** 2)/(2 * x.size))
#各関数に
print("近似誤差値")
print(getApproximationErrorValue(f1, x, y))
print(getApproximationErrorValue(f2, x, y))
print(getApproximationErrorValue(f3, x, y))
print(getApproximationErrorValue(f30, x, y))

### テストデータ(ホールドアウトデータ)と訓練データに分けて、モデル関数を評価する ###

def random_test(x,y,deg): #訓練用データとテスト用データにランダムに分ける
	frac = 0.3 #テストに用いるデータの割合
	split_idx = int(frac * x.size) #テストデータに分けるためのインデックス
	shuffled = sp.random.permutation(list(range(len(x)))) #シャッフル
	test = sorted(shuffled[:split_idx]) #テスト用データのインデックス配列
	train = sorted(shuffled[split_idx:]) #訓練用データのインデックス配列
	
	function = sp.poly1d(sp.polyfit(x[train],y[train],deg))
	#print(function)
	
	print("f%i  :error_value=%f" % (deg,getApproximationErrorValue(function,x[test],y[test])))
	
print("テストデータと訓練データにによる評価")
random_test(x,y,1)
random_test(x,y,2)
random_test(x,y,3)
random_test(x,y,10)
random_test(x,y,30)


### 描画 ###

#散布図
plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)]) #x座標に対しての配置と名前
#plt.autoscale(tight=True) #自動で画面をスケール調整する
plt.grid() #グリッドをつける

#各曲線
f1x = sp.linspace(0,x[-1]+300)#プロットようにx値を生成
plt.plot(f1x,f1(f1x),linewidth=4)

f2x = sp.linspace(0,x[-1]+300)#プロットようにx値を生成
plt.plot(f2x,f2(f2x),linewidth=4)

f3x = sp.linspace(0,x[-1]+200)#プロットようにx値を生成
plt.plot(f3x,f3(f3x),linewidth=4)

f30x = sp.linspace(0,x[-1]+10)#プロットようにx値を生成
plt.plot(f30x,f30(f30x),linewidth=4)


plt.legend(["polyfit_deg=1","polyfit_deg=2","polyfit_deg=3","polyfit_deg=30"],loc="upper left")#左上に直線の名前を配置
plt.show()





