# coding=utf8
import os

import scipy as sp

data = sp.genfromtxt("data/web_traffic.tsv",delimiter="\t")

#ベクトルに分割
x = data[:,0] #経過時間(一時間単位)とする
y = data[:,1] #アクセス数とする

#変化点でデータ分類
inflection1 = 3.5*7*24 #変化点(急激に変化するポイント)の位置
x_a = x[:inflection1]
x_b = x[inflection1:]
y_a = y[:inflection1]
y_b = y[inflection1:]

#不適切なデータの取り除き
x_a = x_a[~sp.isnan(y_a)]
x_b = x_b[~sp.isnan(y_b)]
y_a = y_a[~sp.isnan(y_a)]
y_b = y_b[~sp.isnan(y_b)]


#一次曲線
f1a = sp.poly1d(sp.polyfit(x_a,y_a,1))
f1b = sp.poly1d(sp.polyfit(x_b,y_b,1))
print(f1a)
print(f1b)


#近似誤差値
def error_value(f, x, y): #目的関数を利用
	return (sp.sum((f(x) - y) ** 2)/(2 * x.size))

def random_test(x,y,deg): #訓練用データとテスト用データにランダムに分ける
	frac = 0.7 #訓練データの割合
	split_idx = int(frac * x.size)
	shuffled = sp.random.permutation(list(range(len(x))))
	train = sorted(shuffled[:split_idx])
	test = sorted(shuffled[split_idx:])
	
	function = sp.poly1d(sp.polyfit(x[train],y[train],deg))
	#print(function)
	
	print("f%i  :error_value=%f" % (deg,error_value(function,x[test],y[test])))
	
print("error_value!!")
random_test(x_b,y_b,1)
random_test(x_b,y_b,2)
random_test(x_b,y_b,3)
random_test(x_b,y_b,10)

#プロット
import matplotlib.pyplot as plt

#散布図
plt.scatter(x,y)

#各曲線
f1a_x_a = sp.linspace(0,x_a[-1]+100)#プロットようにx値を生成
plt.plot(f1a_x_a,f1a(f1a_x_a),linewidth=4)

f1b_x_b = sp.linspace(inflection1-100,x_b[-1])#プロットようにx値を生成
plt.plot(f1b_x_b,f1b(f1b_x_b),'--',linewidth=4)

#プロットの表示設定
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)]) #x座標に対しての配置と名前
plt.autoscale(tight=True) #自動で画面をスケール調整する
plt.grid() #グリッドをつける
plt.legend(["f1a","f1b"],loc="upper left")#左上に曲線の名前を配置
plt.show() #プロットを表示する。