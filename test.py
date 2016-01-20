# coding=utf8

import scipy as sp

data = sp.genfromtxt("data/web_traffic.tsv",delimiter="\t")
#print(data[:10])
#print(data.shape)

#ベクトルに分割
x = data[:,0] #経過時間(一時間単位)とする
y = data[:,1] #アクセス数とする
#print(x[:10])
#print(y[:10])

#不適切なデータの取り除き
#print(sp.sum(sp.isnan(y))) #不適切データの数
x = x[~sp.isnan(y)]#isNaNがTrueではない要素だけに
y = y[~sp.isnan(y)]
#print(x.shape)
#print(y.shape)


#単純な直線で近値
print("f1!!")
f1_parameters ,f1_residuals, f1_rank, f1_singular_values, f1_rcond = sp.polyfit(x,y,1,full=True)
print([f1_parameters,f1_residuals, f1_rank, f1_singular_values, f1_rcond])
f1 = sp.poly1d(f1_parameters)
#print(f1)

#二次曲線
print("f2!!")
f2_parameters ,f2_residuals, f2_rank, f2_singular_values, f2_rcond = sp.polyfit(x,y,2,full=True)
print([f2_parameters,f2_residuals, f2_rank, f2_singular_values, f2_rcond])
f2 = sp.poly1d(f2_parameters)
#print(f2)

#三次曲線
print("f3!!")
f3_parameters ,f3_residuals, f3_rank, f3_singular_values, f3_rcond = sp.polyfit(x,y,3,full=True)
print([f3_parameters,f3_residuals, f3_rank, f3_singular_values, f3_rcond])
f3 = sp.poly1d(f3_parameters)
#print(f3)

#多項式曲線
print("f30!!")
f30_parameters ,f30_residuals, f30_rank, f30_singular_values, f30_rcond = sp.polyfit(x,y,30,full=True)
print([f30_parameters,f30_residuals, f30_rank, f30_singular_values, f30_rcond])
f30 = sp.poly1d(f30_parameters)
#print(f30)

#近似誤差値
def error_value(f, x, y): #目的関数を利用
	return (sp.sum((f(x) - y) ** 2)/(2 * x.size))

def random_test(x,y,deg): #訓練用データとテスト用データにランダムに分ける
	frac = 0.3 #データの割合
	split_idx = int(frac * x.size)
	shuffled = sp.random.permutation(list(range(len(x))))
	test = sorted(shuffled[:split_idx])
	train = sorted(shuffled[split_idx:])
	
	function = sp.poly1d(sp.polyfit(x[train],y[train],deg))
	#print(function)
	
	print("f%i  :error_value=%f" % (deg,error_value(function,x[test],y[test])))
	
print("error_value!!")
random_test(x,y,1)
random_test(x,y,2)
random_test(x,y,3)
random_test(x,y,10)
random_test(x,y,30)

#Matplotlib pyplot
print("Matplotlib pyplot")

import matplotlib.pyplot as plt

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



