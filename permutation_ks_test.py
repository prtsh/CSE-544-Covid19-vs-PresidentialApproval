import numpy as np
from numpy import ma
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
from datetime import datetime
from scipy.stats import binom
from scipy.stats import geom
from scipy.stats import poisson
from scipy import stats


def permutation_test(list1, list2, no_permutations):
	N = no_permutations
	T_obs = abs(np.mean(list1) - np.mean(list2))

	counter = 0
	l1=[]

	for i in range(N):
		list1_i = np.random.choice(list1 + list2, size = len(list1), replace = False)
		list2_i = np.random.choice(list1 + list2, size = len(list2), replace = False)
		list1_i_mean = np.mean(list1_i)
		list2_i_mean = np.mean(list2_i)
		T_i = abs(list1_i_mean - list2_i_mean)
		l1.append(T_i)

		
		if T_i > T_obs:
			counter += 1


	p_val = counter/N
	return p_val

def onesamp_KS_test(data1,data2):
	data1 = ma.asarray(data1)
	data2 = ma.asarray(data2)
	n1  = data1.count()
	n2 =  data2.count()
	X = data1.compressed()
	Y = data2.compressed()
	mix = ma.concatenate((X, Y))
	mixsort = mix.argsort(kind='mergesort')
	csum = np.where(mixsort < n1, 1./n1, -1./n2).cumsum()


	length = len(np.unique(mix))
	if length < (n1+n2):
	    ind = np.r_[np.diff(mix[mixsort]).nonzero()[0], -1]
	    csum = csum[ind]
	    mixsort = mixsort[ind]

	csumabs = ma.abs(csum)
	i = csumabs.argmax()

	d = abs(csum[i])
	x = mix[mixsort[i]]

	return d, x

def twosamp_KS_test(data1, data2):
	data1 = ma.asarray(data1)
	data2 = ma.asarray(data2)
	n1  = data1.count()
	n2 =  data2.count()
	X = data1.compressed()
	Y = data2.compressed()
	mix = ma.concatenate((X, Y))
	mixsort = mix.argsort(kind='mergesort')
	csum = np.where(mixsort < n1, 1./n1, -1./n2).cumsum()



	length = len(np.unique(mix))
	if length < (n1+n2):
	    ind = np.r_[np.diff(mix[mixsort]).nonzero()[0], -1]
	    csum = csum[ind]
	    mixsort = mixsort[ind]

	csumabs = ma.abs(csum)
	i = csumabs.argmax()

	d = abs(csum[i])
	x = mix[mixsort[i]]

	return d, x

# def make_plot(list1, list2, point, KS_stat, flag):
# 	S = list1
# 	n = len(S)
# 	Srt = sorted(S)
# 	y_req_1 = 0
# 	y_req_2 = 0
# 	delta = .1
# 	X = [min(Srt)-delta]
# 	Y = [0]
# 	for i in range(0, n):
# 	    X = X + [Srt[i], Srt[i]]
# 	    Y = Y + [Y[len(Y)-1], Y[len(Y)-1]+(1/n)]
# 	    if X[i] == point:
# 	    	y_req_1 = Y[i]
#
# 	X = X + [max(Srt)+delta]
# 	Y = Y + [1]
#
# 	R = list2
# 	n1 = len(R)
# 	Srt1 = sorted(R)
# 	W = [min(Srt1)-delta]
# 	Z = [0]
# 	for i in range(0, n1):
# 	    W = W + [Srt1[i], Srt1[i]]
# 	    Z = Z + [Z[len(Z)-1], Z[len(Z)-1]+(1/n1)]
# 	    if W[i] == point:
# 	    	y_req_2 = Z[i]
#
# 	W = W + [max(Srt1)+delta]
# 	Z = Z + [1]
#
# 	if flag:
# 		plt.figure('eCDF')
# 		plt.plot(X, Y ,label='eCDF of 2019')
# 		plt.plot(W, Z ,label='eCDF of 2009')
# 		plt.vlines(point, y_req_1 + delta/2.7, y_req_1 + KS_stat, colors='green', \
# 			linestyles='solid', label = 'max distance of {0:.4f}'.format(KS_stat))
# 		plt.xlabel('x')
# 		plt.ylabel('Pr[X<=x]')
# 		plt.legend(loc="upper left")
# 		plt.grid()
# 		plt.show()
# 	else:
# 		plt.figure('eCDF')
# 		plt.plot(X, Y ,label='eCDF of 2009')
# 		plt.plot(W, Z ,label='eCDF of 1999')
# 		plt.vlines(point, y_req_1 + delta + 0.03, y_req_1 + delta + KS_stat, colors='green', \
# 			linestyles='solid', label = 'max distance of {0:.4f}'.format(KS_stat))
# 		plt.xlabel('x')
# 		plt.ylabel('Pr[X<=x]')
# 		plt.legend(loc="upper left")
# 		plt.grid()
# 		plt.show()

def main():

	X = pd.read_csv("us.csv")
	X_second_last = X[-15:-8]
	X_last = X[-8:-1]

	case1 = list(X_second_last['cases'])
	case2 = list(X_last['cases'])
	death1 = list(X_second_last['deaths'])
	death2 = list(X_last['deaths'])

# ---------------------------------- Permutation test -----------------------------------------------
	print("**************************************************************************************")
	print("Permutation Test")

	print("-----------  1: cases for N = 20 permutations --------------")
	p_value = permutation_test(case1, case2, 20)
	print("p_value for case 2019 vs 2009 for N = 20 is:", p_value)

	if p_value < 0.05:
		print("Null hypothesis is rejected because p_value < 0.05")
	else:
		print("Null hypothesis is accepted because p_value > 0.05")


	print("-----------  2: deaths for N = 20 permutations --------------")
	p_value = permutation_test(death1, death2, 20)
	print("p_value for case 2019 vs 2009 for N = 20 is:", p_value)

	if p_value < 0.05:
		print("Null hypothesis is rejected because p_value < 0.05")
	else:
		print("Null hypothesis is accepted because p_value > 0.05")


# ----------------------------------- 2 Sampled KS Test -----------------------------------------------
	print("**************************************************************************************")
	print("2-Sampled KS Test")
	print("----------------- Case 1: 2 Sampled KS_test for cases -----------------------")
	KS_stat, point = twosamp_KS_test(case1, case2)
	print("KS_statistic for case 1 is {} at point X = {} ".format(KS_stat, point))

	#make_plot(X, Y, point, KS_stat, 1)

	if KS_stat > 0.05:
		print("Null hypothesis is rejected because KS_stat > 0.05")
	else:
		print("Null hypothesis is accepted because KS_stat < 0.05")


	print("----------------- Case 2: 2 Sampled KS_test for deaths -----------------------")
	KS_stat, point = twosamp_KS_test(death1, death2)
	print("KS_statistic for case 2 is {} at point X = {} ".format(KS_stat, point))
	#make_plot(Y, Z, point, KS_stat, 0)

	if KS_stat > 0.05:
		print("Null hypothesis is rejected because KS_stat > 0.05")
	else:
		print("Null hypothesis is accepted because KS_stat < 0.05")


# ----------------------------------- 1 Sampled KS Test -----------------------------------------------
	print("**************************************************************************************")
	print("1-Sampled KS Test")

	print("----------------- Case 1: 1 Sampled KS_test for cases -----------------------")

	##Geometric:
	MMEparamGeometric_p = (len(case1)*len(case1))/sum(case1)
	print("Geometric:")
	cdf_geometric=geom.cdf(case1, MMEparamGeometric_p)
	# print(cdf_geometric)
	#print(stats.kstest(case2, 'geom', args=(MMEparamGeometric_p)))
	KS_stat, point = onesamp_KS_test(cdf_geometric, case2)
	print("KS_statistic for case 1 is {} at point X = {} ".format(KS_stat, point))
	if KS_stat > 0.05:
		print("Null hypothesis is rejected because KS_stat > 0.05")
	else:
		print("Null hypothesis is accepted because KS_stat < 0.05")

	##Binomial:
	MMEparamBinomial_p = sum(case1) / (len(case1) * len(case1))
	print("Binomial:")
	cdf_binomial= binom.cdf(case1,len(case1), MMEparamBinomial_p)
	#print(cdf_binomial)
	#print(stats.kstest(case2, 'binom', args=(len(case1), MMEparamBinomial_p)))
	KS_stat, point = onesamp_KS_test(cdf_binomial, case2)
	print("KS_statistic for case 1 is {} at point X = {} ".format(KS_stat, point))
	if KS_stat > 0.05:
		print("Null hypothesis is rejected because KS_stat > 0.05")
	else:
		print("Null hypothesis is accepted because KS_stat < 0.05")

	##poisson:
	MMEparamPoisson_lambda = sum(case1) / len(case1)
	print("poisson:")
	cdf_poisson=poisson.cdf(case1, MMEparamPoisson_lambda)
	# print(cdf_poisson)
	# #print(stats.kstest(case2, 'poisson', args=(MMEparamPoisson_lambda)))
	KS_stat, point = onesamp_KS_test(cdf_poisson, case2)
	print("KS_statistic for case 1 is {} at point X = {} ".format(KS_stat, point))
	if KS_stat > 0.05:
		print("Null hypothesis is rejected because KS_stat > 0.05")
	else:
		print("Null hypothesis is accepted because KS_stat < 0.05")








	print("----------------- Case 2: 1 Sampled KS_test for deaths -----------------------")
	##Geometric:
	MMEparamGeometric_p = (len(death1) * len(death1)) / sum(death1)
	print("Geometric:")
	cdf_geometric=geom.cdf(death1, MMEparamGeometric_p)
	# print(cdf_geometric)
	# print(stats.kstest(death2, 'geom', args=(MMEparamGeometric_p)))
	KS_stat, point = onesamp_KS_test(cdf_geometric, death2)
	print("KS_statistic for case 1 is {} at point X = {} ".format(KS_stat, point))
	if KS_stat > 0.05:
		print("Null hypothesis is rejected because KS_stat > 0.05")
	else:
		print("Null hypothesis is accepted because KS_stat < 0.05")

	##Binomial:
	MMEparamBinomial_p = sum(death1) / (len(death1) * len(death1))
	print("Binomial:")
	cdf_binomial=binom.cdf(death1,len(death1), MMEparamBinomial_p)
	# print(cdf_binomial)
	# print(stats.kstest(death2, 'binom', args=(len(death1), MMEparamBinomial_p)))
	KS_stat, point = onesamp_KS_test(cdf_binomial, death2)
	print("KS_statistic for case 1 is {} at point X = {} ".format(KS_stat, point))
	if KS_stat > 0.05:
		print("Null hypothesis is rejected because KS_stat > 0.05")
	else:
		print("Null hypothesis is accepted because KS_stat < 0.05")

	##poisson:
	MMEparamPoisson_lambda = sum(death1) / len(death1)
	print("poisson:")
	cdf_poisson=poisson.cdf(death1, MMEparamPoisson_lambda)
	# print(cdf_poisson)
	# #print(stats.kstest(death2, 'poisson', args=(MMEparamPoisson_lambda)))
	KS_stat, point = onesamp_KS_test(cdf_poisson, death2)
	print("KS_statistic for case 1 is {} at point X = {} ".format(KS_stat, point))
	if KS_stat > 0.05:
		print("Null hypothesis is rejected because KS_stat > 0.05")
	else:
		print("Null hypothesis is accepted because KS_stat < 0.05")


main()


