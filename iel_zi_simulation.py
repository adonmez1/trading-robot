#This is like a G-S simulation but
# one can have buyers or sellers, zi or iel, book or no book.
# it uses the traditional IEL with a simple foregone utility function

#this measures distance to humans

import numpy as np
import statistics
from scipy.stats import truncnorm
import math	
import sys
import time
import random
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import spearmanr
import pandas as pd
import datetime


dt=str(datetime.datetime.now())

directory = "your-directory"
resultfilename = "your-directory/results.txt"

simlines = []  # This is what will ultimately be printed to the Results file

simlines.append("This file collects information about simulations run on " + dt + "\n")
simlines.append("\n")

""" Simulation parameters:"""

sim = 1000  # number of simulations
Draws=[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100] #This lists the range of the number of draws
book = 1

Simintell = [0, 1]  # simintell = 0 if all zi, =1 if all iel

SU = [300] #upper-bound bid limit
simlines.append("The simulation parameters are " + "\n" + "\n")
simlines.append("The number of runs per simulation is: " + str(sim) + "\n")
simlines.append("The range of the number of draws is: " + str(Draws) + "\n")
simlines.append("The range of intelligence types is: " + str(Simintell) + "\n")
simlines.append("The range of upper bounds, su, is :" + str(SU) + "\n")

"""Environment"""

N = 10  # number of traders
atype = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0=buyer, 1= seller

nunits = 3  # max amount buyer/seller can BUY/SELL

avalue = [[150, 135, 120], [147, 132, 117], [144, 129, 114], [141, 126, 111], [138, 123, 108],
		  [70, 95, 120], [65, 90, 115], [60, 85, 110], [55, 80, 105], [50, 75, 100]]

sl = -1  # lower bound on costs and values
pl = 110  # this is the lowest equilibrium price
pu = 114  # this is the largest equilibrium price
prange = [pl, pu]
eqp = 112  # this is used as the equilibrium price in xm-inefficiency
qopt = 13  # this is the equilibrium quantity.
Max = 676  # this is the maximum possible surplus given avalue()
optbuypi = 260  # this is the buyers surplus at the optimum
optsellpi = 416  # this is the sellers surplus at the optimum

"""Extracts marginal utility and marginal cost values for later computations """
V = []
C = []
for i in range(N):
	if atype[i] == 0:
		V.append(avalue[i])
	if atype[i] == 1:
		C.append(avalue[i])
rankedV = 16 - rankdata(V)
rankedC = rankdata(C)
rankedvalues = np.append(rankedV, rankedC)

"""Prints out the information about the environment"""

typename = ["a"] * N
intellname = ["a"] * N

for i in range(N):
	typename[i] = "BUYER"
	if atype[i] == 1:
		typename[i] = "SELLER"
simlines.append("\n")
simlines.append("The environment is:" + "\n")
simlines.append("\n")
simlines.append("There are " + str(N) + " traders, each with " + str(nunits) + " to trade" + "\n")

for i in range(N):
	simlines.append("trader" + str(i) + "is a " + typename[i] + " with values " + str(avalue[i]) + "\n")

simlines.append("\n")
simlines.append("The competitive equilibrium prices are " + str(prange) + "\n")
simlines.append("The competitive equilibrium quantity is " + str(qopt) + "\n")
simlines.append("The maximum social surplus is " + str(Max) + "\n")
simlines.append(
	"The equilibrium surplus for buyers is " + str(optbuypi) + " and for sellers is " + str(optsellpi) + "\n")
simlines.append("\n")

"""################   ZI behavior"""


def getziorder(agent):
	if atype[agent] == 0:
		bid = sl + (random.random() * avalue[agent][h[agent]] - sl)
	else:
		bid = avalue[agent][h[agent]] + random.random() * (su - avalue[agent][h[agent]])
	return int(bid)

"""##################  IEL behavior"""

#### IEL parameters


K = 1
J = 100
muv = .033
mul = .0033

###### IEL agent initialization

def initializeIEL(agent):
	for j in range(J):
		strategies[agent].append(random.randint(0, su))
		utilities[agent].append(1)


####### IEL bid calculation

def getielorder(agent):
	"""
	Takes a request for a either a buy or sell order, the type of which is
	specified by action. Returns an order.

	:param order_side: either OrderSide.BUY or OrderSide.SELL
	"""
	action = atype[agent]
	valuation = avalue[agent][h[agent]]
	removeirrational(action, agent, valuation)
	Vexperimentation(action, agent, valuation)
	updateW(action, agent, valuation)
	replicate(action, agent)
	return strat_selection(action, agent, valuation)


def removeirrational(action, agent, valuation):
	if action == 0:
		strategies[agent] = list(filter(lambda x: x < valuation, strategies[agent]))
	else:
		strategies[agent] = list(filter(lambda x: x > valuation, strategies[agent]))

	if len(strategies[agent]) < J:
		x = J - len(strategies[agent])
		for i in range(x):
			if action == 0:
				strategies[agent].append(int(random.uniform(sl, valuation)))
			if action == 1:
				strategies[agent].append(int(random.uniform(valuation, su)))


def choiceprobabilities(agent):
	"""
	Calculates the probability of choosing a strategy for all strategies of
	the same action. Probability is proportional to the foregone utility of
	one strategy divided by the sum of foregone utilities over all
	strategies.

	:param action: 0 corresponds to buying, 1 corresponds to selling
	"""
	choicep = []
	sumw = sum(utilities[agent])
	if sumw == 0:
		return np.zeros(J)
	for j in range(J):
		choicep.append(utilities[agent][j] / float(sumw))
	return choicep


def strat_selection(action, agent, valuation):
	"""
	Chooses a strategy out of all strategies of the same action.

	:param action: 0 corresponds to buying, 1 corresponds to selling
	"""

	choicep = choiceprobabilities(agent)

	if sum(choicep) == 0:
		choicep = [1 / len(strategies[agent]) for x in strategies[agent]]
	bid = int(rand_choice(strategies[agent], choicep))
	return bid


def rand_choice(items, distr):
	x = random.uniform(0, 1)
	cumulative_probability = 0.0
	saved_item = 0
	for item, item_probability in zip(items, distr):
		cumulative_probability += item_probability
		if x < cumulative_probability:
			saved_item = item
			break
	return saved_item


def Vexperimentation(action, agent, valuation):
	"""
	Value experimentation for strategies of the same action. With a
	probability determined by muv, takes a strategy as a center of a
	distribution and generates a new strategy around the center.

	:param action: 0 corresponds to buying, 1 corresponds to selling
	"""
	for j in range(J):
		if atype == 0:
			sigmav = max(1, 0.1 * (valuation - sl))
			if random.uniform(0, 1) < muv:
				centers = strategies[agent][j]
				r = (truncnorm.rvs((sl - centers) / float(sigmav),
								   (valuation - centers) / float(sigmav),
								   loc=centers, scale=sigmav, size=1))
				strategies[agent][j] = int(np.array(r).tolist()[0])
		if atype == 1:
			sigmav = max(1, 0.1 * (su - valuation))
			if random.uniform(0, 1) < muv:
				centers = strategies[agent][j]
				r = (truncnorm.rvs((valuation - centers) / float(sigmav),
								   (su - centers) / float(sigmav),
								   loc=centers, scale=sigmav, size=1))
				strategies[agent][j] = int(np.array(r).tolist()[0])


def replicate(agent, action):
	"""
	Replicates strategies of the same action by comparing two randomly
	chosen strategies and replacing that with the lower utility with
	the other strategy.

	:param action: 0 corresponds to buying, 1 corresponds to selling
	"""
	for j in range(J):
		j1 = random.randrange(J)
		j2 = random.randrange(J)
		strategies[agent][j] = strategies[agent][j2]
		utilities[agent][j] = utilities[agent][j2]
		if utilities[agent][j1] > utilities[agent][j2]:
			strategies[agent][j] = strategies[agent][j1]
			utilities[agent][j] = utilities[agent][j1]


def foregone_utility(j, action, agent, valuation):
	"""
	Calculates the foregone utility of a strategy with index j and
	corresponding to a certain action. Currently, we only consider the
	current book and look at the current highest bid and current lowest
	offer.

	:param j: The index of the strategy for which we want to update
	the foregone utility.

	:param action: 0 corresponds to buying, 1 corresponds to selling
	"""
	if Qb == []:
		curr_best_bid = sl
	else:
		curr_best_bid = Qb[0][0]

	if Qs == []:
		curr_best_offer = su
	else:
		curr_best_offer = Qs[0][0]

	if action == 0:
		x1 = strategies[agent][j]
		# Return 0 for case where bid exceeds the valuation of item to be bought
		if x1 <= curr_best_offer or h[agent] >= H[agent] or x1 > valuation:
			return 0
		else:
			return valuation - curr_best_offer
	else:
		x1 = strategies[agent][j]
		# Return 0 for case where the offer is lower than the valuation of item to be sold
		if x1 >= curr_best_bid or h[agent] >= H[agent] or x1 < valuation:
			return 0
		else:
			return curr_best_bid - valuation


def updateW(action, agent, valuation):
	"""
	Updates the foregone utilities for strategies of the same action.

	:param action: 0 corresponds to buying, 1 corresponds to selling
	"""
	for j in range(J):
		utilities[agent][j] = foregone_utility(j, action, agent, valuation)


# end of iel bid calculation.

pricerecord = []
simrecord = []
dgridrecord = []
egridrecord = []
pgridrecord = []
edgridrecord = []
Head = ["su"]
for t in Draws:
	Head.append(math.ceil(t))
dgridrecord.append(Head)
egridrecord.append(Head)
pgridrecord.append(Head)
edgridrecord.append(Head)

price_history = []
draw_count = []
sim_count_history = []
intelligence = []
smith_alpha = []
buyer_valuation = []
seller_valuation = []
buyer_profit = []
seller_profit = []
transaction_order = []
eff = []
smith_alp = []
for simintell in Simintell:

	# Agent intelligence:  0=ZI, 1= IEL
	aintell = ["NA"] * N
	if simintell == 0:
		aintell = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	if simintell == 1:
		aintell = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

	# for book in Book:
	for su in SU:
		dgrid = [su]
		egrid = [su]
		pgrid = [su]
		edgrid = [su]

		for T in Draws:

			print()
			simlines.append("\n")
			simlines.append(" Intelligence code =" + str(simintell) + "   SU =" + str(su) + "  ")
			if book == 1:
				print("BOOK, ", end="")
				simlines.append("Book ")
			else:
				print("NO BOOK, ", end="")
				simlines.append("NO BOOK ")
			print("T=", T, ", ", end="")
			simlines.append(" T=" + str(T) + "\n")
			print(" Intelligence code =", simintell)
			print("su =", su)

			print()
			simlines.append("\n")

			x = []
			y = []
			z = []
			w = []
			buypi = []
			sellpi = []
			zipi = []
			ielpi = []
			bcorr = []
			scorr = []
			prices = []
			aveprice = []
			enddrawrec = []
			for g in range(sim):

				soc = 0
				quantity = 0
				buyprofit = 0
				sellprofit = 0
				ziprofit = 0
				ielprofit = 0
				tempprice = []
				tx_order = []
				IB = 0
				IS = 0
				EB = 0
				ES = 0
				prices = []
				sim_count_temp = []
				buyer_val = []
				seller_val = []
				pos_buyer_surplus = 0
				neg_buyer_surplus = 0
				pos_seller_surplus = 0
				neg_seller_surplus = 0
				BSD = 0
				SSD = 0
				Qb = [[-1, -1]]  # bid queue for book
				Qs = [[su + 1, -1]]  # ask queue for book

				"""initialize agents	at beginning of a sim"""
				strategies = []
				utilities = []
				H = [nunits] * N  # max amount buyer.seller can buy/sell
				h = [0] * N  # units bought/sold

				for i in range(N):
					strategies.append([])
					utilities.append([])

				"""initialize IEL strategies and utilities at beginning of sim"""
				for i in range(N):
					initializeIEL(i)

				"""#this leaves S and W empty for ZI and moblab"""

				for t in range(T):

					"""#Determines who bids next"""
					Nb = random.randint(0, N - 1)

					""" Next step is to determine the bids """
					if h[Nb] < H[Nb]:

						Qs.sort(key=lambda x: x[0])
						Qb.sort(key=lambda x: x[0], reverse=True)

						if aintell[Nb] == 0:
							bid = getziorder(Nb)

						# if agent is IEL
						if aintell[Nb] == 1:
							bid = getielorder(Nb)

						if book == 0:
							if atype[Nb] == 0 and bid > Qb[0][0]:
								Qb = [[bid, Nb]]
							if atype[Nb] == 1 and bid < Qs[0][0]:
								Qs = [[bid, Nb]]

						if book == 1:

							if atype[Nb] == 0 and Qb != []:
								for j in Qb:
									if j[1] == Nb:
										Qb.remove(j)
								Qb.append([bid, Nb])

							if atype[Nb] == 1 and Qs != []:
								for j in Qs:
									if j[1] == Nb:
										Qs.remove(j)
								Qs.append([bid, Nb])

					Qb.sort(key=lambda x: x[0], reverse=True)
					Qs.sort(key=lambda x: x[0])

					"""Checks the book to see if a trade is possible and calculates stats if so."""
					if Qb != [] and Qs != [] and Qb[0][0] >= Qs[0][0]:
						buyer = Qb[0][1]
						seller = Qs[0][1]

						soc = soc + avalue[buyer][h[buyer]] - avalue[seller][h[seller]]

						bcorr.append([quantity, rankedvalues[3 * buyer + h[buyer]]])
						scorr.append([quantity, rankedvalues[3 * seller + h[seller]]])

						if atype[Nb] == 0:
							price = Qs[0][0]
						else:
							price = Qb[0][0]
						tempprice.append(price)

						buyer_val.append(avalue[buyer][h[buyer]])
						seller_val.append(avalue[seller][h[seller]])

						buyprofit += avalue[buyer][h[buyer]] - price
						sellprofit += price - avalue[seller][h[seller]]

						if avalue[buyer][h[buyer]] - eqp >= 0:
							pos_buyer_surplus += avalue[buyer][h[buyer]] - eqp
							IB += 1
						else:
							neg_buyer_surplus += eqp - avalue[buyer][h[buyer]]
							EB += 1
						if eqp - avalue[seller][h[seller]] >= 0:
							pos_seller_surplus += eqp - avalue[seller][h[seller]]
							IS += 1
						else:
							neg_seller_surplus += avalue[seller][h[seller]] - eqp
							ES += 1

						if aintell[buyer] == 0:
							ziprofit += avalue[buyer][h[buyer]] - price
						else:
							ielprofit += avalue[buyer][h[buyer]] - price

						if aintell[seller] == 0:
							ziprofit += price - avalue[seller][h[seller]]
						else:
							ielprofit += price - avalue[seller][h[seller]]

						if book == 1:
							Qb.pop(0)
							Qs.pop(0)

						else:
							Qb = [[-1, -1]]  # is this ok? buyer 0 is an artifact here.
							Qs = [[su + 1, -1]]
						tx_order.append(quantity+1)

						quantity += 1
						h[buyer] += 1
						h[seller] += 1
						enddraw = t

				"""collects results for T"""
				prices.append(tempprice)
				price_history.append(tempprice)
				sim_count_history.append(len(tempprice) * [g + 1])
				draw_count.append(len(tempprice) * [T])
				intelligence.append(len(tempprice) * [simintell])
				buyer_valuation.append(buyer_val)
				seller_valuation.append(seller_val)
				buyer_profit.append(len(tempprice) * [buyprofit])
				seller_profit.append(len(tempprice) * [sellprofit])
				tempprice2 = [x - eqp for x in tempprice]
				transaction_order.append(tx_order)
				try:
					smith_alpha.append(math.sqrt(sum(x ** 2 for x in tempprice2) / len(tempprice2)) / eqp)
				except:
					pass
				if tempprice != []:
					aveprice.append(math.ceil(statistics.mean(tempprice)))
				buypi.append(math.ceil(buyprofit))
				sellpi.append(math.ceil(sellprofit))
				zipi.append(math.ceil(ziprofit))
				ielpi.append(math.ceil(ielprofit))
				x.append(soc)
				y.append(quantity)
				enddrawrec.append(enddraw)

				""" computes the extra marginal efficiency index to one sim """

				MIB = qopt - IB
				BSD = optbuypi - pos_buyer_surplus
				MIS = qopt - IS
				SSD = optsellpi - pos_seller_surplus
				e1 = soc / Max
				eff.append(len(tempprice) * [e1])
				smith_alp.append(len(tempprice) * [sum(x for x in smith_alpha) / len(smith_alpha)])
				if e1 != 1:
					if quantity >= qopt:
						emi = 1
					elif quantity < qopt and EB == 0 and ES == 0:
						emi = 0
					else:
						vsb = BSD * (MIB - EB) / MIB
						vss = SSD * (MIS - ES) / MIS
						emsb = BSD - vsb + neg_buyer_surplus
						emss = SSD - vss + neg_seller_surplus
						ems = emsb + emss
						vs = vss + vsb
						emi = ems / (ems + vs)
				z.append(emi)
			# print("extra-marginal inefficiency is: " + str(emi))

			""" createa a histogram of efficiency for T=60 """
			flat_list = [item for sublist in price_history for item in sublist]
			flat_list2 = [item for sublist in sim_count_history for item in sublist]
			flat_list3 = [item for sublist in draw_count for item in sublist]
			flat_list4 = [item for sublist in intelligence for item in sublist]
			flat_list5 = [item for sublist in buyer_valuation for item in sublist]
			flat_list6 = [item for sublist in seller_valuation for item in sublist]
			flat_list7 = [item for sublist in buyer_profit for item in sublist]
			flat_list8 = [item for sublist in seller_profit for item in sublist]
			flat_list9 = [item for sublist in transaction_order for item in sublist]
			flat_list10 = [item for sublist in eff for item in sublist]
			flat_list11 = [item for sublist in smith_alp for item in sublist]

			beta = 100  # bins

			e = [i / float(Max) for i in x]

			if T == 60:
				(m, bins, patches) = plt.hist(e, bins=beta, range=[0, 1], label='eff')

			""" Prints out information about the results for T"""

			simlines.append("Efficiency: " + "min E =" + str(100 * math.ceil(1000 * min(e)) / 1000) + ", max E =" + str(
				100 * math.ceil(1000 * max(e)) / 1000) + ", Median(E) =" + str(
				100 * math.ceil(1000 * statistics.median(e)) / 1000) + ", E(E) = " + str(
				100 * math.ceil(1000 * sum(e) / float(len(e))) / 1000) + "\n")

			simlines.append("Smith's alpha: " + "min a =" + str(
				100 * math.ceil(1000 * min(smith_alpha)) / 1000) + ", max a =" + str(
				100 * math.ceil(1000 * max(smith_alpha)) / 1000) + ", Median(a) =" + str(
				100 * math.ceil(1000 * statistics.median(smith_alpha)) / 1000) + ", E(a) = " + str(
				100 * math.ceil(1000 * sum(smith_alpha) / float(len(smith_alpha))) / 1000) + "\n")

			simlines.append("Quantity: " + "min q =" + str(min(y)) + ", max q =" + str(max(y)) + ", median(q) =" + str(
				statistics.median(y)) + ", E(q) = " + str(math.ceil(10 * sum(y) / float(len(y))) / 10) + "\n")
			simlines.append(
				"Buy Profit: " + "min  =" + str(min(buypi)) + ", max  =" + str(max(buypi)) + ", median =" + str(
					statistics.median(buypi)) + ", E(buyprofit)= " + str(
					math.ceil(10 * sum(buypi) / float(len(buypi))) / 10) + "\n")
			simlines.append(
				"Sell Profit: " + "min  =" + str(min(sellpi)) + ", max  =" + str(max(sellpi)) + ", median =" + str(
					statistics.median(sellpi)) + ", E(sellprofit)= " + str(
					math.ceil(10 * sum(sellpi) / float(len(sellpi))) / 10) + "\n")
			simlines.append(
				"IEL Profit: " + "min  =" + str(min(ielpi)) + ", max  =" + str(max(ielpi)) + ", median =" + str(
					statistics.median(ielpi)) + ", E(IELprofit)= " + str(
					math.ceil(10 * sum(ielpi) / float(len(ielpi))) / 10) + "\n")
			simlines.append(
				"ZI Profit: " + "min  =" + str(min(zipi)) + ", max  =" + str(max(zipi)) + ", median =" + str(
					statistics.median(zipi)) + ", E(ZIprofit)= " + str(
					math.ceil(10 * sum(zipi) / float(len(zipi))) / 10) + "\n")
			if prices != []:
				simlines.append("Average Price: " + "min  =" + str(min(aveprice)) + ", max  =" + str(
					max(aveprice)) + ", median =" + str(
					math.ceil(statistics.median(aveprice))) + ", E(average price)= " + str(
					math.ceil(10 * sum(aveprice) / float(len(aveprice))) / 10) + "\n")
			simlines.append(
				"XM inefficiency: " + "min  =" + str(min(z)) + ", max  =" + str(max(z)) + ", median =" + str(
					statistics.median(z)) + ", E(XM)= " + str(math.ceil(10000 * sum(z) / float(len(z))) / 10000) + "\n")
			simlines.append(
				"EndDraw: " + "min  =" + str(min(enddrawrec)) + ", max  =" + str(max(enddrawrec)) + ", median =" + str(
					statistics.median(enddrawrec)) + ", E(EndDraw)= " + str(
					math.ceil(10 * sum(enddrawrec) / float(len(enddrawrec))) / 10) + "\n")

			bresult = "na"
			sresult = "na"
			try:
				bresult, _ = spearmanr(bcorr)

				simlines.append(
					"Correlation Coefficient between buyer value and trade order" + str('%.3f' % bresult) + "\n")
				sresult, _ = spearmanr(scorr)

				simlines.append(
					"Correlation Coefficient between seller cost and trade order" + str('%.3f' % sresult) + "\n")
			except:
				pass

			# computes "distance" from human data on efficiency, 87, and price, 113.1.
			print(statistics.mean(e), statistics.mean(aveprice))
			print(100 * statistics.mean(e) - 87, statistics.mean(aveprice) - 113.1)
			print((100 * statistics.mean(e) - 87) ** 2 + (statistics.mean(aveprice) - 113.1) ** 2)

			print("E(EndDraw)= ", str(math.ceil(10 * sum(enddrawrec) / float(len(enddrawrec))) / 10))
			distance = round(((100 * statistics.mean(e) - 87) ** 2 + (statistics.mean(aveprice) - 113.1) ** 2), 1)

			print("distance = ", distance)

			dgrid.append(distance)
			egrid.append(100 * math.ceil(1000 * sum(e) / float(len(e))) / 1000)
			pgrid.append(math.ceil(10 * sum(aveprice) / float(len(aveprice))) / 10)
			edgrid.append(math.ceil(10 * sum(enddrawrec) / float(len(enddrawrec))) / 10)

			simlines.append("The distance between sim and human is " + str(distance) + "\n")

			"""collects the simulation results to save to a csv file"""

			simrecord2 = [simintell, book, T, su, 100 * math.ceil(1000 * sum(e) / float(len(e))) / 1000,
						  round(sum(aveprice) / float(len(aveprice)), 1), distance]

			simrecord.append(simrecord2)

			""" collects data on prices and price averages"""

			z3 = []
			for i in prices:
				z3.append(len(i))
			M = max(z3)

			z1 = [simintell, book, T]

			for k in range(M):
				tp = []
				for i in range(len(prices)):
					if len(prices[i]) > k:
						tp.append(prices[i][k])
				ave = round(sum(tp) / len(tp), 1)
				z1.append(ave)

			pricerecord.append(z1)

		dgridrecord.append(dgrid)
		egridrecord.append(egrid)
		pgridrecord.append(pgrid)
		edgridrecord.append(edgrid)

timestr = time.strftime("%Y%m%d-%H%M%S")
history_of_prices = pd.DataFrame(
        list(zip(flat_list4, flat_list3, flat_list2, flat_list, flat_list5, flat_list6, flat_list7, flat_list8, flat_list9, flat_list10, flat_list11)),
        columns=['intelligence', 'draw_count', 'sim_number', 'trade_price', 'buyer_val', 'seller_val', 'buyer_profit', 'seller_profit', 'tx_order', 'efficiency', 'smith_alpha'])
filename1 = directory + timestr + ".csv"
history_of_prices.to_csv(filename1, index=False, encoding='utf-8')

pricefile = pd.DataFrame(pricerecord)
pricefile.to_csv(directory+'/pricerecord.csv')

pricefile=pd.DataFrame(pricerecord)
pricefile.to_csv(directory+'/pricerecord.csv')

simfile=pd.DataFrame(simrecord)
simfile.to_csv(directory+'/simrecord.csv',header=["intelligence", "book?","draws", "su","efficiency", "price","distance"])

dgridfile=pd.DataFrame(dgridrecord)
dgridfile.to_csv(directory+'/dgridrecord.csv')

egridfile=pd.DataFrame(egridrecord)
egridfile.to_csv(directory+'/egridrecord.csv')

pgridfile=pd.DataFrame(pgridrecord)
pgridfile.to_csv(directory+'/pgridrecord.csv')

edgridfile=pd.DataFrame(edgridrecord)
edgridfile.to_csv(directory+'/edgridrecord.csv')


#puts full set of results to a file...
file1=open(resultfilename,"a")
file1.writelines(simlines)
file1.close()

""" displays the histogram of efficiency at the end of the simulations"""


