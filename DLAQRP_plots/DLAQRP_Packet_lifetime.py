import matplotlib.pyplot as plt

E = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


#DLL = [59165, 27582, 18721, 14791, 11833, 9860, 8452, 7395, 6573, 5916]
#RBL = [2806, 5396, 8242, 11005, 13331, 15531, 17289, 19586, 21805, 23365]
#RBL = [29365, 14682, 9788, 7341, 5873, 4894, 4195, 3670, 3262, 2936]
#RL = [2724, 5377, 7966, 10576, 13135, 14576, 16586, 19021, 21351, 22884]
#RLL = [45768, 22884, 15256, 11442, 9153, 7628, 6538, 5721, 5085, 4576]

DLL = [57915, 28860, 17970, 16485, 12885, 9540, 9540, 9330, 7755, 6750]
RBL = [44352, 22920, 14220, 12216, 9744, 7608, 6912, 6756, 5232, 4692]
RLL = [29835, 16857, 10647, 8730, 6390, 5661, 4950, 3978, 3447, 3438]


d1 = ((sum(DLL)-sum(RBL))/sum(RBL))*100
d2 = ((sum(DLL)-sum(RLL))/sum(RLL))*100
print(d1)
print(d2)


plt.plot(E, DLL, 'g-o', label = "DLAQRP")
plt.plot(E, RBL, 'r-s', label = "RLBR")
plt.plot(E, RLL, 'b-*', label = "R2LTO")
plt.xlabel('Packet Generation Rate (/s)')
plt.ylabel('Network Lifetime (s)')
plt.legend()
plt.show()
