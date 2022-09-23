import matplotlib.pyplot as plt

E = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


DLE = [1.2089, 1.4899, 1.6102, 1.7771, 1.8441, 1.9504, 2.0918, 2.16756, 2.2665, 2.3061]

#RBE = [1.6918, 2.0257, 2.2597, 2.3956, 2.5601, 2.6011, 2.685, 2.7034, 2.7778, 2.8165]

RBE = [1.4117, 1.7257, 1.9597, 2.0956, 2.2601, 2.3011, 2.385, 2.4034, 2.4778, 2.5165]

RLE = [1.7117, 2.1627, 2.3409, 2.4824, 2.6402, 2.7055, 2.7518, 2.7578, 2.8049, 2.8561]

d1 = ((sum(RBE)-sum(DLE))/sum(RBE))*100
d2 = ((sum(RLE)-sum(DLE))/sum(RLE))*100
print(d1)
print(d2)


plt.plot(E, DLE, 'g-o', label = "DLAQRP")
plt.plot(E, RBE, 'r-s', label = "RLBR")
plt.plot(E, RLE, 'b-*', label = "R2LTO")

plt.xlabel('Initial Node Energy (J)')
plt.ylabel('AECR (J/Round)')
plt.legend()
plt.show()