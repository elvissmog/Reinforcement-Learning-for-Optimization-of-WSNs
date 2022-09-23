import matplotlib.pyplot as plt

E = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


DLL = [5620, 11545, 17829, 23753, 29678, 36133, 41326, 46954, 53657, 59165]
#RBL = [2806, 5396, 8242, 11005, 13331, 15531, 17289, 19586, 21805, 23365]
RBL = [4806, 8396, 10242, 13005, 16331, 17531, 19289, 22586, 24805, 29365]
#RL = [2724, 5377, 7966, 10576, 13135, 14576, 16586, 19021, 21351, 22884]
RLL = [5448, 10754, 15932, 21152, 26270, 29152, 33172, 38042, 42702, 45768]


d1 = ((sum(DLL)-sum(RBL))/sum(RBL))*100
d2 = ((sum(DLL)-sum(RLL))/sum(RLL))*100
print(d1)
print(d2)


plt.plot(E, DLL, 'g-o', label = "DLAQRP")
plt.plot(E, RBL, 'r-s', label = "RLBR")
plt.plot(E, RLL, 'b-*', label = "R2LTO")
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Network Lifetime (s)')
plt.legend()
plt.show()
