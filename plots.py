import matplotlib.pyplot as plt

'''
Graphical comparison of the performance metrics under the proposed Centralized Q-routing algorithm and the baseline algorithms (Distributed Q-routing, RLBR, and R2LTO )


 


plt.plot(Q_Round, Q_Vals, label = "Q-routing")
plt.plot(CQ_Round, CQ_Vals, label = "CQR")
plt.plot(Rlbr_Round, Rlbr_QVals, label = "RLBR")
plt.plot(Rllto_Round, Rllto_QVals, label = "R2LTO")
plt.xlabel('Round')
plt.ylabel('Q-value')
plt.title('Q-value Convergence')
plt.legend()
plt.show()

plt.plot(Q_Round, Q_Delay, label = "Q-routing")
plt.plot(CQ_Round, CQ_Delay, label = "CQR")
plt.plot(Rlbr_Round, Rlbr_Delay, label = "RLBR")
plt.plot(Rllto_Round, Rllto_Delay, label = "R2LTO")
plt.xlabel('Round')
plt.ylabel('Delay')
plt.title('Transmission Delay per Round')
plt.legend()
plt.show()

plt.plot(Q_Round, Q_Energy, label = "Q-routing")
plt.plot(CQ_Round, CQ_Energy, label = "CQR")
plt.plot(Rlbr_Round, Rlbr_Energy, label = "RLBR")
plt.plot(Rllto_Round, Rllto_Energy, label = "R2LTO")
plt.xlabel('Round')
plt.ylabel('Energy Consumption')
plt.title('Energy Consumption per Round')
plt.legend()
plt.show()

plt.plot(Q_Round, Q_Total_Energy, label = "Q-routing")
plt.plot(CQ_Round, CQ_Total_Energy, label = "CQR")
plt.plot(Rlbr_Round, Rlbr_Total_Energy, label = "RLBR")
plt.plot(Rllto_Round, Rllto_Total_Energy, label = "R2LTO")
plt.xlabel('Round')
plt.ylabel('Cummulative Energy Consumption')
plt.title('Cummulative Energy Consumption per Round')
plt.legend()
plt.show()


#Energy = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


MY_ARE = [50.295,101.667,152.723,204.329,255.232,305.931,356.524,406.890,457.137,506.604]
SD_ARE = [49.993,99.999,149.999,199.999,249.999,299.9997,349.999,399.999,449.999,499.999]
plt.plot(Energy, MY_ARE, label = "CQR Protocol")
plt.plot(Energy, SD_ARE, label = "MST Protocol")
plt.xlabel('Initial Energy')
plt.ylabel('Average Node MRE')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
plt.show()

Range = [10, 20, 30, 40, 50, 60, 70, 80]
M_l = [49978,49956,51230,49922,58799,57695,54359,49337]
S_l = [49978,49998,49999,49212,49999,47709,49999,48449]
plt.plot(Range, M_l, label = "CQR Protocol")
plt.plot(Range, S_l, label = "MST Protocol")
plt.xlabel('Transmission Range')
plt.ylabel('Lifetime')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
plt.show()


E = [1000,900,800,700,600,500]
m_l = [535419,482070,429459,377108,324747,272368]
s_l = [519066,468406,414375,362944,310637,259557]
r_l = [604804,544032,483553,423192,362638,302271]
plt.plot(E, m_l, label = "MY")
plt.plot(E, s_l, label = "SUP")
plt.plot(E, r_l, label = "RAD")
plt.xlabel('Initial Energy')
plt.ylabel('Lifetime')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
plt.show()

E = [1000,900,800,700,600,500]
m_r = [506.62015,457.13754,406.89006,356.52488,305.9317,255.23294]
s_r = [498.54340,449.346284,399.67362,349.22216,299.90812,249.59038]
r_r = [499.93707,450.43078,400.14654,349.79124,300.09112,249.95499]
plt.plot(E, m_r, label = "MY")
plt.plot(E, s_r, label = "SUP")
plt.plot(E, r_r, label = "RAD")
plt.xlabel('Initial Energy')
plt.ylabel('Average Minimum RE')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
plt.show()

#R = [30,40,50,60,70,80]
#R_m =[257329,251424,272368,298744,275980,247999]
#R_s = [249998,246062,249999,238549,249999,242247]
R = [40,50,60,70,80,90,100]
R_m =[632399,661394,787554,698382,623306,700265,581152]
R_s = [615155,624999,596373,624999,605619,656512,497611]
plt.plot(R, R_m, 'g-o', label = "LACQRP")
plt.plot(R, R_s, 'r-s', label = "STP")
plt.xlabel('Transmission Range (m)')
plt.ylabel('Network Lifetime')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
plt.show()

Energy = [500, 600, 700, 800, 900, 1000]
#My_LT = [58826,113053,167288,219804,272368,324747,377108,429459,482070,535338]
My_LT = [272368,324747,377108,429459,482070,535338]
#SD_LT = [49999,99999,149999,199999,249999,299999,349999,399999,449999,499999]
SD_LT = [249999,299999,349999,399999,449999,499999]

plt.plot(Energy, My_LT, 'g-o', label = "LACQRP")
plt.plot(Energy, SD_LT, 'r-s', label = "STP")
plt.xlabel('Initial Node Residual Energy (J)')
plt.ylabel('Network Lifetime')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
plt.show()


E = [100,200,300,400,500,600,700,800,900,1000]
CQR_LT1 = [33286,71219,111577,146770,184364,222198,259901,297681,335451,373272]
#CQR_LT2 = [33429,68395,102373,137637,170453,203121,236086,268837,301546,334254]
CQR_EC1 = [1218,2612,4113,5390,6770,8160,9546,10935,12323,13713]
CQR_EC2 = [1223,2505,3750,5041,6231,7412,8611,9765,10976,12128]
RLBR_LT = [17844,37671,57497,77324,97151,116977,136804,156631,176457,196284]
RLBR_EC = [367,775,1183,1591,2000,2408,2816,3224,3632,4040]
RL2TO_LT = [30044,63428,100238,134803,169368,203933,238499,273064,307629,342194]
RL2TO_EC = [541,1143,1812,2436,3061,3686,4311,4936,5561,6185]

#cr1 = ((sum(CQR_LT1) - sum(RL2TO_LT))/sum(RL2TO_LT))*100
#cr2 = ((sum(CQR_LT1) - sum(RLBR_LT))/sum(RLBR_LT))*100

#print('a', cr1)
#print('b', cr2)
cr1 = ((sum(CQR_EC1) - sum(RL2TO_EC))/sum(CQR_EC1))*100
cr2 = ((sum(CQR_EC1) - sum(RLBR_EC))/sum(CQR_EC1))*100
print('a', cr1)
print('b', cr2)

plt.plot(E, CQR_LT1, 'g-o', label = "CQRP1")
plt.plot(E, CQR_LT2, 'g-*', label = "CQRP2")
plt.plot(E, RLBR_LT, 'r-s', label = "RLBR")
plt.plot(E, RL2TO_LT, 'b-*', label = "R2LTO")
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Network Lifetime (s)')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
#plt.grid()
#plt.show()

plt.plot(E, CQR_EC1, 'g-o', label = "CQRP1")
plt.plot(E, CQR_EC2, 'g-*', label = "CQRP2")
plt.plot(E, RLBR_EC, 'r-s', label = "RLBR")
plt.plot(E, RL2TO_EC, 'b-*', label = "R2LTO")
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Network Energy Consumption (J)')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
#plt.grid()
#plt.show()


N = [1,2,3,4,5,6,7,8,9,10]
#cqr_lt1 =[33225,16023,10499,7785,6202,5126,4388,3837,3416,3056]
#cqr_lt2 = [33492,15993,10487,7809,6168,5142,4392,3853,3432,3071]
cqr_lt = [373272,188599,122786,91041,74505,61902,52871,46139,41109,37104]
#cqr_ec1 = [1215,1169,1148,1135,1129,1119,1118,1116,1118,1111]
#cqr_ec2 = [1220,1168,1147,1138,1123,1123,1118,1121,1124,1117]
cqr_ec = [13713,13714,13713,13712,13711,13711,13711,13710,13711,13712]
#rlbr_lt = [17844,8922,5948,4461,3569,2974,2549,2230,1983,1784]
rlbr_lt = [196284,98142,65428,49071,39256,32714,27802,24504,21809,19628]
#rlbr_ec = [367,367,367,367,367,367,367,367,367,368]
rlbr_ec = [4040,4041,4040,4041,4040,4041,4042,4042,4041,4041]
#rl2to_lt = [30044,15022,10014,7511,6008,5007,4292,3755,3338,3004]
rl2to_lt = [342194,171097,114064,85122,68166,57032,48884,42774,38021,34219]
#rl2to_ec = [541,541,541,541,541,541,541,541,541,541]
rl2to_ec = [6185,6185,6185,6184,6186,6187,6186,6187,6186,6187]

#cr1 = ((sum(cqr_lt) - sum(rl2to_lt))/sum(rl2to_lt))*100
#cr2 = ((sum(cqr_lt) - sum(rlbr_lt))/sum(rlbr_lt))*100

#print('a', cr1)
#print('b', cr2)

cr1 = ((sum(cqr_ec) - sum(rl2to_ec))/sum(cqr_ec))*100
cr2 = ((sum(cqr_ec) - sum(rlbr_ec))/sum(cqr_ec))*100

print('a', cr1)
print('b', cr2)


plt.plot(N, cqr_lt, 'g-o', label = "LACQRP")
#plt.plot(N, cqr_lt2, 'g-*', label = "CQRP2")
plt.plot(N, rlbr_lt, 'r-s', label = "RLBR")
plt.plot(N, rl2to_lt, 'b-*', label = "R2LTO")
plt.xlabel('Packet Generation Rate (/s)')
plt.ylabel('Network Lifetime (s)')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
#plt.grid()
plt.show()

plt.plot(N, cqr_ec, 'g-o', label = "CQRP")
#plt.plot(N, cqr_ec2, 'g-*', label = "CQRP2")
plt.plot(N, rlbr_ec, 'r-s', label = "RLBR")
plt.plot(N, rl2to_ec, 'b-*', label = "R2LTO")
plt.xlabel('Packet Generation Rate (/s)')
plt.ylabel('Network Energy Consumption (J)')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
#plt.grid()
plt.show()

#cr1 = ((sum(cqr_lt1) - sum(rl2to_lt))/sum(rl2to_lt))*100
#cr2 = ((sum(cqr_lt1) - sum(rlbr_lt))/sum(rlbr_lt))*100

#print('a', cr1)
#print('b', cr2)


E = [100,200,300,400,500,600,700,800,900,1000]
#CQR_LT1 = [33286,71219,111577,146770,184364,222198,259901,297681,335451,373272]
#CQR_LT2 = [33429,68395,102373,137637,170453,203121,236086,268837,301546,334254]
CQR_LT1 = [33451,71219,111577,146770,184364,222198,259901,297681,335451,373272]
CQR_EC1 = [1218,2612,4113,5390,6770,8160,9546,10935,12323,13713]
CQR_EC2 = [1223,2505,3750,5041,6231,7412,8611,9765,10976,12128]
RLBR_LT = [17844,37671,57497,77324,97151,116977,136804,156631,176457,196284]
RLBR_EC = [367,775,1183,1591,2000,2408,2816,3224,3632,4040]
RL2TO_LT = [30044,63428,100238,134803,169368,203933,238499,273064,307629,342194]
RL2TO_EC = [541,1143,1812,2436,3061,3686,4311,4936,5561,6185]

UT = [0.720456788,0.632820158,0.865784676,13.18685219,25.05870263,34.0267796,41.08208753,46.77928217,51.24977098,54.66643812]
#cr1 = ((sum(CQR_LT1) - sum(RL2TO_LT))/sum(RL2TO_LT))*100
#cr2 = ((sum(CQR_LT1) - sum(RLBR_LT))/sum(RLBR_LT))*100

#print('a', cr1)
#print('b', cr2)
cr1 = ((sum(CQR_EC1) - sum(RL2TO_EC))/sum(CQR_EC1))*100
cr2 = ((sum(CQR_EC1) - sum(RLBR_EC))/sum(CQR_EC1))*100
print('a', cr1)
print('b', cr2)

plt.plot(E, CQR_LT1, 'g-o', label = "LACQRP")
#plt.plot(E, CQR_LT2, 'g-*', label = "CQRP2")
plt.plot(E, RLBR_LT, 'r-s', label = "RLBR")
plt.plot(E, RL2TO_LT, 'b-*', label = "R2LTO")
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Network Lifetime (s)')
#plt.title('Cummulative Energy Consumption per Round')
plt.legend()
#plt.grid()
plt.show()

plt.plot(E, UT, 'g-o', label = "LACQRP")
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Optimal RT Utilization (%)')
plt.show()

E = [100,200,300,400,500,600,700,800,900,1000]
CQR_LT = [10364,22703,35677,48385,59723,71723,84024,96894,108890,120119]
CQR_AEC = [0.1127,0.11299,0.11314,0.11316,0.112727,0.11263,0.112726,0.11283,0.11279,0.112422]
CQR_ARE = [89.1361,176.1425,262.4566,349.07346,437.4002,524.88885,611.92973,698.33022,785.79507,874.4612]
RLBR_LT = [5841,12341,18851,25383,31891,38416,44920,51470,57999,64537]
RLBR_AEC = [0.2188,0.2517029,0.266541,0.27629,0.282345,0.2895529,0.290971,0.296065,0.30077,0.302596]
RLBR_ARE = [87.675,169.91112,251.2429,331.8742,412.47716,491.8013,572.8461,651.6838,730.1411,809.8156]
R2LTO_LT = [14340,27839,34218,37936,42881,52567,57851,63381,69594,75379]
#R2LTO_AEC = [0.227607,0.3048406,0.48883,0.70474,0.847138,0.844346,0.930746,0.995995,1.03934,1.08408]
R2LTO_AEC = [0.237607,0.2648406,0.27883,0.30474,0.317138,0.324346,0.330746,0.345995,0.35034,0.35408]
R2LTO_ARE = [68.4932,117.3346,135.4334,135.6419,140.12237,160.3025,166.12072,173.7321,182.173,188.7809]
plt.plot(E, CQR_LT, 'g-o', label = 'EACQR')
plt.plot(E, RLBR_LT, 'r-s', label = 'RLBR')
plt.plot(E, R2LTO_LT, 'b-*', label = 'R2LTO')
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Network Lifetime (s)')
plt.legend()
#plt.show()
plt.plot(E, CQR_AEC, 'g-o', label = 'EACQR')
plt.plot(E, RLBR_AEC, 'r-s', label = 'RLBR')
plt.plot(E, R2LTO_AEC, 'b-*', label = 'R2LTO')
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Average Energy Consumption (J)')
plt.legend()
#plt.show()
plt.plot(E, CQR_ARE, 'g-o', label = 'EACQR')
plt.plot(E, RLBR_ARE, 'r-s', label = 'RLBR')
plt.plot(E, R2LTO_ARE, 'b-*', label = 'R2LTO')
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Average Remaining Energy (J)')
plt.legend()
#plt.show()

print((sum(R2LTO_AEC)-sum(CQR_AEC))/sum(R2LTO_AEC))
print((sum(RLBR_AEC)-sum(CQR_AEC))/sum(RLBR_AEC))


E = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#OL = [149137, 316677, 499329, 755941, 943193, 1095605, 1243364, 1452164, 1566690, 1622717]
OL = [149137, 316677, 499329, 755941, 943193, 1095605, 1243364, 1452164, 1566690, 1629505]
#OT = [1262, 3389, 8763, 14697, 18414, 24271, 32725, 41209, 51960, 65674]
OT = [1262, 3389, 8763, 14697, 18414, 24271, 32725, 41209, 51960, 76002]
SL = [138089, 285309, 396221, 536378, 719764, 854733, 1036381, 1151500, 1309236, 1406677]
ST = [496, 1201, 4403, 8969, 14418, 18068, 22124, 30221, 39033, 51573]


PL = (sum(OL) - sum(SL))/sum(OL)
PT = (sum(OT) - sum(ST))/sum(OT)

print(PL)
print(PT)

plt.plot(E, SL, 'g', label = 'ILACQRP', linestyle="--")
plt.plot(E, OL, 'r', label = 'LACQRP', linestyle=":")
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Network Lifetime (s)')
plt.legend()
plt.show()


plt.plot(E, ST, 'g', label = 'ILACQRP', linestyle="--")
plt.plot(E, OT, 'r', label = 'LACQRP', linestyle=":")
plt.xlabel('Initial Node Energy (J)')
plt.ylabel('Computation Time (s)')
plt.legend()
plt.show()
'''

import json

with open('nqrvals.txt', 'r') as filehandle:
    vals = json.load(filehandle)

qvals = []
for qv in vals:
    qvals.append(qv)

Round = []
for j in range(len(qvals)):
    Round.append(j)

plt.plot(Round, qvals)
plt.xlabel('Generation')
plt.ylabel('Convergence rate')
plt.show()