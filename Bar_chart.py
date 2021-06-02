import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


'''
sup_Round = []
for i in range(len(sup_Actions)):
    sup_Round.append(i)

y = max(set(my_Actions), key=my_Actions.count)
#print('mode:', y)

my_data = Counter(my_Actions)
print('my:', my_data.most_common())  # Returns all unique items and their counts
print('lifetime:', len(my_Actions))

sup_data = Counter(sup_Actions)
print('sup:', sup_data.most_common())
print('lifetime:', len(sup_Actions))
'''
# Numbers of pairs of bars you want
N = 10

# Data on X-axis
'''
c = [(18, 137807), (184, 5360), (210, 5308), (204, 5041), (170, 5022), (48, 5010), (49, 5009), (45, 5001), (99, 4996), (38, 4994), (78, 4987), (144, 4987), (62, 4983), (90, 4982), (209, 4982), (26, 4981), (56, 4979), (60, 4976), (100, 4975), (200, 4975)]

x = []
y = []
for ind in c:
    x.append(str(ind[0]))
    y.append(ind[1])

print('x:', x)
print('y:', y)
#a = [903, 875, 852, 895, 894, 830, 847, 833, 911, 880, 125569, 868, 869, 843, 900, 873]
#b = [9986, 632, 626, 618, 2904, 2754, 605, 655, 591, 602, 77056, 1161, 698, 642, 1458, 644]
'''
#a = [137807, 5360, 5308, 5041, 5022, 5010, 5009, 5001, 4996, 4994, 4987, 4987, 4983, 4982, 4982, 4981, 4979, 4976, 4975, 4975]
#a = [137807, 5360, 5308, 5041, 5022, 5010, 5009, 5001, 4996, 4994,]
a =[17738, 3159, 3111, 3103, 3063, 2960, 2942, 2911, 2901, 2898]
my = []
for i in a:
    #my.append(i*100/535338)
    my.append(i*100/272368)
'''
sup = []
for j in b:
    sup.append(j*100/sum(b))
'''
# Position of bars on x-axis
ind = np.arange(N)

# Figure size
#plt.figure(figsize=(10,5))

# Width of a bar
width = 0.1

# Plotting
#plt.bar(ind, my , width, label='CQR Design 1')
plt.bar(ind, my, width)
#plt.bar(ind + width, sup, width, label='CQR Design 2')

plt.xlabel('Routing Tables')
plt.ylabel('Percentage Utilization of RTs (%)')
#plt.title('Here goes title of the plot')

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations
#plt.xticks(ind + width / 2, ('18', '184', '210', '204', '170', '48', '49', '45', '99', '38', '78', '144', '62', '90', '209', '26', '56', '60', '100', '200'))
#plt.xticks(ind + width / 4, ('RT18', 'RT184', 'RT210', 'RT204', 'RT170', 'RT8', 'RT49', 'RT45', 'RT99', 'RT38'))
plt.xticks(ind + width / 4, ('RT18', 'RT45', 'RT60', 'RT62', 'RT184', 'RT49', 'RT26', 'RT48', 'RT38', 'RT144'))
# Finding the best position for legends and putting it
#plt.legend(loc='best')
plt.show()

