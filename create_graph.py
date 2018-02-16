from pandas import read_csv
import matplotlib.pyplot as plt

read_output = read_csv('orig_prediction.csv',index_col=0)
#print read_output
plt.plot(read_output)
plt.savefig('graph.png')
