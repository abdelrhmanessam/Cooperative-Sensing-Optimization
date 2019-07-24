import numpy as np
import math
from scipy.stats import invgauss
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


min_sen_t = 100
pd = 0.9
pf = 0.1
nodes_in_net = 50
snr = -13
pds = []
pfs = []
q_inv_pds = []
q_inv_pfs = []
ts = []
#loop to get the values of pds
for v in range(1,nodes_in_net+1):
    single_pds = 1 - math.pow((1-pd), 1/v)
    pds.append(single_pds)

pds_mean = np.mean(pds)

#loop to get the values of pfs
for v in range(1,nodes_in_net+1):
    single_pds = 1 - math.pow((1-pf), 1/v)
    pfs.append(single_pds)

pfs_mean = np.mean(pfs)

#get all q inverse for pds
for i in pds:
    x = invgauss.pdf(i, pds_mean)
    q_inv_pds.append(x)

#get all q inverse for pfs
for i in pfs:
    x = invgauss.pdf(i, pfs_mean)
    q_inv_pfs.append(x)

for i in range(nodes_in_net):
    tsi = math.pow( ( ( (q_inv_pfs [i] - q_inv_pds[i]) * (1+ snr)  ) / snr ),2)
    ts.append(tsi)
ts_sorted = sorted(ts)   # Sorted the nodes in ascending order based on TS i

cmu_sum = np.cumsum(ts_sorted)

# loop over the cumulative sum of sensing time of nodes and get the nodes with ST less than minimum sensing time
less_than=[]
greater_than=[]
for i in range(len(cmu_sum)) :
    if cmu_sum[i] < min_sen_t :
        less_than.append(cmu_sum[i])
    else:
        greater_than.append(cmu_sum[i])

#####################Naive#####################

#spliting data with a percentage into  training and testing sets
split_percentage= .7

training= cmu_sum[0:int(len(cmu_sum) * split_percentage)]
testing=cmu_sum[len(training):len(cmu_sum)]

training_L =[]
testing_L=[]
for i in range(len(training)):
    training_L.append([training[i]])

for i in range(len(testing)):
    testing_L.append([testing[i]])

classes = []
for i in cmu_sum:
    if (i in less_than):
        classes.append(0)
    else:
        classes.append(1)

class_train= classes[0:int(len(classes) * split_percentage)]
class_test=classes[len(class_train):len(classes)]

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(training_L, class_train)

predicts = model.predict(testing_L)

# Calculate Accuracy Rate by using accuracy_score()
print ("Accuracy Rate: %f" % accuracy_score(class_test, predicts))

