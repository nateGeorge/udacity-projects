'''
analyzes the data from alpha/gamma optimization
'''

import pickle as pk
import seaborn as sns
import pylab as plt
import numpy as np
import pandas as pd

penalties = pk.load(open('penalties.pk','r'))
rewards = pk.load(open('rewards.pk','r'))
total_actions = pk.load(open('total_actions.pk','r'))
non_ideal_actions = pk.load(open('non_ideal_actions.pk','r'))

'''
using an example from running the optimizer:
best by penalties: a=0.2, g=0.6 -16.0
best by rewards: a=0.7, g=0.2 2309.5
best by time steps: a=0.3, g=0.1 1170
best by non_ideal_actions: a=0.1, g=0.1 16

I decided to normalize and weight 
the metrics so that they 
have weights I believe are appropriate.

Reward is weighted most heavily (50%),
penalties are weighted second most (20%),
non-ideal actions are third most (20%),
and time steps last (10%).
'''

maxRW = max([v for k,v in rewards.items()])
minRW = min([v for k,v in rewards.items()])

maxP = max([v for k,v in penalties.items()])
minP = min([v for k,v in penalties.items()])

maxTA = max([v for k,v in total_actions.items()])
minTA = min([v for k,v in total_actions.items()])

maxNIA = max([v for k,v in non_ideal_actions.items()])
minNIA = min([v for k,v in non_ideal_actions.items()])

scaledR = {}
scaledP = {}
scaledTA = {}
scaledNIA = {}

combined_weight = {}

for combo in penalties.keys():
    combined_weight[combo] = 0
    scaledR[combo] = (rewards[combo] - minRW) / float((maxRW - minRW))
    scaledP[combo] = (penalties[combo] - minP) / float((maxP - minP))
    scaledTA[combo] = (total_actions[combo] - minTA) / float((maxTA - minTA))
    scaledNIA[combo] = (non_ideal_actions[combo] - minNIA) / float((maxNIA - minNIA))
    
    combined_weight[combo] += scaledR[combo] * 0.5
    combined_weight[combo] += scaledP[combo] * 0.2
    combined_weight[combo] += scaledTA[combo] * 0.2
    combined_weight[combo] += scaledNIA[combo] * 0.1

best = max(combined_weight, key=combined_weight.get)
print 'best:', str(best) + ',', 'combined weight score of:', round(combined_weight[best], 3)

print 'scaled rewards of best:', round(scaledR[best], 3)
print 'scaled penalties of best:', round(scaledP[best], 3)
print 'scaled total actions of best:', round(scaledTA[best], 3)
print 'scaled non-ideal actions of best:', round(scaledNIA[best], 3)

twoDcw = pd.DataFrame(np.nan, index=range(len(combined_weight)), columns = ['alpha', 'gamma', 'score'])
twoDr = pd.DataFrame(np.nan, index=range(len(combined_weight)), columns = ['alpha', 'gamma', 'score'])
twoDp = pd.DataFrame(np.nan, index=range(len(combined_weight)), columns = ['alpha', 'gamma', 'score'])
twoDta = pd.DataFrame(np.nan, index=range(len(combined_weight)), columns = ['alpha', 'gamma', 'score'])
twoDnia = pd.DataFrame(np.nan, index=range(len(combined_weight)), columns = ['alpha', 'gamma', 'score'])

cnt = 0
for combo in combined_weight.keys():
    alpha = combo[2:5]
    gamma = combo[9:]
    
    twoDcw.iloc[cnt]['alpha'] = alpha
    twoDcw.iloc[cnt]['gamma'] = gamma
    twoDcw.iloc[cnt]['score'] = combined_weight[combo]
    
    twoDr.iloc[cnt]['alpha'] = alpha
    twoDr.iloc[cnt]['gamma'] = gamma
    twoDr.iloc[cnt]['score'] = rewards[combo]
    
    twoDp.iloc[cnt]['alpha'] = alpha
    twoDp.iloc[cnt]['gamma'] = gamma
    twoDp.iloc[cnt]['score'] = penalties[combo]
    
    twoDta.iloc[cnt]['alpha'] = alpha
    twoDta.iloc[cnt]['gamma'] = gamma
    twoDta.iloc[cnt]['score'] = total_actions[combo]
    
    twoDnia.iloc[cnt]['alpha'] = alpha
    twoDnia.iloc[cnt]['gamma'] = gamma
    twoDnia.iloc[cnt]['score'] = non_ideal_actions[combo]
    
    cnt += 1

twoDcw2 = twoDcw.pivot('alpha', 'gamma', 'score')
twoDr = twoDr.pivot('alpha', 'gamma', 'score')
twoDp = twoDp.pivot('alpha', 'gamma', 'score')
twoDta = twoDta.pivot('alpha', 'gamma', 'score')
twoDnia = twoDnia.pivot('alpha', 'gamma', 'score')

ax = sns.heatmap(twoDcw2)
plt.title('combined weight')
plt.savefig('figures/combined Weight.png')
plt.close()
#plt.show()

ax = sns.heatmap(twoDr)
plt.title('rewards')
plt.savefig('figures/rewards.png')
plt.close()
#plt.show()

ax = sns.heatmap(twoDp)
plt.title('penalties')
plt.savefig('figures/penalties.png')
plt.close()
#plt.show()

ax = sns.heatmap(twoDta)
plt.title('total_actions')
plt.savefig('figures/total_actions.png')
plt.close()
#plt.show()

ax = sns.heatmap(twoDnia)
plt.title('non_ideal_actions')
plt.savefig('figures/non_ideal_actions.png')
plt.close()
#plt.show()
