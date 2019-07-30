import numpy as np
import bilby
from collections import namedtuple

Summary = namedtuple('Summary', ['n_posterior', 'n_mode_1', 'n_mode_2', 'frac_1_2'])


res_list = bilby.result.ResultList([])
for i in range(128):
    res_list.append(bilby.result.read_in_result('outdir/multidim_gaussian_bimodal_{}_result.json'.format(i)))

output = []

for res in res_list:
    x0 = res.posterior['x0'].values
    n_posterior = len(x0)
    n_mode_1 = len(x0[np.where(x0 < 0)])
    n_mode_2 = len(x0[np.where(x0 > 0)])
    frac_1_2 = n_mode_1/n_posterior
    output.append(Summary(n_posterior=n_posterior, n_mode_1=n_mode_1, n_mode_2=n_mode_2, frac_1_2=frac_1_2))

within_variance = 0
for summary in output:
    print(summary)
    variance = 0.5 * np.sqrt(summary.n_posterior)
    if not variance > np.abs(summary.n_mode_1 - summary.n_posterior):
        within_variance += 1

print('Fraction within variance: ' + str(within_variance/len(res_list)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fracs = [summary.frac_1_2 for summary in output]
plt.hist(np.array(fracs) * 100)
plt.xlabel('Percentage within mode 1')
plt.ylabel('Count')
plt.savefig('outdir/fracplot')
plt.clf()
