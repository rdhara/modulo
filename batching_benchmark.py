import time
import random
import pandas
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

class Module():
    def __init__(self, proc_time=None):
        self.proc_time = proc_time
    def run(self):
        time.sleep(self.proc_time)

class ModuleA(Module):
    def __init__(self, proc_time=1e-4):
        super(ModuleA, self).__init__(proc_time)

class ModuleB(Module):
    def __init__(self, proc_time=2e-4):
        super(ModuleB, self).__init__(proc_time)

class ModuleC(Module):
    def __init__(self, proc_time=3e-4):
        super(ModuleC, self).__init__(proc_time)

class ModuleD(Module):
    def __init__(self, proc_time=4e-4):
        super(ModuleD, self).__init__(proc_time)

class ModuleE(Module):
    def __init__(self, proc_time=5e-4):
        super(ModuleE, self).__init__(proc_time)

def generate_task(N, sigma, k=64, max_l=5, mode='homogeneous'):
    T = []
    if mode == 'homogeneous':
        assert(N % k == 0), 'N must be a multiple of batch size'
        for _ in range(N // k):
            layout =  [' '.join(random.choice(sigma) for _ in range(random.randrange(2, max_l)))]
            T.extend(layout*k)
    elif mode == 'heterogeneous':
        for _ in range(N):
            T.append(' '.join(random.choice(sigma) for _ in range(random.randrange(2, max_l))))
    else:
        raise ValueError('Invalid mode {}'.format(mode))
    return T

class HomogeneousBatching():
    def __init__(self, T, M, k, batch_cost=0.0005):
        self.T = T
        self.M = M
        self.k = k
        self.batch_cost = batch_cost
    
    def train(self):
        t0 = time.time()
        N = len(self.T)
        for b in range(N // self.k):
            layout = list(map(lambda x: self.M[x], self.T[b].split()))
            for mod in layout:
                mod.run()
            time.sleep(self.batch_cost)
        dt = time.time() - t0
        print('ELAPSED TIME: {:.2f} s'.format(dt))
        return dt

class NoBatching():
    def __init__(self, T, M):
        self.T = T
        self.M = M
    
    def train(self):
        t0 = time.time()
        flat_T = [item for sublist in self.T for item in sublist]
        for layout in flat_T:
            for mod in layout.split():
                self.M[mod].run()
        dt = time.time() - t0
        print('ELAPSED TIME: {:.2f} s'.format(dt))
        return dt

class HeterogeneousBatching():
    def __init__(self, T, M, k, batch_cost=5e-4):
        self.T = [t.split() for t in T]
        self.M = M
        self.k = k
        self.batch_cost = batch_cost

    def train(self):
        t0 = time.time()
        N = len(self.T)     
        H = {(-1, j) for j in range(N)}
        T_done = set()
        T_prog = [0 for _ in range(N)]

        while len(T_done) < N:
            P = {m: [] for m in self.M.keys()}
            P_full = {m: False for m in self.M.keys()}
            while_ctr = 0
            while not sum(P_full.values()) and while_ctr < 1:
                while_ctr = 0
                for t in range(len(self.T)):
                    if t not in T_done and (T_prog[t] - 1, t) in H:
                        mod = self.T[t][T_prog[t]]
                        curr_queue = len(P[mod])
                        if curr_queue < self.k:
                            while_ctr -= 1
                            P[mod].append(self.M[mod])
                            H.remove((T_prog[t] - 1, t))
                            T_prog[t] += 1
                            H.add((T_prog[t] - 1, t))
                            if T_prog[t] == len(self.T[t]):
                                T_done.add(t)
                            if curr_queue + 1 == self.k:
                                P_full[mod] = True
                while_ctr += 1
            for m in P.keys():
                self.M[m].run()
                time.sleep(self.batch_cost)
        dt = time.time() - t0
        print('ELAPSED TIME: {:.2f} s'.format(dt))
        return dt

M = {
        'a': ModuleA(),
        'b': ModuleB(),
        'c': ModuleC(),
        'd': ModuleD(),
        'e': ModuleE()
    }

res = []
ns = [256, 1024, 4096, 16384, 65536, 262144]
ks = [16, 32, 64, 128]
ls = [5, 10, 15, 20]

for n in ns:
    for k in ks:
        for l in ls:
            dataset = generate_task(n, 'abcde', k, l, mode='heterogeneous')
            htb = HeterogeneousBatching(dataset, M, k)
            nb = NoBatching(dataset, M)
            res.append({
                'n': n,
                'k': k,
                'l': l,
                'htb_time': htb.train(),
                'nb_time': nb.train()
            })

df = pandas.DataFrame(res)
# df = pandas.read_csv('../batching.csv')
df['log_htb_time'] = np.log10(df['htb_time'])
df['log_nb_time'] = np.log10(df['nb_time'])
df['log_n'] = np.log10(df['n'])

sns.set_style('whitegrid')
sns.set_context('paper',  rc={'axes.titlesize':22, 'legend.fontsize':'xx-large', 'axes.labelsize': 16})
sns.set_palette('muted', color_codes=True)

f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(20,16))
plt.suptitle('Experiment Set 1: Fixing k = 64', fontsize=26)
plt.subplots_adjust(hspace=0.25)
axmap = {1: ax1, 2: ax2, 3: ax3, 4: ax4}
for i in range(1, 5):
    ax = axmap[i]
    ax.set_title('Maximum layout length $\ell_{max} = $' + str(ls[i-1]))
    df_sub = df[(df['k'] == 64) & (df['l'] == ls[i-1])]
    ax.plot(df_sub['log_n'], df_sub['log_htb_time'], c='r', marker='o', label='HTB')
    ax.plot(df_sub['log_n'], df_sub['log_nb_time'], c='b', marker='o', label='NB')
    ax.set_xlabel('Log training size (log N)')
    ax.set_ylabel('Log time (s)')
    ax.tick_params(labelsize=14)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')

f.legend(loc=7)
sns.despine()
f.savefig('../Figures/kfixed.pdf', dpi=200)

f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(20,16))
plt.suptitle('Experiment Set 2: Fixing N = 65536', fontsize=26)
plt.subplots_adjust(hspace=0.25)
axmap = {1: ax1, 2: ax2, 3: ax3, 4: ax4}
for i in range(1, 5):
    ax = axmap[i]
    ax.set_title('Batch size k = {}'.format(ks[i-1]))
    df_sub = df[(df['n'] == 65536) & (df['k'] == ks[i-1])]
    ax.plot(df_sub['l'], df_sub['log_htb_time'], c='r', marker='o', label='HTB')
    ax.plot(df_sub['l'], df_sub['log_nb_time'], c='b', marker='o', label='NB')
    ax.set_xlabel('Max layout length ($\ell_{max}$)')
    ax.set_ylabel('Log time (s)')
    ax.tick_params(labelsize=14)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')

f.legend(loc=7)
sns.despine()
f.savefig('../Figures/Nfixed.pdf', dpi=200)

f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(20,16))
plt.suptitle('Experiment Set 3: Fixing $\ell_{max}$ = 15', fontsize=26)
plt.subplots_adjust(hspace=0.25)
axmap = {1: ax1, 2: ax2, 3: ax3, 4: ax4}
for i in range(1, 5):
    ax = axmap[i]
    ax.set_title('Training examples N = {}'.format(ns[i+1]))
    df_sub = df[(df['l'] == 10) & (df['n'] == ns[i+1])]
    ax.plot(df_sub['k'], df_sub['log_htb_time'], color='r', marker='o', label='HTB')
    ax.plot(df_sub['k'], df_sub['log_nb_time'], color='b', marker='o', label='NB')
    ax.set_xlabel('Batch size (k)')
    ax.set_ylabel('Log time (s)')
    ax.tick_params(labelsize=14)
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')

f.legend(loc=7)
sns.despine()
f.savefig('../Figures/ellfixed.pdf', dpi=200)