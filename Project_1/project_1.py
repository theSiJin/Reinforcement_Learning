import numpy as np
import matplotlib.pyplot as plt

# design on the experiment
num_training = 100
num_seq = 10

# generate actions
def action():
    return np.random.choice([-1, 1])

# generate episodes based on number of training and number of sequences
def generate_data(num_training = num_training, num_seq = num_seq):
    training_sets = {}
    for i in range(num_training):
        seqs = []
        for j in range(num_seq):
            seq = []
            while (np.sum(seq) != 3) and (np.sum(seq) != -3):
                seq.append(action())
            seqs.append(seq)
        training_sets[i] = seqs
    return training_sets

# del of P in the paper
def nabla(i):
    d = np.zeros(7)
    d[i] = 1
    return d

# compute RMSE
def rms(y):
    y_true = [1/6, 2/6, 3/6, 4/6, 5/6]
    err = y - y_true
    mse = np.mean(err ** 2)
    return np.sqrt(mse)

# compute standard error
def se(x):
    return np.std(x, ddof = 1) / np.sqrt(len(x))

# replicate of fig 3
def rep_fig3(dataset, lmd_list, alpha = 0.01, tol = 1e-4, max_iter = 10000):
    meanRMSE = []
    for lmd in lmd_list:
        err = []
        for i in range(num_training):
            w = np.ones(7) * 0.5
            w[0] = 0
            w[-1] = 1
            converge = 0
            cnt = 0
            while converge == 0:
                seqs = dataset[i]
                delta_w = np.zeros(7)

                for episode in seqs:
                    # transfrom actions into states (from 0 to 6)
                    states = np.cumsum(episode) + 3
                    # add initial state into the array
                    states = np.insert(states, 0, 3, axis = 0)

                    n_step = len(states)
                    et = 0  # value of the sum in eq.(4) as introduced on page 16
                    for t in range(n_step-1):
                        this_state = states[t]
                        next_state = states[t+1]
                        et = lmd * et + nabla(this_state)
                        delta_w = delta_w + alpha * (w[next_state] - w[this_state]) * et  # eq (4)
                
                w_old = w
                w = w + delta_w
                if np.all(np.abs(w - w_old) < tol):
                    converge = 1

                cnt += 1
                if cnt > max_iter:
                    break
            err.append(rms(w[1:-1]))     
        meanRMSE.append(np.mean(err))
    return meanRMSE

# replicate of fig 4
def rep_fig4(dataset, lmd_list, alpha_list):
    meanRMSE = {}
    for lmd in lmd_list:
        meanRMSE[lmd] = []
        for alpha in alpha_list:
            err = []
            for i in range(num_training):
                w = np.ones(7) * 0.5
                w[0] = 0
                w[-1] = 1
                seqs = dataset[i]
                
                for episode in seqs:
                    delta_w = np.zeros(7)
                    states = np.cumsum(episode) + 3
                    states = np.insert(states, 0, 3, axis = 0)
                    n_step = len(states)
                    et = 0
                    for t in range(n_step-1):
                        this_state = states[t]
                        next_state = states[t+1]
                        et = lmd * et + nabla(this_state)
                        delta_w = delta_w + alpha * (w[next_state] - w[this_state]) * et
                    w = w + delta_w
                err.append(rms(w[1:-1]))
            meanRMSE[lmd].append(np.mean(err))
    return meanRMSE

# replicate of fig 5
def rep_fig5(dataset, lmd_list, alpha_list):
    err_with_best_alpha = []
    for lmd in lmd_list:
        meanRMSE = []
        for alpha in alpha_list:
            err = []
            for i in range(num_training):
                w = np.ones(7) * 0.5
                w[0] = 0
                w[-1] = 1
                seqs = dataset[i]
                
                for episode in seqs:
                    delta_w = np.zeros(7)
                    states = np.cumsum(episode) + 3
                    states = np.insert(states, 0, 3, axis = 0)
                    n_step = len(states)
                    et = 0
                    for t in range(n_step-1):
                        this_state = states[t]
                        next_state = states[t+1]
                        et = lmd * et + nabla(this_state)
                        delta_w = delta_w + alpha * (w[next_state] - w[this_state]) * et
                    w = w + delta_w
                err.append(rms(w[1:-1]))
            meanRMSE.append(np.mean(err))
        best_alpha_idx = np.argmin(meanRMSE)
        err_with_best_alpha.append(meanRMSE[best_alpha_idx])
    return err_with_best_alpha


training = generate_data()

##################
## Experiment 1 ##
##################
lmd_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
rmseFig3 = rep_fig3(training, lmd_list)
seFig3 = se(rmseFig3)


plt.plot([0,1,3,5,7,9,10], rmseFig3, marker = "o")
plt.title("Replicate of Figure 3. Average error under repeated presentations")
plt.xlabel(r"$\lambda$")
plt.ylabel("ERROR")
plt.xticks([0,1,3,5,7,9,10], lmd_list)
plt.savefig("fig3.png")
plt.show()

##################
## Experiment 2 ##
##################

## fig 4
lmd_list = [0, 0.3, 0.8, 1.0]
alpha_list = np.arange(0, 0.65, step = 0.05)
rmseFig4 = rep_fig4(training, lmd_list, alpha_list)

for lmd in lmd_list:
    plt.plot(rmseFig4[lmd], label = r"$\lambda=${0}".format(lmd), marker = "o")
plt.legend(loc="upper left")
plt.xlabel(r"$\alpha$")
plt.ylim(0, 0.8)
plt.xticks(np.arange(0,13,2), [0,0.1,0.2,0.3,0.4,0.5,0.6])
plt.title(r"Replicate of Figure 4. Average error after 10 episodes")
plt.savefig("fig4.png")
plt.show()

## fig 5
lmd_list = np.arange(0, 1.1, step = 0.1)
alpha_list = np.arange(0, 0.7, step = 0.05)
rmseFig5 = rep_fig5(training, lmd_list, alpha_list)
  
plt.plot(rmseFig5, marker = "o")
plt.title(r"Replicate of Figure 5. Average error at best $\alpha$ value")
plt.xlabel(r"$\lambda$")
plt.ylabel("ERROR")
plt.xticks(np.arange(0,11,2), [0.0,0.2,0.4,0.6,0.8,1.0])
plt.savefig("fig5.png")
plt.show()