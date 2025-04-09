import numpy as np
import matplotlib.pyplot as plt


class VMC:

    def __init__(self, alpha, wf):

        self.alpha = alpha
        self.wf = wf

    def prob_density(self, r1, r2):
        return self.wf(r1, r2, self.alpha) ** 2

    def E_local(self, r1, r2):
        norm_r1 = np.linalg.norm(r1)
        norm_r2 = np.linalg.norm(r2)
        r12 = np.linalg.norm(r1 - r2)        
        dot_product = np.dot(r1 / norm_r1 - r2 / norm_r2, r1 - r2)
        energy = - 4 + dot_product / (r12 * (1 + self.alpha * r12)**2) - 1 / (r12 * (1 + self.alpha * r12)**3) - 1/(4 * (1 + self.alpha * r12)**4) + 1 / r12 
        return energy
    
    def metropolis(self, N):            
        L = 1
        r1 = np.random.rand(3) * 2 * L - L
        r2 = np.random.rand(3) * 2 * L - L #random number from -L to L
        E = 0
        E2 = 0
        Eln_average = 0
        ln_average = 0
        rejection_ratio = 0
        step = 0
        max_steps = 500
        
        for i in range(N):
            chose = np.random.rand()
            step = step + 1
            if chose < 0.5:
                r1_trial = r1 + 0.5 * (np.random.rand(3) * 2 * L-L)
                r2_trial = r2
            else:
                r2_trial = r2 + 0.5 * (np.random.rand(3) * 2 * L-L)
                r1_trial = r1

            if self.prob_density(r1_trial, r2_trial) >= self.prob_density(r1, r2):
                r1 = r1_trial
                r2 = r2_trial
            
            else:
                dummy = np.random.rand()
                if dummy < self.prob_density(r1_trial, r2_trial) / self.prob_density(r1, r2):
                    r1 = r1_trial
                    r2 = r2_trial
                else:
                    rejection_ratio += 1./N
                    
            if step > max_steps:
                E += self.E_local(r1, r2) / (N - max_steps)
                E2 += self.E_local(r1, r2) ** 2 / (N - max_steps)
                r12 = np.linalg.norm(r1 - r2)
                Eln_average += (self.E_local(r1,r2) * -r12 ** 2 / (2 * (1 + self.alpha * r12) ** 2))/(N - max_steps)
                ln_average += -r12**2/(2 * (1 + self.alpha * r12)**2) / (N - max_steps)

        return E, E2, Eln_average, ln_average, rejection_ratio

