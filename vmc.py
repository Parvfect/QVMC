import numpy as np
import matplotlib.pyplot as plt


class VMC:

    def __init__(self, alpha, wf, batched=False):

        self.alpha = alpha
        self.wf = wf
        self.batched = batched

    def prob_density(self, r1, r2):
        return self.wf(r1, r2, self.alpha) **  2

    def E_local(self, r1, r2):

        if self.batched:
            axis = 1
        else:
            axis = 0
        norm_r1 = np.linalg.norm(r1, axis=axis)
        norm_r2 = np.linalg.norm(r2, axis=axis)
        r12 = np.linalg.norm(r1 - r2, axis=axis)

        if self.batched:
            norm_r1 = norm_r1.reshape(norm_r1.shape[0], 1)
            norm_r2 = norm_r2.reshape(norm_r1.shape[0], 1)
            dot_product = np.diag(np.dot(r1 / norm_r1 - r2 / norm_r2, (r1 - r2).T))
        else:
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


class VMCvec:

    def __init__(self, alpha, wf):

        self.alpha = alpha
        self.wf = wf
        self.runs = self.alpha.shape[0]
        
    def prob_density(self, r1, r2):
        return self.wf(r1, r2, self.alpha) **  2

    def E_local(self, r1, r2):

        norm_r1 = np.linalg.norm(r1, axis=1)
        norm_r2 = np.linalg.norm(r2, axis=1)
        r12 = np.linalg.norm(r1 - r2, axis=1)

        norm_r1 = norm_r1.reshape(norm_r1.shape[0], 1)
        norm_r2 = norm_r2.reshape(norm_r1.shape[0], 1)
        #dot_product = np.einsum('ij,ij->i', r1 / norm_r1 - r2 / norm_r2, r1 - r2)
        dot_product = np.sum((r1 / norm_r1 - r2 / norm_r2) * (r1 - r2), axis=1)
        #dot_product = np.diag(np.dot(r1 / norm_r1 - r2 / norm_r2, (r1 - r2).T))

        return - 4 + dot_product / (r12 * (1 + self.alpha * r12)**2) - 1 / (r12 * (1 + self.alpha * r12)**3) - 1/(4 * (1 + self.alpha * r12)**4) + 1 / r12 
        
    def metropolis(self, N):            
        L = 1
        r1 = np.random.rand(self.runs, 3) * 2 * L - L
        r2 = np.random.rand(self.runs, 3) * 2 * L - L #random number from -L to L
        E = 0
        E2 = 0
        Eln_average = 0
        ln_average = 0
        rejection_ratio = 0
        step = 0
        max_steps = 500
        
        for i in range(N):
            chose = np.random.rand(self.runs).reshape(self.runs, 1)
            dummy = np.random.rand(self.runs)
            
            step = step + 1

            perturbed_r1 = r1 + 0.5 * (np.random.rand(self.runs, 3) * 2 * L - L)
            perturbed_r2 = r2 + 0.5 * (np.random.rand(self.runs, 3) * 2 * L - L)

            r1_trial = np.where(chose < 0.5, perturbed_r1, r1)
            r2_trial = np.where(chose >= 0.5, perturbed_r2, r2)

            psi = self.prob_density(r1, r2)
            psi_trial = self.prob_density(r1_trial, r2_trial)            
            psi_ratio = psi_trial / psi

            density_comp = psi_trial >= psi
            dummy_comp = dummy < psi_ratio

            condition = density_comp + dummy_comp

            rejection_ratio += np.where(condition, 1./N, 0.0)

            condition = condition.reshape(condition.shape[0], 1)

            # Careful with overwriting
            r1 = np.where(condition, r1_trial, r1)
            r2 = np.where(condition, r2_trial, r2)
                    
            if step > max_steps:
                E += self.E_local(r1, r2) / (N - max_steps)
                E2 += self.E_local(r1, r2) ** 2 / (N - max_steps)
                r12 = np.linalg.norm(r1 - r2, axis=1)
                Eln_average += (self.E_local(r1, r2) * - r12 ** 2 / (2 * (1 + self.alpha * r12) ** 2))/(N - max_steps)
                ln_average += -r12 ** 2 / (2 * (1 + self.alpha * r12) ** 2) / (N - max_steps)

        return E, E2, Eln_average, ln_average, rejection_ratio

