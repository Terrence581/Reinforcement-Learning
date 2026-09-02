import numpy as np
import numpy.linalg as nplg
import numpy.random as npra
import matplotlib.pyplot as plt
from first_crossing_time import first_crossing_time_type1
from first_crossing_time import first_crossing_time_type2
from first_crossing_time import first_crossing_time_type3
from Multi_Arm_Bandit_Machine import *
from sympy import Matrix

class Whittle_Index_Three_State(object):
    """
    Introduction:
    - This is a class for the user of multi_arm bandit machine with 3 states(0, 1 and 2), who uses the Whittle Index Policy. Given the necessary parameters for the machine and the rule of the game, the class could simulate the game process of the user in given time. 
      
    Input:
    - MABM part:
      - arm_num: a scalar indicating the number of arms for the machine(or the number of total channels)
      - state_num: a scalar indicating the number of state for each arm(channel)(, given that they are all 2 in this class)
      - initial_dist: a matrix with dimension (arm_num, state_num), each row of which shows the initial  distribution of each arm(channel)
      - transition_matrics: a tensor with dimension (arm_num, state_num, state_num), each matrix of which shows the transition matrix of each arm(channel)
    
    - Game Rule part:
      - active_num: a scalar indicating the number of active arms(channels)(, that is the arms user could observe)
      - reward_set: an matrix with dimension (arm_num, state_num) showing the reward of each state for each arm(channel)
      - time: a scalar indicating the time user plays the game
    """
    def __init__(self, arm_num, state_num, initial_dist, transition_matrics, active_num, reward_set, time):
        """
        Parameters:
        - mabm: an object simulating the Multi_Arm Bandit Machine
        - action: an (N,) array showing the current action of each arm(channel)(active or passive)
        - N: a scalar for arm_num
        - K: a scalar for active_num
        - M: a scalar for state_num
        - belief_vec: a (N,) array showing the current belief state of each arm(channel)
        - P: a (N, M, M) tensor, each matrix of which shows the transition matrix of each arm(channel)
        - reward: a (N, M) matrix showing the reward of each state for each arm(channel)
        - T: a scalar for the playing time
        - AR: an(T,) array showing the actual reward of each time
        - J: a (N, M) matrix, each row of which shows the eigenvalues of the transition matric of each arm (channel)
        - A: a (N, M, M) tensor, each matrix of which shows the eigenvectors of the transition matric of each arm (channel)
        - A_inv: a (N, M, M) tensor, each matrix of which shows the inverse of the each matrix of A
        - right_side: a (N, M) matrix, each row of which is the matrix product of each of A_inv and each row of reward(which is just the rightside term of J**k in k-step reward function) 
        - pos: a (M,) list, each element of which shows whether the mth(m=1,...,M) eigenvalue of each arm(channel) is one
        - b1: a (N,) array showing the base of the first exponential term in k-step reward function f(k), which is also one of the two eigenvalues(that are not equal to one) of each arm(channel) 
        - b2: a (N,) array showing the base of the second exponential term in k-step reward function f(k), which is also the other one of the two eigenvalues(that are not equal to one) of each arm(channel) 
        - type1_pos: a (N,)array showing whether the k-step reward function of each arm is in the form of type one in their first crossing time
        - type2_pos: a (N,)array showing whether the k-step reward function of each arm is in the form of type two in their first crossing time
        - type3_pos: a (N,)array showing whether the k-step reward function of each arm is in the form of type three in their first crossing time
        - complex_eign: an array recording those complex eigenvalues which are used to simplify their k-step reward function
        - a1b1B: an tensor, each matrix of which showing the product of the right and left eigenvectors, which are corresponding to the complex_eign, and the reward vector B
        - a0b0B: an (N, M, 1) tensor, each matrix of which showing the product of the right and left eigenvectors, which are corresponding to the eigenvalue 1, and the reward vector B 
        """
        self.mabm = MABM(arm_num, state_num, initial_dist, transition_matrics)
        self.P = transition_matrics.copy()
        self.action = np.zeros(arm_num)
        self.belief_vec = initial_dist.copy()
        self.M = state_num
        self.N = arm_num
        self.K = active_num
        self.reward = reward_set.copy()
        self.T = time
        self.action = np.zeros(self.N)
        self.AR = np.array([])
        self.A = np.zeros(np.shape(self.P))
        self.A_inv = np.zeros(np.shape(self.A))
        self.J = np.zeros((arm_num, 3))
        self.right_side = np.zeros((arm_num, 3))
        self.pos = []
        self.b1 = np.zeros(arm_num)
        self.b2 = np.zeros(arm_num)
        self.type1_pos = np.zeros(self.N)
        self.type2_pos = np.zeros(self.N)
        self.type3_pos = np.zeros(self.N)
        self.complex_eign = []
        self.a1b1B = []
        self.a0b0B = np.zeros((self.N, 3, 1)).astype(complex)

    def matrics_decomposition(self, param):
        """
        Function:
        - Decompose the transition matrix of each arm(channel).
        - Set the parameters pos, b1 and b2 according to the position of eigenvalue of each arm(channel).
        - Compute the parameter right_side of the class.

        Input:
        - param: a (4,) list contains 4 parameters: set_decom, A, J, A_inv
               - set_decom: a boolean variable indicating that whether the matrics decomposition has been done
               - A:  if set_decom is True, a  (N, M, M) tensor, each matrix of which shows the eigenvectors of the transition matric of each arm (channel), or a empty list if set_decom is False
               - J: if set_decom is True, a (N, M) matrix, each row of which shows the eigenvalues of the transition matric of each arm (channel), or a empty list if set_decom is False
               - A_inv: if set_decom is True, a (N, M, M) tensor, each matrix of which shows the inverse of the each matrix of A, or a empty list if set_decom is False
               - all_type: if set_decom is True, a list contains the parameters type1_pos, type2_pos and type3_pos of the class. 
        """
        set_decom, A, J, A_inv, all_type = param
        if set_decom:
            self.A = A.copy(); self.J = J.copy()
            self.A_inv = A_inv.copy()
            self.type1_pos, self.type2_pos, self.type3_pos = all_type
        else:
            self.A, self.J = nplg.eig(self.P)
            self.A_inv = nplg.inv(self.A)
            J1 = np.array([np.diag(J[k]) for k in range(self.N)])
            P1 = self.A@J1@self.A_inv
            self.type2_pos = (((P1 - self.P)**2).sum(axis = 1).sum(axis = 1) > 1e-7)
            if self.type2_pos.any():
                for l in range(self.N):
                    if self.type2_pos[l]:
                        P2 = Matrix(self.P[l])
                        A, _ = P2.jordan_form()
                        self.A[l] = np.array(A)
                        self.A_inv[l] = nplg.inv(self.A[l])
            
            self.type3_pos = (((self.J[:, 0] - self.J[:, 1]).imag) > 1e-5)
            self.type1_pos = True^(self.type2_pos + self.type3_pos)
        
        self.right_side = (A_inv@self.reward.reshape((-1, 3, 1))).reshape((-1, 3))

        pos1 = (np.abs(self.J[:, 0] - 1) < 1e-6)
        pos2 = (np.abs(self.J[:, 1] - 1) < 1e-6)
        pos3 = (np.abs(self.J[:, 2] - 1) < 1e-6)
        self.J[pos1, 0] = 1
        self.J[pos2, 1] = 1
        self.J[pos3, 2] = 1
        self.pos = [pos1, pos2, pos3]

        if self.type3_pos.any():
            self.b1 = self.b1.astype(complex)
            self.b2 = self.b2.astype(complex)

            self.a1b1B = self.A[self.type3_pos, :, 1].reshape((-1, 3, 1))@self.A_inv[self.type3_pos, 1].reshape((-1, 1, 3))@self.reward[self.type3_pos].reshape((-1, 3, 1))
            self.complex_eign = self.J[self.type3_pos, 1]

            temp_pos1 = self.type3_pos*pos1; temp_pos2 = self.type3_pos*pos3
            self.a0b0B[temp_pos1] = self.A[temp_pos1, :, 0].reshape((-1, 3, 1))@self.A_inv[temp_pos1, 0].reshape((-1, 1, 3))@self.reward[temp_pos1].reshape((-1, 3, 1))
            self.a0b0B[temp_pos2] = self.A[temp_pos2, :, 2].reshape((-1, 3, 1))@self.A_inv[temp_pos2, 2].reshape((-1, 1, 3))@self.reward[temp_pos2].reshape((-1, 3, 1))
            self.a0b0B = self.a0b0B[self.type3_pos].reshape((-1, 3, 1))
            
        self.b1[pos1] = J[pos1, 1]; self.b2[pos1] = J[pos1, 2]
        self.b1[pos2] = J[pos2, 0]; self.b2[pos2] = J[pos2, 2]
        self.b1[pos3] = J[pos3, 0]; self.b2[pos3] = J[pos3, 1]

    def type1_find_ac(self, omega):
        """
        Function:
        -Compute the current cofficients of each term of k-step reward function f_1(k) = a1*b1**k + a2*b2**k + c.

        Input:
        - omega: a (N, M) array showing the current belief state ofr k-step reward function

        Output:
        - a1: a (N,) array for the cofficients of the first exponential term b1**k in function f(k)
        - a2: a (N,) array for the cofficients of the second exponential term b2**k in function f(k)
        - c: a (N,) array for the constant term of the function f(k)
        """

        a1 = np.zeros(self.N); a2 = np.zeros(self.N)
        c = np.zeros(self.N)
        left_side = np.zeros((self.N, 3)); cofficient = np.zeros((self.N, 3))

        if type(self.A[0,0,0]) == np.complex128:
            left_side[self.type1_pos] = (omega.reshape((-1, 1, 3))@self.A[self.type1_pos].real.reshape((-1, 3, 3))).reshape((-1, 3))
            cofficient[self.type1_pos] = left_side[self.type1_pos]*self.right_side[self.type1_pos].real
        
        else:
            left_side[self.type1_pos] = (omega.reshape((-1, 1, 3))@self.A[self.type1_pos].reshape((-1, 3, 3))).reshape((-1, 3))
            cofficient[self.type1_pos] = left_side[self.type1_pos]*self.right_side[self.type1_pos]

        pos1, pos2, pos3 = self.pos.copy()
        pos1 = pos1*self.type1_pos; pos2 = pos2*self.type1_pos; pos3 = pos3*self.type1_pos

        a1[pos1] = cofficient[pos1, 1]; a2[pos1] = cofficient[pos1, 2]
        c[pos1] = cofficient[pos1, 0]

        a1[pos2] = cofficient[pos2, 0]; a2[pos2] = cofficient[pos2, 2]
        c[pos2] = cofficient[pos2, 1]

        a1[pos3] = cofficient[pos3, 0]; a2[pos3] = cofficient[pos3, 1]
        c[pos3] = cofficient[pos3, 2]

        return a1, a2, c
    
    def type2_find_acd(self, omega):
        """
        Function:
        -Compute the current cofficients of each term of k-step reward function f_2(k) = ab**k + c*k*b**(k - 1) + d.

        Input:
        - omega: a (N, M) array showing the current belief state ofr k-step reward function

        Output:
        - a: a (N,) array for the cofficients of the first exponential term b**k in function f_2(k)
        - c: a (N,) array for the cofficients of the linear and exponential product term k*b**(k - 1) in function f_2(k)
        - d: a (N,) array for the constant term of the function f_2(k)
        """

        a = np.zeros(self.N); c = np.zeros(self.N); d = np.zeros(self.N)
        left_side = np.zeros((self.N, 3))
        cofficient = np.zeros((self.N, 3))

        if type(self.A[0,0,0]) == np.complex128:
            left_side[self.type2_pos] = (omega.reshape((-1, 1, 3))@self.A[self.type2_pos].real.reshape((-1, 3, 3))).reshape(left_side[self.type2_pos].shape)
            cofficient[self.type2_pos] = left_side[self.type2_pos]*self.right_side[self.type2_pos].real

        else:
            left_side[self.type2_pos] = (omega.reshape((-1, 1, 3))@self.A[self.type2_pos].reshape((-1, 3, 3))).reshape(left_side[self.type2_pos].shape)
            cofficient[self.type2_pos] = left_side[self.type2_pos]*self.right_side[self.type2_pos]

        pos1, _, pos3 = self.pos.copy()
        pos1 = pos1*self.type2_pos; pos3 = pos3*self.type2_pos

        a[pos1] = cofficient[pos1, 1] + cofficient[pos1, 2]
        c[pos1] = left_side[pos1, 1]*self.right_side[pos1, 2]
        d[pos1] = cofficient[pos1, 0]

        a[pos3] = cofficient[pos3, 0] + cofficient[pos3, 1]
        c[pos3] = left_side[pos3, 0]*self.right_side[pos3, 1]
        d[pos3] = cofficient[pos3, 2]

        return a, c, d

    def type1_L_omega(self, a1, a2, c, target):
        """
        Function:
        - Compute the first crossing time of type-one k-step reward function f_1(k).

        Output:
        - L: a (N,) array showing the first crossing time of each arm(channel)
        """

        L = first_crossing_time_type1(a1, a2, self.b1[self.type1_pos], self.b2[self.type1_pos], c, target)
        return L
    
    def type2_L_omega(self, a, c, d, target):
        """
        Function:
        - Compute the first crossing time of type-two k-step reward function f_2(k).

        Input:
        - omega_star: a (N, M) matrix, each row of which shows the target belief state at which the k-step reward function want to arrive

        Output:
        - L: a (N,) array showing the first crossing time of each arm(channel)
        """

        L = first_crossing_time_type2(a, self.b1[self.type2_pos], c, d, target)
        return L

    def type3_L_omega(self, omega, target):
        """
        Function:
        - Compute the first crossing time of type-three k-step reward function f_3(k).

        Output:
        - L: a (N,) array showing the first crossing time of each arm(channel)
        """

        wa0b0B = omega.reshape((-1, 1, 3))@self.a0b0B
        L = first_crossing_time_type3(omega, self.P[self.type3_pos], self.reward[self.type3_pos], self.complex_eign, wa0b0B, self.a1b1B, target)
        return L
    
    def P_power_inf(self, lambd, D, D_inv):
        """
        Function:
        - Compute the infinite power of specified matrix.

        Input:
        - lambd: a (, M) matrix, each row of which shows the eigenvalue of each matrix we want to compute the power
        - D: a (, M, M) tensor, each matrix of which shows the eigenvectors of each matrix we want to compute the power
        - D_inv: a (, M, M) tensor, each matrix of which shows the inverse of each matrix of D

        Output:
        - P_inf: a (, M, M) tensor, each matrix of which shows the result power of each given matrix
        """
        if (type(lambd[0]) == np.float64):
            lambd_inf = lambd**np.inf
        else:
            lambd_inf = lambd.real.copy()
            lambd_inf[abs(lambd_inf - 1) > 1e-10] = 0
            lambd_inf = lambd_inf.astype(complex)

        P_inf = np.zeros(np.shape(D))
        for k in range(lambd.shape[0]):
            P_inf[k] = (D[k]@np.diag(lambd_inf[k])@D_inv[k]).real
        
        return P_inf

    def P_power(self, k):
        """
        Function:
        - Compute the kth power of transition matrix of each arm(channel).

        Input:
        - k: a (N,) array each element of which shows the power each matrix need to arrive at

        Output:
        - P_k: a (N, M, M) tensor, each matrix of which shows the computing result of each arm(channel)
        """
        j = 0
        k1 = k.copy()
        pos_inf = (k == np.inf)
        k1[pos_inf] = 0
        P_k = np.array([np.eye(3) for l in range(np.size(k1))])
        while (j < k1).any():
            P_k[j < k1] = P_k[j < k1]@self.P[j < k1]
            j = j + 1
        if pos_inf.any():
            P_k[pos_inf] = self.P_power_inf(self.J[pos_inf], self.A[pos_inf], self.A_inv[pos_inf])
        
        return P_k

    def Whittle_index(self, beta):
        """
        Function:
        - Compute the whittle index for each arm.

        Input:
        - beta: a scalar indicating the discounted factor

        Output:
        whittle_idx: a (N,) array showing the whittle index of each arm
        """
        L_p0 = np.zeros(self.N); L_p1 = np.zeros(self.N)
        L_p2 = np.zeros(self.N); L_wP = np.zeros(self.N)
        wP = (self.belief_vec.reshape((-1, 1, 3))@self.P).copy().reshape((-1, 3))
        target = np.sum(self.belief_vec.reshape((-1, 3))*self.reward, axis = 1)

        if self.type1_pos.any():
            a_1_10, a_1_20, c_1_0 = self.type1_find_ac(self.P[self.type1_pos, 0])
            a_1_11, a_1_21, c_1_1 = self.type1_find_ac(self.P[self.type1_pos, 1])
            a_1_12, a_1_22, c_1_2 = self.type1_find_ac(self.P[self.type1_pos, 2])
            a_1_1wP, a_1_2wP, c_1_wP = self.type1_find_ac(wP[self.type1_pos])
            L_p0[self.type1_pos] = self.type1_L_omega(a_1_10[self.type1_pos], a_1_20[self.type1_pos], 
                                            c_1_0[self.type1_pos], target[self.type1_pos])
            L_p1[self.type1_pos] = self.type1_L_omega(a_1_11[self.type1_pos], a_1_21[self.type1_pos], 
                                            c_1_1[self.type1_pos], target[self.type1_pos])
            L_p2[self.type1_pos] = self.type1_L_omega(a_1_12[self.type1_pos], a_1_22[self.type1_pos], 
                                            c_1_2[self.type1_pos], target[self.type1_pos]) 
            L_wP[self.type1_pos] = self.type1_L_omega(a_1_1wP[self.type1_pos], a_1_2wP[self.type1_pos], 
                                            c_1_wP[self.type1_pos], target[self.type1_pos])          
        if self.type2_pos.any():
            a_2_0, c_2_0, d_2_0 = self.type2_find_acd(self.P[self.type2_pos, 0])
            a_2_1, c_2_1, d_2_1 = self.type2_find_acd(self.P[self.type2_pos, 1])
            a_2_2, c_2_2, d_2_2 = self.type2_find_acd(self.P[self.type2_pos, 2])
            a_2_wP, c_2_wP, d_2_wP = self.type2_find_acd(wP[self.type2_pos])

            L_p0[self.type2_pos] = self.type2_L_omega(a_2_0[self.type2_pos], c_2_0[self.type2_pos], 
                                                d_2_0[self.type2_pos], target[self.type2_pos])
            L_p1[self.type2_pos] = self.type2_L_omega(a_2_1[self.type2_pos], c_2_1[self.type2_pos], 
                                                d_2_1[self.type2_pos], target[self.type2_pos]) 
            L_p2[self.type2_pos] = self.type2_L_omega(a_2_2[self.type2_pos], c_2_2[self.type2_pos], 
                                                d_2_2[self.type2_pos], target[self.type2_pos]) 
            L_wP[self.type2_pos] = self.type2_L_omega(a_2_wP[self.type2_pos], c_2_wP[self.type2_pos], 
                                                  d_2_wP[self.type2_pos], target[self.type2_pos])
        
        if self.type3_pos.any():
            L_p0[self.type3_pos] = self.type3_L_omega(self.P[self.type3_pos, 0], target[self.type3_pos])
            L_p1[self.type3_pos] = self.type3_L_omega(self.P[self.type3_pos, 1], target[self.type3_pos])
            L_p2[self.type3_pos] = self.type3_L_omega(self.P[self.type3_pos, 2], target[self.type3_pos])
            L_wP[self.type3_pos] = self.type3_L_omega(wP[self.type3_pos], target[self.type3_pos])

        beta_Lp0 = beta**L_p0; beta_Lp1 = beta**L_p1
        beta_Lp2 = beta**L_p2; beta_LwP = beta**L_wP

        p0P_Lp0 = beta_Lp0.reshape((-1, 1, 1))*(self.P[:, 0].reshape((-1, 1, 3))@self.P_power(L_p0)).reshape((-1, 1, 3))
        p1P_Lp1 = beta_Lp1.reshape((-1, 1, 1))*(self.P[:, 1].reshape((-1, 1, 3))@self.P_power(L_p1)).reshape((-1, 1, 3))
        p2P_Lp2 = beta_Lp2.reshape((-1, 1, 1))*(self.P[:, 2].reshape((-1, 1, 3))@self.P_power(L_p2)).reshape((-1, 1, 3))
        wPP_LwP = beta_LwP.reshape((-1, 1, 1))*(wP.reshape((-1, 1, 3))@self.P_power(L_wP)).reshape((-1, 1, 3))

        temp1 = np.array([beta_Lp0, beta_Lp1, beta_Lp2])
        temp1 = np.array([temp1[:, k] for k in range(self.N)]).reshape((-1, 3, 1))

        F_P = (1 - temp1)/(1 - beta)

        G_P = np.zeros((self.N, 3, 3))
        for i in range(self.N):
            G_P[i] = np.array([p0P_Lp0[i], p1P_Lp1[i], p2P_Lp2[i]]).reshape(3, 3)
        f_wP = ((1 - beta_LwP)/(1 - beta)).reshape((-1, 1, 1))
        g_wP = wPP_LwP

        I_betaGP = nplg.inv(np.eye(3) - beta*G_P)
        w = self.belief_vec.copy().reshape((-1, 1, 3))
        B = self.reward.copy().reshape((-1, 3, 1))
        temp_val = beta*I_betaGP@G_P
        numerator = w@B - beta*g_wP@(np.eye(3) + temp_val)@B + w@temp_val@B
        denominator = 1 + beta*f_wP + beta*(beta*g_wP - w)@(I_betaGP)@F_P
        whittle_idx = (numerator/denominator).reshape(self.N)
        return whittle_idx

    
    def belief_vec_update(self):
        """
        Function:
        - Update the belief vector according to the action of each arm(channel).
        """
        self.belief_vec[self.action == 1] = self.mabm.tran_prob[self.action == 1]
        idx0 = (self.action == 0)
        self.belief_vec[idx0] = (self.belief_vec[idx0].reshape((self.N - self.K, 1, 3))@self.mabm.P[idx0, :, :]).reshape((self.N - self.K, 3))

    def action_AR_update(self, beta):
        """
        Function:
        - Find the active arms(channels) of each time according to the whittle indices of them.
        - Compute actual reward according to the action of each time.
        """
        whittle_idx = self.Whittle_index(beta)
        self.action = np.zeros(self.N)
        self.action[whittle_idx.argsort()[::-1][0:self.K]] = 1
        self.AR = np.append(self.AR, np.sum(self.reward[self.action == 1, self.mabm.state[self.action == 1]]))

    def User_process(self, beta, param):
        """
        Function:
        - Simulate the game process of a user, who uses whittle index policy, with above functions.
        """
        self.matrics_decomposition(param)

        for t in range(self.T):
            self.mabm.MABM_update()
            self.action_AR_update(beta)
            self.belief_vec_update()
