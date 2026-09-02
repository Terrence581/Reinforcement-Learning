import numpy as np
import numpy.linalg as nplg

"""
Introduction:
- This file contains all the functions required to compute the first crossing time of 3D Markov Chain.

- The main function is first_crossing_time, with proper parameters, it could return the first crossing time
  for which the belief state distribution, stochastic matrix and reward vector have been specified.

- This file could compute the first crossing time for totally three kinds of stochastic matrics:
  - type1: the matrix could be diagonalized, whose k-step reward function could be formulated as 
           f_1(k) = a1*b1**k + a2*b2**k + c, where b1 and b2 are the real eigenvalues of the matrix 
           and are in [-1, 1].

  - type2: the matrix could not be diagonalized, due to the linear dependent eigenvectors, so the 
           corresponding k-step reward function has the form: f_2(k) = ab**k + ckb**(k - 1) + d, 
           where b is the real non-one eigenvalue and is in (-1, 1).

  - type3: the matrix could be diagonalized, but has comjugate complex eigenvlaues, and thus its 
           k-step reward function could be formulated as f_3(k) = a(A**k)sin(k\theta + b).
"""


def _real_type1_array(value):
    arr = np.asarray(np.real_if_close(value, tol=1000))
    if np.iscomplexobj(arr):
        if arr.size and np.max(np.abs(arr.imag)) > 1e-8:
            raise ValueError("type1 first-crossing-time inputs must be real.")
        arr = arr.real
    return arr.astype(float, copy=False)




def find_odd(num):
    """
    Function:
    - Find the biggest odd numbers that less than the target numbers you specified.

    Input:
    - num: an array for the target numbers you specified

    Output:
    - result: an array for the biggest odd number you find
    """
    result = num.copy()
    result[num < 1] = -1
    even_pos = (result%2 == 0)*(result > 0)
    result[even_pos] = result[even_pos] - 1
    return result

def find_even(num):
    """
    Function:
    - Find the biggest even numbers that less than the target numbers you specified.

    Input:
    - num: an array for the target numbers you specified
    
    Output:
    - result: an array for the biggest even number you find
    """
    result = num.copy()
    result[num < 0] = -2
    odd_pos = (result%2 != 0)*(result > 0)
    result[odd_pos] = result[odd_pos] - 1
    return result

def type1_zero(a1, a2, c, target):
    """
    Function:
    - Compute the first crossing time for those type1 stochastic matrics whose k-step rewards
      get their maximum at k = 0, so their first crossing time may be 0 for infinite.

    Input:
    - a1: an array for the cofficients of the first exponential term b1**k in function f_1(k)
    - a2: an array for the cofficients of the second exponential term b2**k in function f_1(k)
    - c: an array for the constant term of the function f_1(k)
    - target: an array for the target reward at which the function f_1(k) want to arrive

    Output:
    - L: an array for the first crossing time 
    """

    L = np.zeros(np.size(a1))
    L[((a1 + a2 + c) - target < 1e-15).reshape(-1)] = np.inf
    return L

def type1_enu_station(a1, a2, b1, b2, c, target):
    """
    Function: 
    - Compute the first crossing time by enumeration for those type2 stochastics matrics whose k-step 
      rewards may not have the maximum or may obtain its maximum at the infinity, so their first crossing 
      time may be any finite number when they have large stationary reward and infinite when they have 
      small stationary reward.
    
    Input:
    - b1: an array for the base of the first expoenetial term b1**k in function f_1(k)
    - b2: an array for the base of the second expoenetial term b2**k in function f_1(k)

    Output:
    - L: an array for the first crossing time  
    """
    L = np.zeros(np.size(c))
    nore = (c - target > 1e-15).reshape(-1)
    L[True^nore] = np.inf
    k = 0
    while nore.any():
        reach = ((a1*(b1**k) + a2*(b2**k) + c) - target > 1e-15).reshape(-1)
        reach[nore == False] = True
        L[nore*reach] = k
        nore = True^reach
        nore[L == np.inf] = False
        k = k + 1
    return L

def type1_enu_k(a1, a2, b1, b2, c, k, target):
    """
    Function:
    - Compute the first crossing time for those type1 stochastic matrics whose k-step reward get 
      their maximum at k, so their first crossing time may be any number less than k for infinite.
    
    Input:
    - k: an array for the argument of the maximum of function f_1(k)

    Output:
    - L: an array for the first crossing time   
    """
    L = np.zeros(np.size(c))
    max_f = a1*(b1**k) + a2*(b2**k) + c
    nore = (max_f - target > 1e-15).reshape(-1)
    L[True^nore] = np.inf
    j = 0
    while nore.any():
        reach = ((a1*(b1**j) + a2*(b2**j) + c) - target > 1e-15).reshape(-1)
        reach[nore == False] = True
        L[nore*reach] = j
        nore = (True^reach)
        nore[L == np.inf] = False
        j = j + 1
    return L

def type1_one_param(a1_origin, a2_origin, b1_origin, b2_origin, c_origin, target_origin):
    """
    Function:
    - Compute the first crossing time for those type1 stochastic matrics whose k-step reward could be 
      reduced to one exponential term, that is f_1(k) = a1*b1**k + c
    
    Output:
    - L: an array for the first crossing time  
    """
    a1 = _real_type1_array(a1_origin).copy().reshape(-1)
    a2 = _real_type1_array(a2_origin).copy().reshape(-1)
    b1 = _real_type1_array(b1_origin).copy().reshape(-1)
    b2 = _real_type1_array(b2_origin).copy().reshape(-1)
    c = _real_type1_array(c_origin).copy().reshape(-1)
    target = _real_type1_array(target_origin).copy().reshape(-1)
    L = np.zeros(np.size(a1))
    #print(a1, b1, c)

    abs_a1 = np.abs(a1); abs_a2 = np.abs(a2)
    abs_b1 = np.abs(b1); abs_b2 = np.abs(b2)
    case0 = ((abs_a1 < 1e-15) + (abs_b1 < 1e-15))*((abs_a2 < 1e-15) + (abs_b2 < 1e-15))
    non_case0 = True^case0
    temp_pos1 = ((abs_a1 < 1e-15) + (abs_b1 < 1e-15))*non_case0
    temp_a1 = a1.copy(); temp_b1 = b1.copy()
    a1[temp_pos1] = a2[temp_pos1]; a2[temp_pos1] = temp_a1[temp_pos1]
    b1[temp_pos1] = b2[temp_pos1]; b2[temp_pos1] = temp_b1[temp_pos1]

    case1 = (a1 > 0)*(b1 > 0)*non_case0
    case2 = (a1 < 0)*(b1 > 0)*non_case0
    case3 = (a1 > 0)*(b1 < 0)*non_case0
    case4 = (a1 < 0)*(b1 < 0)*non_case0

    if case0.any():
        case01 = case0*(target - c >= -1e-15); case02 = case0*(target - c < -1e-15)
        L[case01] = np.inf
        L[case02] = 0
    
    if case1.any():
        case11 = case1*(a1 + c - target > 1e-15); case12 = case1*(a1 + c - target <= 1e-15)
        if case11.any():
            L[case11] = 0
        if case12.any():
            L[case12] = np.inf
    if case2.any():
        case21 = case2*(c - target > 1e-15); case22 = case2*(c - target <= 1e-15)
        if case21.any():
            #print(a1, b1, c)
            L[case21] = np.floor(np.log((target[case21] - c[case21])/a1[case21])/np.log(b1[case21])).astype(int) + 1
        if case22.any():
            L[case22] = np.inf
    
    if case3.any():
        case31 = case1*(a1 + c - target > 1e-15); case32 = case3*(a1 + c - target <= 1e-15)
        if case31.any():
            L[case31] = 0
        if case32.any():
            L[case32] = np.inf
    
    if case4.any():
        case41 = case4*(a1 + c - target > 1e-15); case42 = case4*(a1 + c - target <= 1e-15)
        if case41.any():
            L[case41] = 0
        if case42.any():
            case421 = case42*(a1*b1 + c - target > 1e-15); case422 = case42*(a1*b1 + c - target <= 1e-15)
            if case421.any():
                L[case421] = 1
            if case422.any():
                L[case422] = np.inf

    return L

def first_crossing_time_type1(a1_origin, a2_origin, b1_origin, b2_origin, c_origin, target_origin):
    """
    Function:
    - Compute the first crossing time for the k-step rewards of those type1 stochastic matrics.
    
    Output:
    - L: an array for the first crossing time  
    """
    a1 = _real_type1_array(a1_origin).copy().reshape(-1)
    a2 = _real_type1_array(a2_origin).copy().reshape(-1)
    b1 = _real_type1_array(b1_origin).copy().reshape(-1)
    b2 = _real_type1_array(b2_origin).copy().reshape(-1)
    c = _real_type1_array(c_origin).copy().reshape(-1)
    target = _real_type1_array(target_origin).copy().reshape(-1)
    L = np.zeros(np.size(c))
    abs_a1 = np.abs(b1); abs_a2 = np.abs(a2)
    abs_b1 = np.abs(b1); abs_b2 = np.abs(b2)

    case0 = (abs_a1 < 1e-15) + (abs_a2 < 1e-15) + (abs_b1 < 1e-15) + (abs_b2 < 1e-15)
    L[case0] = type1_one_param(a1[case0], a2[case0], b1[case0], b2[case0], c[case0], target[case0])
    
    temp_a1 = a1.copy(); temp_b1 = b1.copy() 
    non_case0 = True^case0
    
    
    case1 = (a1 > 0)*(a2 > 0)*(b1 > 0)*(b2 > 0)*non_case0
    temp_pos1 = (a2 < 0)*(a1 > 0)*(b1 > 0)*(b2 > 0)
    a1[temp_pos1] = a2[temp_pos1]; a2[temp_pos1] = temp_a1[temp_pos1]
    b1[temp_pos1] = b2[temp_pos1]; b2[temp_pos1] = temp_b1[temp_pos1]
    case2 = (a1 < 0)*(a2 > 0)*(b1 > 0)*(b2 > 0)*non_case0
    temp_pos2 = (b2 < 0)*(a1 > 0)*(a2 > 0)*(b1 > 0)
    a1[temp_pos2] = a2[temp_pos2]; a2[temp_pos2] = temp_a1[temp_pos2]
    b1[temp_pos2] = b2[temp_pos2]; b2[temp_pos2] = temp_b1[temp_pos2]
    case3 = (b1 < 0)*(a1 > 0)*(a2 > 0)*(b2 > 0)*non_case0
    temp_pos3 = (a2 < 0)*(b2 < 0)*(a1 > 0)*(b1 > 0)
    a1[temp_pos3] = a2[temp_pos3]; a2[temp_pos3] = temp_a1[temp_pos3]
    b1[temp_pos3] = b2[temp_pos3]; b2[temp_pos3] = temp_b1[temp_pos3]
    case4 = (a1 < 0)*(b1 < 0)*(a2 > 0)*(b2 > 0)*non_case0
    temp_pos4 = (a1 < 0)*(b2 < 0)*(a2 > 0)*(b1 > 0)
    a1[temp_pos4] = a2[temp_pos4]; a2[temp_pos4] = temp_a1[temp_pos4]
    b1[temp_pos4] = b2[temp_pos4]; b2[temp_pos4] = temp_b1[temp_pos4]
    case5 = (a2 < 0)*(b1 < 0)*(a1 > 0)*(b2 > 0)*non_case0
    case6 = (b1 < 0)*(b2 < 0)*(a1 > 0)*(a2 > 0)*non_case0
    case7 = (a1 < 0)*(a2 < 0)*(b1 > 0)*(b2 > 0)*non_case0
    temp_pos5 = (a1 < 0)*(a2 < 0)*(b2 < 0)*(b1 > 0)
    a1[temp_pos5] = a2[temp_pos5]; a2[temp_pos5] = temp_a1[temp_pos5]
    b1[temp_pos5] = b2[temp_pos5]; b2[temp_pos5] = temp_b1[temp_pos5]
    case8 = (a1 < 0)*(a2 < 0)*(b1 < 0)*(b2 > 0)*non_case0
    temp_pos6 = (a1 < 0)*(b1 < 0)*(b2 < 0)*(a2 > 0)
    a1[temp_pos6] = a2[temp_pos6]; a2[temp_pos6] = temp_a1[temp_pos6]
    b1[temp_pos6] = b2[temp_pos6]; b2[temp_pos6] = temp_b1[temp_pos6]
    case9 = (a2 < 0)*(b1 < 0)*(b2 < 0)*(a1 > 0)*non_case0
    case10 = (a1 < 0)*(a2 < 0)*(b1 < 0)*(b2 < 0)*non_case0
    
    abs_b1 = np.abs(b1); abs_b2 = np.abs(b2)
    b1_g_pos = (abs_b1 - abs_b2 > 1e-15); b1_l_pos = (abs_b1 - abs_b2 < -1e-15)
    b1_e_pos = (abs_b1 - abs_b2 > -1e-15)*(abs_b1 - abs_b2 < 1e-15)
    f0 = a1 + a2 + c; f1 = a1*b1 + a2*b2 + c
    k1 = np.zeros(np.size(b1)); k2 = np.zeros(np.size(b2))
    k1[non_case0] = np.log(np.abs(a2[non_case0]*(b2[non_case0] - 1)/(a1[non_case0]*(b1[non_case0] - 1))))/np.log(np.abs(b1[non_case0]/b2[non_case0]))
    #print(k1)
    k1[k1 < 0] = -1
    k1 = k1.astype(int)
    #print(k1)
    k2[non_case0] = np.log(np.abs(a2[non_case0]*(b2[non_case0]**2 - 1)/(a1[non_case0]*(b1[non_case0]**2 - 1))))/np.log(np.abs(b1[non_case0]/b2[non_case0]))
    k2[k2 < 0] = -1
    k2 = k2.astype(int)
    k2_even = find_even(k2); k2_odd = find_odd(k2)

    if case1.any():
        #print(case1)
        L[case1] = type1_zero(a1[case1], a2[case1], c[case1], target[case1])
    
    

    if case2.any():
        case21 = case2*(a1 + a2 >= -1e-15)*b1_g_pos; case22 = case2*(a1 + a2 < -1e-15)*b1_g_pos
        if case21.any():
            L[case21] = type1_zero(a1[case21], a2[case21], c[case21], target[case21])
        if case22.any():
            L[case22] = type1_enu_station(a1[case22], a2[case22], b1[case22], b2[case22], c[case22], target[case22])

        case23 = case2*b1_l_pos
        k11 = k1 + 1
        case231 = case23*(k11 > 0); case232 = case23*(k11 <= 0)
        if case231.any():
            L[case231] = type1_enu_k(a1[case231], a2[case231], b1[case231], b2[case231],c[case231], k11[case231], target[case231])
        if case232.any():
            L[case232] = type1_zero(a1[case232], a2[case232], c[case232], target[case232])

    if case3.any():
        L[case3] = type1_zero(a1[case3], a2[case3], c[case3], target[case3])


    if case4.any():
        case41 = case4*(f0 - f1 < -1e-15); case42 = case4*(f0 - f1 >= -1e-15)
        if case41.any():
            L[case41] = type1_enu_k(a1[case41], a2[case41], b1[case41], b2[case41], c[case41], 1, target[case41])
        if case42.any():
            L[case42] = type1_zero(a1[case42], a2[case42], c[case42], target[case42])


    if case5.any():
        case51 = case5*b1_g_pos; case52 = case5*b1_l_pos; case53 = case5*b1_e_pos
        case511 = case51*(k2_even >= -1e-15); case512 = case51*(k2_even < -1e-15)
        #print(case511, case512)
        if case511.any():
            L[case511] = type1_enu_k(a1[case511], a2[case511], b1[case511], b2[case511], c[case511], k2_even[case511] + 2, target[case511])
        if case512.any():    
            L[case512] = type1_zero(a1[case512], a2[case512], c[case512], target[case512])
        
        case521 = case52*(k2_even >= -1e-15)*(f0 - c >= -1e-15)
        case522 = case52*(k2_even >= -1e-15)*(f0 - c < -1e-15) + case52*(k2_even < -1e-15)
        if case521.any():
            L[case521] = type1_zero(a1[case521], a2[case521], c[case521], target[case521])
        if case522.any():
            L[case522] = type1_enu_station(a1[case522], a2[case522], b1[case522], b2[case522], c[case522], target[case522])
        temp_pos7 = (-a2*(b2 - 1)/(a1*(b1 - 1)) - 1 >= -1e-15) 
        temp_pos8 = (-a2*(b2**2 - 1)/(a1*(b1**2 - 1)) - 1 >= -1e-15)
        case531 = case53*temp_pos7 + case53*(True^temp_pos7)*temp_pos8
        case532 = case53*(True^temp_pos7)*(True^temp_pos8)
        if case531.any():
            L[case531] = type1_enu_station(a1[case531], a2[case531], b1[case531], b2[case531], c[case531], target[case531])
        if case532.any():
            L[case532] = type1_zero(a1[case532], a2[case532], c[case532], target[case532])


    if case6.any():
        L[case6] = type1_zero(a1[case6], a2[case6], c[case6], target[case6])
    if case7.any():
        L[case7] = type1_enu_station(a1[case7], a2[case7], b1[case7], b2[case7], c[case7], target[case7])

    if case8.any():
        case81 = case8*b1_g_pos; case82 = case8*b1_l_pos; case83 = case8*b1_e_pos
        case811 = case81*(k2_odd >= -1e-15); case812 = case81*(k2_odd < -1e-15)
        if case811.any():
            L[case811] = type1_enu_k(a1[case811], a2[case811], b1[case811], b2[case811], c[case811], k2_odd[case811] + 2, target[case811])
        if case812.any():
            L[case812] = type1_enu_k(a1[case812], a2[case812], b1[case812], b2[case812], c[case812], 1, target[case812])

        case821 = case82*(k2_odd >= -1e-15)*(f1 - c >= -1e-15)
        case822 = case82*(k2_odd >= -1e-15)*(f1 - c < -1e-15) + case82*(k2_odd < -1e-15)
        if case821.any():
            L[case821] = type1_enu_k(a1[case821], a2[case821], b1[case821], b2[case821], c[case821], 1, target[case821])
        if case822.any():
            L[case822] = type1_enu_station(a1[case822], a2[case822], b1[case822], b2[case822], c[case822], target[case822])

        temp_pos9 = (-a2*(b2 - 1)/(a1*(b1 - 1)) > -1); temp_pos10 = (-a2*(b2**2 - 1)/(a1*(b1**2 - 1)) < -1)
        case831 = case83*(True^temp_pos9) + case83*temp_pos9*temp_pos10; case832 = case83*temp_pos9*(True^temp_pos10)    
        if case831.any():
            L[case831] = type1_enu_station(a1[case831], a2[case831], b1[case831], b2[case831], c[case831], target[case831])
        if case832.any():
            L[case832] = type1_enu_k(a1[case832], a2[case832], b1[case832], b2[case832], c[case832], 1, target[case832])

    if case9.any():
        case91 = case9*b1_g_pos; case92 = case9*b1_l_pos; case93 = case9*b1_e_pos
        f_k22 = a1*(b1**(k2_even + 2)) + a2*(b2**(k2_even + 2)) + c
        case911 = case91*(f1 - f0 > 1e-15)*(f1 - f_k22 >= -1e-15)
        case912 = case91*(f_k22 - f0 > 1e-15)*(f_k22 - f1 > 1e-15)
        case913 = case91*(f0 - f1 >= -1e-15)*(f0 - f_k22 >= -1e-15)
        if case911.any():
            L[case911] = type1_enu_k(a1[case911], a2[case911], b1[case911], b2[case911], c[case911], 1, target[case911])
        if case912.any():
            L[case912] = type1_enu_k(a1[case912], a2[case912], b1[case912], b2[case912], c[case912], k2_even[case912] + 2, target[case912])
        if case913.any():
            L[case913] = type1_zero(a1[case913], a2[case913], c[case913], target[case913])


        f_k12 = a1*(b1**(k2_odd + 2)) + a2*(b2**(k2_odd + 2)) + c
        case921 = case92*(f0 - f1 >= -1e-15)*(f0 - f_k12 >= -1e-15)
        case922 = case92*(f_k12 - f0 > 1e-15)*(f_k12 - f1 > 1e-15)
        case923 = case92*(f1 - f0 > 1e-15)*(f1 - f_k12 >= -1e-15)

        if case921.any():
            L[case921] = type1_zero(a1[case921], a2[case921], c[case921], target[case921])
        if case922.any():
            L[case922] = type1_enu_k(a1[case922], a2[case922], b1[case922], b2[case922], c[case922], k2_odd[case922] + 2, target[case922])
        if case923.any():
            L[case923] = type1_enu_k(a1[case923], a2[case923], b1[case923], b2[case923], c[case923], 1, target[case923])

        case931 = case93*(f0 >= f1); case932 = case93*(f0 < f1)
        if case931.any():
            L[case931] = type1_zero(a1[case931], a2[case931], c[case931], target[case931])
        if case932.any():
            L[case932] = type1_enu_k(a1[case932], a2[case932], b1[case932], b2[case932], c[case932], 1, target[case932])

    if case10.any():
        L[case10] = type1_enu_k(a1[case10], a2[case10], b1[case10], b2[case10], c[case10], 1, target[case10])

        #case = np.array([case1, case21, case22, case231, case232, case3, case41, case42, case511, case512, case521, case522, case531, case532, case6, case7, case811, case812, case821, case822, case831, case832, case911, case912, case913, case921, case922, case923, case931, case932, case10])
    #case = [case0, case1, case2, case3, case4, case5, case6, case7, case8, case9, case10]
        #case = np.array([case9, case91, case911, case912, case913, case92, case921, case922, case923, case93, case931, case932])
    return L


##############################################################################################################################
def type2_zero(a, d, target):
    """
    Function:
    - Compute the first crossing time for those type2 stochastic matrics whose k-step rewards
      get their maximum at k = 0, so their first crossing time may be 0 for infinite.

    Input:
    - a: an array for the cofficients of the first exponential term b**k in function f_2(k)
    - d: an array for the constant term of the function f_2(k)
    - target: an array for the target reward at which the function f_2(k) want to arrive

    Output:
    - L: an array for the first crossing time 
    """

    L = np.zeros(np.size(a))
    L[(a + d) - target < 1e-10] = np.inf
    return L


def type2_enu_k(a, b, c, d, k, target):
    """
    Function:
    - Compute the first crossing time for those type1 stochastic matrics whose k-step reward get 
      their maximum at k, so their first crossing time may be any number less than k for infinite.
    
    Input:
    - b: an array for the base of the expoenetial term b**k and b**(k-1) in function f_2(k)
    - c: an array for the cofficients of the linear and exponential product term k*b**k in function f_2(k)
    - k: an array for the argument of the maximum of function f_2(k)

    Output:
    - L: an array for the first crossing time   
    """

    L = np.zeros(np.size(c))
    max_f = a*(b**k) + c*k*(b**(k - 1)) + d
    nore = (max_f - target > 1e-10)
    L[True^nore] = np.inf
    j = 0
    while nore.any():
        reach = ((a*(b**j) + c*j*(b**(j - 1)) + d) - target > 1e-10)
        L[nore*reach] = j
        nore = True^reach
        nore[L == np.inf] = False
        j = j + 1
    return L

def type2_enu_station(a, b, c, d, target):
    """
    Function:
    - Compute the first crossing time by enumeration for those type2 stochastics matrics whose k-step 
      rewards may not have the maximum or may obtain its maximum at the infinity, so their first crossing 
      time may be any finite number when they have large stationary reward and infinite when they have 
      small stationary reward..

    Output:
    - L: an array for the first crossing time  
    """

    L = np.zeros(np.size(c))
    nore = (d - target > 1e-10)
    L[True^nore] = np.inf
    k = 0
    while nore.any():
        reach = ((a*(b**k) + c*k*(b**(k - 1)) + d) - target > 1e-10)
        L[nore*reach] = k
        nore = True^reach
        nore[L == np.inf] = False
        k = k + 1
    return L

def first_crossing_time_type2(a_origin, b_origin, c_origin, d_origin, target_origin):
    """
    Function:
    - Compute the first crossing time for the k-step rewards of those type1 stochastic matrics.
    
    Output:
    - L: an array for the first crossing time  
    """

    if type(a_origin) == np.float64:
        a_origin = np.array([a_origin]); b_origin = np.array([b_origin])
        c_origin = np.array([c_origin]); d_origin = np.array([d_origin])
        target_origin = np.array([target_origin])
    a = a_origin.copy(); b = b_origin.copy()
    c = c_origin.copy(); d = d_origin.copy()
    target = target_origin.copy()
    L = np.zeros(np.size(a))
    case_a_ne_0 = (abs(a) > 1e-10)
    case_b_ne_0 = (abs(b) > 1e-10)
    case_c_ne_0 = (abs(c) > 1e-10)
    case1 = (b > 1e-10)*(c > 1e-10)
    case2 = (b > 1e-10)*(c < -1e-10)
    case3 = (b < -1e-10)*(c > 1e-10)
    case4 = (b < -1e-10)*(c < -1e-10)
    case5 = True^case_b_ne_0 + (True^case_a_ne_0)*(True^case_c_ne_0)
    case6 = case_a_ne_0*case_b_ne_0(True^case_c_ne_0)

    k1 = (a*b - a*b*b - c*b)/(c*(b - 1))
    k1 = k1.astype(int)
    k2 = (a*b - a*b**3 - 2*c*b*b)/(c*(b*b - 1))
    k2 = k2.astype(int)
    k2_odd = find_odd(k2); k2_even = find_even(k2)
    f0 = a + d; f1 = a*b + c + d

    if case1.any():
        case11 = case1*(k1 >= 0); case12 = case1*(k1 < 0)

        if case11.any():
            L[case11] = type2_enu_k(a[case11], b[case11], c[case11], d[case11], k1[case11] + 1, target[case11])
        if case12.any():
            L[case12] = type2_zero(a[case12], d[case12], target[case12])

    if case2.any():
        case21 = case2*(a >= -1e-10); case22 = case2*(a < -1e-10)

        if case21.any():
            L[case21] = type2_zero(a[case21], d[case21], target[case21])
        if case22.any():
            L[case22] = type2_enu_station(a[case22], b[case22], c[case22], d[case22], target[case22])

    if case3.any():
        temp1 = a*(b**(k2_odd + 2)) + c*(k2_odd + 2)*(b**(k2_odd + 1)) + d
        case31 = case3*(k2_odd > 0)*(f0 >= temp1) + case3*(k2_odd < 0)
        case32 = case3*(k2_odd > 0)*(f0 < temp1)

        if case31.any():
            L[case31] = type2_zero(a[case31], d[case31], target[case31])
        if case32.any():
            L[case32] = type2_enu_k(a[case32], b[case32], c[case32], d[case32], k2_odd[case32] + 2, target[case32])

    if case4.any():
        temp2 = a*(b**(k2_even + 2)) + c*(k2_even + 2)*(b**(k2_even + 1)) + d
        case41 = case4*(k2_even >= 0)*(f1 >= temp2) + case4*(k2_even < 0)
        case42 = case4*(k2_even >= 0)*(f1 < temp2)

        if case41.any():
            L[case41] = type2_enu_k(a[case41], b[case41], c[case41], d[case41], 1, target[case41])
        if case42.any():
            L[case42] = type2_enu_k(a[case42], b[case42], c[case42], d[case42], k2_even[case42] + 2, target[case42])
    
    if case5.any():
        L[case5] = type2_enu_k(a[case5], b[case5], c[case5], d[case5], 1, target[case5])
    
    if case6.any():
        L[case6] = type1_one_param(a[case6], np.zeros(np.size(a[case6])), b[case6], np.zeros(np.size(a[case6])), c[case6], target[case6])


    return L



 ###########################################################################################################################################################   

def type3_enu_k(omega, transition_P, reward, target_origin, stop_time):
    """
    Function:
    - Compute the first crossing time for those type3 stochastic matrics whose k-step reward get 
      their maximum at k, so their first crossing time may be any number less than k for infinite.
    
    Input:
    - omega: a tensor for current belief state distributions 
    - transition_P: a tensor for the transition matrics
    - reward: a tensor for the reward vectors
    - target_origin: an array at the target reward the function f_3(k) want to arrive
    - stop_time: an array for the argument of the maximum of function f_3(k), which is also the  
      stopping time for the enumeration

    Output:
    - L: an array for the first crossing time   
    """

    w = omega.copy()
    P = transition_P.copy()
    B = reward.copy()
    target = target_origin.copy()
    L = np.zeros(np.size(target))

    wP_k = w.copy()
    wP_k_B = (wP_k@B).reshape(-1)
    nore = (wP_k_B - target <= 1e-10)
    #print(nore)
    k = 0
    complete = (k > stop_time)

    while (True^complete).any():
        k = k + 1
        wP_k[nore] = wP_k[nore]@P[nore]
        wP_k_B[nore] = (wP_k[nore]@B[nore]).reshape(-1)
        reach = (wP_k_B - target > 1e-10)
        #print(reach)
        L[nore*reach] = k
        nore[nore*reach] = False
        if (True^nore).all():
            break
        else:
            complete = complete + (k > stop_time)
    
    L[nore] = np.inf
    
    return L


def type3_enu(omega, transition_P, reward, target_origin):
    """
    Function:
    - Compute the first crossing time by enumeration for those type3 stochastics matrics whose k-step 
      rewards may not have the maximum or may obtain its maximum at the infinity, so their first crossing 
      time may be any finite number when they have large stationary reward and infinite when they have 
      small stationary reward.

    Output:
    - L: an array for the first crossing time  
    """

    w = omega.copy()
    P = transition_P.copy()
    B = reward.copy()
    target = target_origin.copy()
    L = np.zeros(np.size(target))

    wP_k = w.copy()
    wP_k_B = (wP_k@B).reshape(-1)
    nore = (wP_k_B - target <= 1e-10)
    k = 0

    while True:
        k = k + 1
        wP_k[nore] = wP_k[nore]@P[nore]
        wP_k_B[nore] = (wP_k[nore]@B[nore]).reshape(-1)
        reach = (wP_k_B - target > 1e-10)
        L[nore*reach] = k
        nore[nore*reach] = False
        if (True^nore).all():
            break
    
    return L


def type3_enu_period(omega, transition_P, reward, complex_eign, target_origin):
    """
    Function:
    - Compute the first crossing time for those type3 stochastics matrics which have a complex eigenvalue
      with norm one and so will be strictly periodic. The function will only enumerate those k-step rewards
      in a period of those matrics.
    
    Input:
    - complex_eign: a array for the complex eigenvalue used in simplifying k-step reward function

    Output:
    - L: an array for the first crossing time  
    """
    w = omega.copy()
    P = transition_P.copy()
    B = reward.copy()
    if type(complex_eign) == complex:
        lamda = np.array([complex_eign])
    else:
        lamda = complex_eign.copy()
    target = target_origin.copy()
    L = np.zeros(np.size(target))

    wP_k = w.copy()
    wP_k_B = (wP_k@B).reshape(-1)
    nore = (wP_k_B - target <= 1e-10)
    k = 0
    complete = (abs((lamda**k).real - 1) > 1e-10)
    while (True^complete).any():
        k = k + 1
        wP_k[nore] = wP_k[nore]@P[nore]
        wP_k_B[nore] = (wP_k[nore]@B[nore]).reshape(-1)
        reach = (wP_k_B - target > 1e-10)
        L[nore*reach] = k
        #print(L)
        nore[nore*reach] = False
        if (True^nore).all():
            break
        else:
            complete = complete + (abs((lamda**k).real - 1) <= 1e-10)
    
    L[nore] = np.inf
    
    return L


def first_crossing_time_type3(omega, transition_matrics, reward, complex_eign, station_reward, lamda_PB, target_origin):
    """
    Function:
    - Compute the first crossing time for the k-step rewards of those type3 stochastic matrics.
    
    Input: 
    - station_reward: an array for the stationary reward brought by the stochastic matrix, which is also 
      the constant d in function f_3(k)
    - lamda_PB: a tensor for the product of the right and left eigenvectors, which are corresponding to 
      the complex_eign, and the reward

    Output:
    - L: an array for the first crossing time  
    """

    w = omega.copy().reshape((-1, 1, 3))
    P = transition_matrics.copy().reshape((-1, 3, 3))
    B = reward.copy().reshape((-1, 3, 1))
    if type(complex_eign) == complex:
        lamda = np.array([complex_eign])
        target = np.array([target_origin])
        stat_B = np.array([station_reward])
    else:
        lamda = complex_eign.copy()
        target = target_origin.copy().reshape(-1)
        stat_B = station_reward.copy().reshape(-1)
    A = abs(lamda.copy()).reshape(-1)
    a1b1B = lamda_PB.copy().reshape((-1, 3, 1))
    a = abs((w@a1b1B).reshape(-1))
    L = np.zeros(np.size(target))
    d = (target - stat_B)/(2*a)
    
    case1 = abs(A - 1) < 1e-10
    case2 = abs(A - 1) > 1e-10

    if case1.any():
        L[case1] = type3_enu_period(w[case1], P[case1], B[case1], lamda[case1], target[case1])
    
    if case2.any():
        case21 = case2*(d > 1e-10); case22 = case2*(d < -1e-10); case23 = case2*(abs(d) <= 1e-10)

        if case21.any():
            case211 = case21*(1 - d > 1e-10); case212 = case21*(1 - d <= 1e-10)
            if case211.any():
                k211 = np.log(d[case211].real)/np.log(A[case211].real)
                if isinstance(k211, (float, np.floating)):
                    k211 = np.array([k211])
                L[case211] = type3_enu_k(w[case211], P[case211], B[case211], target[case211], k211)
            
            if case212.any():
                L[case212] = np.inf
        
        if case22.any():
            L[case22] = type3_enu(w[case22], P[case22], B[case22], target[case22])
        
        if case23.any():
            sinb = (w@a1b1B.real)/a
            case231 = (case23*(abs(sinb) < 1e-10)*(abs(lamda.imag/A + 1) < 1e-10)).reshape(-1)
            case232 = case23*(True^case231)
            
            if case231.any():
                L[case231] = np.inf
            
            if case232.any():
                L[case232] = type3_enu(w[case232], P[case232], B[case232], target[case232])
    
    return L

##############################################################################################################################

def typeenu_enu(omega, transition_matrics, reward, target):
    w = omega.copy().reshape((-1, 1, 3))
    P = transition_matrics.copy()
    B = reward.copy().reshape((-1, 3, 1))
    L = np.zeros(np.size(target))

    dist_n = w.copy()
    B_n = (dist_n@B).reshape(-1)
    nore = (B_n - target <= 1e-15)
    j = 0
    while nore.any():
        j = j + 1
        dist_n[nore] = dist_n[nore]@P[nore]
        B_n[nore] = (dist_n[nore]@B[nore]).reshape(-1)
        reach = (B_n - target > 1e-15)
        L[nore*reach] = j
        nore[nore*reach] = False
    
    return L

def typeenu_enu_k(omega, transition_matrics, reward, target):
    w = omega.copy().reshape((-1, 1, 3))
    P = transition_matrics.copy()
    B = reward.copy().reshape((-1, 3, 1))
    L = np.zeros(np.size(target))

    dist_n = w.copy()
    B_n = (dist_n@B).reshape(-1)
    nore = (B_n - target <= 1e-15)
    j = 0
    while nore.any():
        j = j + 1
        dist_n[nore] = dist_n[nore]@P[nore]
        B_n[nore] = (dist_n[nore]@B[nore]).reshape(-1)
        reach = (B_n - target > 1e-15)
        L[nore*reach] = j
        nore[nore*reach] = False
        if j == 40:
            break
    L[nore] = np.inf
    return L
def first_crossing_time_enu(omega, transition_matrics, reward, target, params):
    w = omega.copy().reshape((-1, 1, 3))
    P = transition_matrics.copy()
    B = reward.copy().reshape((-1, 3, 1))
    L = np.zeros(np.size(target))

    set_decom, A, J, A_inv = params
    if True^set_decom:
        J, A = nplg.eig(transition_matrics)
        A_inv = nplg.inv(A)
    
    pos1 = (np.abs(J[:, 0] - 1) < 1e-15)
    pos2 = (np.abs(J[:, 1] - 1) < 1e-15)
    pos3 = (np.abs(J[:, 2] - 1) < 1e-15)
    
    stat_reward = np.zeros(np.size(target))
    stat_reward[pos1] = (w[pos1]@A[pos1, :, 0].reshape((-1, 3, 1))@A_inv[pos1, 0].reshape((-1, 1, 3))@B[pos1]).reshape(-1).real
    stat_reward[pos2] = (w[pos2]@A[pos2, :, 1].reshape((-1, 3, 1))@A_inv[pos2, 1].reshape((-1, 1, 3))@B[pos2]).reshape(-1).real
    stat_reward[pos3] = (w[pos3]@A[pos3, :, 2].reshape((-1, 3, 1))@A_inv[pos3, 2].reshape((-1, 1, 3))@B[pos3]).reshape(-1).real

    case1 = (stat_reward - target > 1e-15)
    case2 = (stat_reward - target <= 1e-15)

    L[case1] = typeenu_enu(w[case1], P[case1], B[case1], target[case1])
    L[case2] = typeenu_enu_k(w[case2], P[case2], B[case2], target[case2])

    return L
############################################################################################################################################
def check_L(L, w, P, B, c, target):
    """
    Function:
    - Check the first crossing time result return by the first_crossing_time functions above.
    - Could only check those results with size one.
    - If the result is not infinite, the function will check whether the k-step reward function f(k)
        would reach the target for the first time at the step the above functions provide.
    - if the result is infinite, the function will check that the k-step reward function f(k) would not
        reach the target at the first 20 step.

    Input:
    - L: an array with size one indicating the first crossing time result returned by the above functions
    - w: an array for the belief state(the probability distribution) required for the k-step reward fucntion
    - P: a matrix showing the stochastic matrix required for the k-step reward function
    - B: an array for the reward vector.
    - c: stationary reward
    - target: a scalar indicating the target k-step reward
    """
    l = L.copy()
    if l == np.inf:
        l = 50
    l = int(l)
    y_P = w.copy()
    
    if (y_P@B.T - target < 1e-15) and (l > 0):
        for k in range(l - 1):
            y_P = y_P@P
            y = y_P@B.T
            if (y - target > 1e-15):
                return False
        y_P = y_P@P
        y = y_P@B.T
        if (y - target > 1e-15) and (l != 50):
            return True
        elif (y - target <= 1e-15) and (l != 50):
            return False
        
        elif (y - target > 1e-15) and (l == 50):
            return False
        else:
            if (c - target < -1e-15):
                return True
            else:
                return False
    elif (y_P@B.T - target > 1e-15) and np.abs(l) < 1e-20:
        return True
    else:
        return False
