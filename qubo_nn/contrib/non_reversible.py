import itertools
from qubo_nn.problems import Max2SAT
from qubo_nn.problems import Max3SAT
from qubo_nn.problems import SetPartitioning


# M2SAT
n = []
m = []
for comb in itertools.product([True, False], repeat=6):
    x = Max2SAT([((0, comb[0]), (1, comb[1])), ((0, comb[2]), (1, comb[3])), ((0, comb[4]), (2, comb[5]))], 3)
    a = x.gen_qubo_matrix()
    a.flags.writeable = False
    h = hash(a.tostring())
    n.append(comb)
    m.append(a.tolist())

for m_ in m:
    if m.count(m_) > 1:
        print(m_)
        for i, nnn in enumerate(m):
            if nnn == m_:
                print(n[i])


# # M3SAT
# n = []
# m = []
# o = []
# for comb in itertools.product([True, False], repeat=6):
#     x = Max3SAT([((0, comb[0]), (1, comb[1]), (0, comb[2])), ((1, comb[3]), (0, comb[4]), (2, comb[5]))], 3)
#     a = x.gen_qubo_matrix()
#     a.flags.writeable = False
#     h = hash(a.tostring())
#     n.append(comb)
#     m.append(a.tolist())
#     o.append(h)
# 
# for m_ in m:
#     if m.count(m_) > 1:
#         print(m_)
#         for i, nnn in enumerate(m):
#             if nnn == m_:
#                 print(n[i])
# print(len(o), len(set(o)))
# 
# 
# # NP
# sp = SetPartitioning([1, 2], [[1], [2]], [5, 9], 20)
# a = sp.gen_qubo_matrix()
# print(a)
# sp = SetPartitioning([1, 2], [[1], [2]], [5, 9], 15)
# a = sp.gen_qubo_matrix()
# print(a)
# 
# 
# # QK
# qk = QuadraticKnapsack(np.array([[13, 4], [4, 6]]), np.array([23, 10, 2]), 5)
# a = qk.gen_qubo_matrix()
# print(a)
