import numpy as np
import pulp

from number_partition_problem import GreedyNPPTotal, NPPIPTotal, NPPIPMean, \
    GreedyNPPMean
from quasi_clique_partitioning import GreedyQCPP, QCPPIP
from gini import GiniMinimization, GreedyGini


number_of_students = 10
number_of_groups = 3
min_size = 2
max_size = 4

student_2_score = dict(zip(range(number_of_students),
                           np.round(np.random.normal(loc=1500,
                                                     scale=1000,
                                                     size=number_of_students)))
                       )
print(student_2_score)

student_2_group = dict(zip(range(number_of_students),
                           np.random.randint(low=0,
                                             high=10,
                                             size=number_of_students)
                           )
                       )

npp_total = GreedyNPPTotal(student_2_score,
                           number_of_groups,
                           min_size,
                           max_size)
greedy_npp_total_sol = npp_total.run()

print([len(studs) for studs in greedy_npp_total_sol.values()])
print(npp_total.group_2_total_score.values())
print(min(npp_total.group_2_total_score.values()))


npp_ip_total = NPPIPTotal(student_2_score,
                          number_of_groups,
                          min_size,
                          max_size)
ip_npp_total_sol = npp_ip_total.run(pulp.CPLEX, 30)
print([len(studs) for studs in ip_npp_total_sol.values()])
print([sum([student_2_score[s] for s in studs])
       for studs in ip_npp_total_sol.values()])
print(npp_ip_total.model.objective.value())


npp_mean = GreedyNPPMean(student_2_score,
                         number_of_groups,
                         min_size,
                         max_size)
greedy_npp_mean_sol = npp_mean.run()

print([len(studs) for studs in greedy_npp_mean_sol.values()])
print(npp_mean.group_2_mean_score.values())
print(min(npp_mean.group_2_mean_score.values()))


npp_ip_mean = NPPIPMean(student_2_score,
                        number_of_groups,
                        min_size,
                        max_size)
ip_npp_mean_sol = npp_ip_mean.run(pulp.CPLEX, 30)
print([len(studs) for studs in ip_npp_mean_sol.values()])
print([np.mean([student_2_score[s] for s in studs])
       for studs in ip_npp_mean_sol.values()])
print(npp_ip_mean.model.objective.value())


greedy_qcpp = GreedyQCPP(student_2_group,
                         number_of_groups,
                         min_size,
                         max_size)

greedy_qcpp_sol = greedy_qcpp.run()
print([len(val) for val in greedy_qcpp_sol.values()])
for studs in greedy_qcpp_sol.values():
    print(greedy_qcpp.get_density(studs))


ip_qcpp = QCPPIP(student_2_group,
                 number_of_groups,
                 min_size,
                 max_size)

ip_qcpp_sol = ip_qcpp.run(pulp.PULP_CBC_CMD, 30)
print([len(val) for val in ip_qcpp_sol.values()])

for studs in ip_qcpp_sol.values():
    print(greedy_qcpp.get_density(studs))


ip_gini = GiniMinimization(student_2_group,
                           number_of_groups,
                           min_size,
                           max_size)

gini_min_sol = ip_gini.run(pulp.CPLEX, 30)

print([len(val) for val in gini_min_sol.values()])

for studs in gini_min_sol.values():
    print(ip_gini.get_gini_index(studs))


greedy_gini = GreedyGini(student_2_group,
                         number_of_groups,
                         min_size,
                         max_size)

greedy_gini_sol = greedy_gini.run()

print([len(val) for val in greedy_gini_sol.values()])

for studs in greedy_gini_sol.values():
    print(greedy_gini.get_gini_index(studs))


