import numpy as np
import pulp

from number_partition_problem import GreedyNPP, NPPIP
from quasi_clique_partitioning import GreedyQCPP, QCPPIP


number_of_students = 50
number_of_groups = 3
min_size = 10
max_size = 36

student_2_score = dict(zip(range(number_of_students),
                           np.random.normal(loc=1500,
                                            scale=1000,
                                            size=number_of_students)))

student_2_group = dict(zip(range(number_of_students),
                           np.random.randint(low=0,
                                             high=10,
                                             size=number_of_students)
                           )
                       )

npp = GreedyNPP(student_2_score,
                number_of_groups,
                min_size,
                max_size)
greedy_npp_sol = npp.run()

print([len(studs) for studs in greedy_npp_sol.values()])
print(min(npp.group_2_total_score.values()))


nppip = NPPIP(student_2_score,
              number_of_groups,
              min_size,
              max_size)
ip_npp_sol = nppip.run(pulp.CPLEX, 30)
print([len(studs) for studs in ip_npp_sol.values()])
print(nppip.model.objective.value())


qcpp = GreedyQCPP(student_2_group,
                  number_of_groups,
                  min_size,
                  max_size)

greedy_qcpp_sol = qcpp.run()
print([len(val) for val in greedy_qcpp_sol.values()])
for studs in greedy_qcpp_sol.values():
    print(qcpp.get_density(studs))

qcppip = QCPPIP(student_2_group,
                number_of_groups,
                min_size,
                max_size)

ip_qcpp_sol = qcppip.run(pulp.CPLEX, 30)
print([len(val) for val in ip_qcpp_sol.values()])

for studs in ip_qcpp_sol.values():
    print(qcpp.get_density(studs))

