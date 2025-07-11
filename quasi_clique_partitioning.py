from copy import deepcopy

import numpy as np
import pulp


class GreedyQCPP:
    def __init__(self,
                 student_2_group,
                 number_of_groups,
                 min_size,
                 max_size,
                 ):
        self.student_2_group = student_2_group
        self.number_of_groups = number_of_groups
        self.min_size = min_size
        self.max_size = max_size

        self.cliques = {g: [] for g in student_2_group.values()}
        for s, g in student_2_group.items():
            self.cliques[g].append(s)

        self.group_2_students = {g: [] for g in range(self.number_of_groups)}
        self.group_2_number_of_students = {g: 0 for g in
                                           range(self.number_of_groups)}

    def get_density(self, quasi_clique):
        components = {}
        for s in quasi_clique:
            if self.student_2_group[s] not in components:
                components[self.student_2_group[s]] = [s]
            else:
                components[self.student_2_group[s]].append(s)

        sizes = [len(component) for component in components.values()]

        return (sum(size * (size - 1) for size in sizes)
                / (sum(sizes) * (sum(sizes) - 1)))

    def get_remaining_seats(self, g):
        return np.sum(np.clip(np.array(
            [self.min_size - self.group_2_number_of_students[g_] for g_ in
             self.group_2_students if g_ != g]),
            min=0))

    def run(self):
        while sum(len(clique) for clique in self.cliques.values()) > 0:
            max_clique = max(self.cliques, key=lambda c: len(self.cliques[c]))
            allowable_sizes = [min(len(self.cliques[max_clique]),
                                   sum(len(c) for c in self.cliques.values())
                                   - self.get_remaining_seats(g),
                                   self.max_size
                                   - len(self.group_2_students[g]))
                               for g in self.group_2_students]

            self.group_2_students_pre = deepcopy(self.group_2_students)

            for g, size in zip(self.group_2_students_pre.keys(),
                               allowable_sizes):
                self.group_2_students_pre[g] += self.cliques[max_clique][:size]

            group = max(self.group_2_students_pre,
                        key=lambda g:
                        0 if self.group_2_students_pre[g]
                            == self.group_2_students[g]
                        else self.get_density(self.group_2_students_pre[g]))

            self.cliques[max_clique] = self.cliques[max_clique][
                                       len(self.group_2_students_pre[group])
                                       - len(self.group_2_students[group]):]
            self.group_2_students[group] = self.group_2_students_pre[group]
            self.group_2_number_of_students[group] \
                = len(self.group_2_students[group])

        return self.group_2_students


class QCPPIP:
    def __init__(self,
                 student_2_group,
                 number_of_groups,
                 min_size,
                 max_size,
                 ):
        self.student_2_group = student_2_group
        self.number_of_groups = number_of_groups
        self.min_size = min_size
        self.max_size = max_size

        self.students = list(self.student_2_group.keys())
        self.group_2_students = {g: [] for g in range(self.number_of_groups)}

        self.cliques = {g: [] for g in student_2_group.values()}
        for s, g in student_2_group.items():
            self.cliques[g].append(s)

        self.big_M = 100_000
        self.model = pulp.LpProblem("QCPP", pulp.LpMaximize)

    def create_vars(self):
        x_indices = [(k, v) for k in range(self.number_of_groups)
                   for v in self.students]
        self.x_vars = pulp.LpVariable.dicts(name="x",
                                            indices=x_indices,
                                            cat=pulp.LpBinary)

        o_indices = [(k, u, v) for k in range(self.number_of_groups)
                   for i, u in enumerate(self.students)
                   for v in self.students[i + 1:]]

        self.o_vars = pulp.LpVariable.dicts(name="o",
                                            indices=o_indices,
                                            cat=pulp.LpBinary)

        z_indices = [(k, size) for k in range(self.number_of_groups)
                   for size in range(self.min_size, self.max_size + 1)]

        self.z_vars = pulp.LpVariable.dicts(name="z",
                                            indices=z_indices,
                                            cat=pulp.LpBinary)

        self.gamma = pulp.LpVariable(name="gamma", lowBound=0, upBound=1,
                                     cat=pulp.LpContinuous)

    def create_of(self):
        self.model += self.gamma

    def create_assignment_con(self):
        for s in self.students:
            self.model += pulp.lpSum(self.x_vars[k, s] for k in
                                     range(self.number_of_groups)) == 1

    def create_size_con(self):
        for k in range(self.number_of_groups):
            self.model += pulp.lpSum(self.x_vars[k, s] for s in
                                     self.students) \
                          == pulp.lpSum(size * self.z_vars[k, size] for size in
                                        range(self.min_size, self.max_size + 1)
                                        )

    def create_one_size_con(self):
        for k in range(self.number_of_groups):
            self.model += pulp.lpSum(self.z_vars[k, size] for size in
                                     range(self.min_size, self.max_size + 1)) \
                          == 1

    def create_edge_con(self):
        for k in range(self.number_of_groups):
            for clique in self.cliques.values():
                for i, u in enumerate(clique):
                    for v in clique[i + 1:]:
                        self.model += self.o_vars[k, u, v] <= self.x_vars[k, u]
                        self.model += self.o_vars[k, u, v] <= self.x_vars[k, v]

    def create_density_con(self):
        for k in range(self.number_of_groups):
            for size in range(self.min_size, self.max_size + 1):
                self.model += pulp.lpSum(self.o_vars[k, u, v] for clique in
                                         self.cliques.values() for i, u in
                                         enumerate(clique)
                                         for v in clique[i + 1:]) \
                              >= self.gamma * size * (size - 1) / 2 \
                              - self.big_M * (1 - self.z_vars[k, size])

    def run(self, solver=pulp.COIN, timelimit=1800):
        self.create_vars()
        self.create_of()
        self.create_assignment_con()
        self.create_size_con()
        self.create_one_size_con()
        self.create_edge_con()
        self.create_density_con()

        self.model.solve(solver(msg=True, timeLimit=timelimit))

        for (k, s), var in self.x_vars.items():
            if round(var.varValue) == 1:
                self.group_2_students[k].append(s)

        return self.group_2_students










