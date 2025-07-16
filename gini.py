from copy import deepcopy

import pulp
import numpy as np


class GreedyGini:
    def __init__(self,
                 student_2_score,
                 number_of_groups,
                 min_size,
                 max_size,
                 ):

        self.student_2_score = student_2_score
        self.number_of_groups = number_of_groups
        self.min_size = min_size
        self.max_size = max_size

        self.group_2_students = {g: [] for g in range(self.number_of_groups)}
        self.group_2_gini_index = {g: 0 for g in range(self.number_of_groups)}
        self.group_2_number_of_students = {g: 0 for g in
                                           range(self.number_of_groups)}
        self.number_of_students = len(student_2_score)

    def get_remaining_seats(self, g):
        return np.sum(np.clip(np.array(
            [self.min_size - self.group_2_number_of_students[g_] for g_ in
             self.group_2_gini_index if g_ != g]),
            min=0))

    def get_group_candidates(self, i):
        return [g for g in self.group_2_gini_index
                if self.get_remaining_seats(g) <= self.number_of_students - i]

    def get_gini_index(self, group):
        index = sum(abs(self.student_2_score[s] - self.student_2_score[c]) for
                    s in group for c in group) / (2 * len(group)
                                                  * sum(self.student_2_score[s]
                                                        for s in group))
        return index

    def run(self):
        students = set(self.student_2_score.keys())
        i = 1
        while students:
            group_2_of = {}
            group_2_gini_index = {}
            for student in students:
                for group in self.get_group_candidates(i):
                    group_2_student_pre = deepcopy(self.group_2_students)
                    group_2_student_pre[group].append(student)
                    group_2_gini_index[group, student] = self.get_gini_index(
                        group_2_student_pre[group])
                    min_mean_score = min([score for g, score in
                                          self.group_2_gini_index.items()
                                          if g != group])
                    group_2_of[group, student] = min(min_mean_score,
                                                     group_2_gini_index[group,
                                                                        student
                                                                        ])

            group, student = max(group_2_of,
                                 key=lambda seq:
                                 group_2_of[seq[0], seq[1]])

            students = students - {student}

            self.group_2_students[group].append(student)
            self.group_2_gini_index[group] = group_2_gini_index[group, student]
            self.group_2_number_of_students[group] += 1
            i += 1

        return self.group_2_students


class GiniMinimization:
    def __init__(self,
                 student_2_score,
                 number_of_groups,
                 min_size,
                 max_size,
                 ):
        self.student_2_score = student_2_score
        self.number_of_groups = number_of_groups
        self.min_size = min_size
        self.max_size = max_size

        self.big_M = max(self.student_2_score.values()) \
                     - min(self.student_2_score.values())

        self.group_2_students = {g: [] for g in range(self.number_of_groups)}

        self.model = pulp.LpProblem("Gini", pulp.LpMinimize)

    def create_vars(self):
        x_indices = [(k, v) for k in range(self.number_of_groups)
                     for v in self.student_2_score]
        z_indices = [(k, size) for k in range(self.number_of_groups)
                     for size in range(self.min_size, self.max_size + 1)]
        o_indices = [(k, s, c) for k in range(self.number_of_groups)
                     for s in self.student_2_score
                     for c in self.student_2_score
                     if s != c]
        self.x_vars = pulp.LpVariable.dicts(name="x",
                                            indices=x_indices,
                                            cat=pulp.LpBinary)
        self.z_vars = pulp.LpVariable.dicts(name="z",
                                            indices=z_indices,
                                            cat=pulp.LpBinary)
        self.o_vars = pulp.LpVariable.dicts(name="o",
                                            indices=o_indices,
                                            cat=pulp.LpBinary)
        self.d_vars = pulp.LpVariable.dicts(name="d",
                                            indices=o_indices,
                                            cat=pulp.LpContinuous,
                                            lowBound=0)
        self.y_vars = pulp.LpVariable.dicts(name="y",
                                            indices=x_indices,
                                            cat=pulp.LpContinuous,
                                            lowBound=0)
        self.gamma = pulp.LpVariable(name="gamma",
                                     cat=pulp.LpContinuous,
                                     lowBound=0)

    def create_of(self):
        self.model += self.gamma

    def create_assignment_con(self):
        for s in self.student_2_score:
            self.model += pulp.lpSum(self.x_vars[k, s] for k in
                                     range(self.number_of_groups)) == 1

    def create_size_con(self):
        for k in range(self.number_of_groups):
            group = pulp.lpSum(self.x_vars[k, s] for s in
                               self.student_2_score)
            self.model += group == pulp.lpSum(size * self.z_vars[k, size] for
                                              size in range(self.min_size,
                                                            self.max_size + 1))

    def create_one_size_con(self):
        for k in range(self.number_of_groups):
            self.model += pulp.lpSum(self.z_vars[k, size] for
                                     size in range(self.min_size,
                                                   self.max_size + 1)) == 1

    def create_pair_in_one_group_con(self):
        for k, s, c in self.o_vars:
            self.model += self.o_vars[k, s, c] >= self.x_vars[k, s] \
                          + self.x_vars[k, c] - 1

    def create_abs_diff_con(self):
        for k, s, c in self.d_vars:
            self.model += self.d_vars[k, s, c] >= self.student_2_score[s] \
                          - self.student_2_score[c] - self.big_M \
                          * (1 - self.o_vars[k, s, c])

            self.model += self.d_vars[k, s, c] >= self.student_2_score[c] \
                          - self.student_2_score[s] - self.big_M \
                          * (1 - self.o_vars[k, s, c])

    def create_gini_coef_con(self):
        for k, size in self.z_vars:
            self.model += 2 * size * pulp.lpSum(r * self.y_vars[k, s] for s, r
                                                in self.student_2_score.items()
                                                ) \
                          >= pulp.lpSum(var for (k_, _, _), var in
                                        self.d_vars.items() if k_ == k) \
                          - 10 * sum(r for r in self.student_2_score.values())\
                          * (1 - self.z_vars[k, size])

    def create_y_vars_con(self):
        for k, s in self.y_vars:
            self.model += self.y_vars[k, s] >= self.gamma - self.big_M \
                          * (1 - self.x_vars[k, s])
            self.model += self.y_vars[k, s] <= self.gamma + self.big_M \
                          * (1 - self.x_vars[k, s])
            self.model += self.y_vars[k, s] >= - self.big_M * self.x_vars[k, s]
            self.model += self.y_vars[k, s] <= self.big_M * self.x_vars[k, s]

    def get_gini_index(self, group):
        index = sum(abs(self.student_2_score[s] - self.student_2_score[c]) for
                    s in group for c in group) / (2 * len(group)
                                                  * sum(self.student_2_score[s]
                                                        for s in group))
        return index

    def run(self, solver=pulp.COIN, timelimit=1800):
        self.create_vars()
        self.create_of()
        self.create_assignment_con()
        self.create_size_con()
        self.create_one_size_con()
        self.create_pair_in_one_group_con()
        self.create_abs_diff_con()
        self.create_gini_coef_con()
        self.create_y_vars_con()

        self.model.solve(solver(msg=True, timeLimit=timelimit))

        for (k, s), var in self.x_vars.items():
            if round(var.varValue) == 1:
                self.group_2_students[k].append(s)

        return self.group_2_students

