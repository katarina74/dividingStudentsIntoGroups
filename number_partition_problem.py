import pulp
import numpy as np


class GreedyNPP:
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
        self.group_2_total_score = {g: 0 for g in range(self.number_of_groups)}
        self.group_2_number_of_students = {g: 0 for g in
                                           range(self.number_of_groups)}
        self.number_of_students = len(student_2_score)

    def get_remaining_seats(self, g):
        return np.sum(np.clip(np.array(
            [self.min_size - self.group_2_number_of_students[g_] for g_ in
             self.group_2_total_score if g_ != g]),
            min=0))

    def get_group_candidates(self, i):
        return [g for g in self.group_2_total_score
                if self.get_remaining_seats(g) <= self.number_of_students - i]

    def run(self):
        for i, student in enumerate(sorted(self.student_2_score,
                                           key=lambda s: self.student_2_score[s],
                                           reverse=True),
                                    start=1):

            group = min(self.get_group_candidates(i),
                        key=lambda g: self.group_2_total_score[g])
            self.group_2_students[group].append(student)
            self.group_2_total_score[group] += self.student_2_score[student]
            self.group_2_number_of_students[group] += 1

        return self.group_2_students


class NPPIP:
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

        self.model = pulp.LpProblem("NPP", pulp.LpMaximize)

    def create_vars(self):
        x_indices = [(k, v) for k in range(self.number_of_groups)
                   for v in self.student_2_score]
        self.x_vars = pulp.LpVariable.dicts(name="x",
                                            indices=x_indices,
                                            cat=pulp.LpBinary)
        self.gamma = pulp.LpVariable(name="gamma",
                                     cat=pulp.LpContinuous,
                                     lowBound=0,
                                     upBound=sum(self.student_2_score.values())
                                     )

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
            self.model += group >= self.min_size
            self.model += group <= self.max_size

    def create_gamma_con(self):
        for k in range(self.number_of_groups):
            self.model += self.gamma <= pulp.lpSum(r * self.x_vars[k, s] for
                                                   s, r in
                                                   self.student_2_score.items()
                                                   )

    def run(self, solver=pulp.COIN, timelimit=1800):
        self.create_vars()
        self.create_of()
        self.create_assignment_con()
        self.create_size_con()
        self.create_gamma_con()

        self.model.solve(solver(msg=True, timeLimit=timelimit))

        for (k, s), var in self.x_vars.items():
            if round(var.varValue) == 1:
                self.group_2_students[k].append(s)

        return self.group_2_students
