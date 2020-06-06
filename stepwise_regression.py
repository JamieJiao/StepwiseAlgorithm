import statsmodels.api as sm 

class StepwiseRegression:
    def __init__(self, y, X):
        self.y = y
        self.X = X
    
    def linear_regression(self, y, X):
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        return results
    
    def find_var_with_smallest_p(self, p_values):
            mini = 0
            initial = True
            var = None
            for key, value in p_values.items():
                if key == 'const':
                    continue
                else:
                    if initial:
                        mini = value
                        var = key
                    initial = False
                    if value < mini:
                        mini = value
                        var = key
            return var

    def enter_new_var(self, y, X, entered_vars, dropped_vars):
        var_candidates = {}
        for var in X.columns:
            if var not in entered_vars or var not in dropped_vars:
                    ols = self.linear_regression(y, X[entered_vars + [var]])
                    p_values = ols.pvalues
                    var_candidates[var] = p_values[var]

        var_enter = self.find_var_with_smallest_p(var_candidates)
        try:
            if var_candidates[var_enter] < 0.1:
                entered_vars.append(var_enter)
                stop = False
                return entered_vars, stop
            else:
                stop = True
                return entered_vars, stop
        except:
            if var_enter == None:
                stop = True
                return entered_vars, stop

    def vars_drop(self, y, X, entered_vars, p_criteria):
        ols = self.linear_regression(y, X[entered_vars])
        # ** pandas series
        p_values = ols.pvalues
        dropped_vars = []
        for index in p_values.index:
            if p_values[index] > p_criteria and index != 'const':
                entered_vars.remove(index)
                dropped_vars.append(index)
        return entered_vars, dropped_vars

    def stepwise_regression(self, y, X, p_criteria=0.15):
        y = self.y
        X = self.X
        ols = self.linear_regression(y, X)
        p_values = {}
        # ** convert to dictionary
        for index in ols.pvalues.index:
            p_values[index] = ols.pvalues[index]

        first_var = self.find_var_with_smallest_p(p_values)
        entered_vars = []
        dropped_vars = []
        entered_vars.append(first_var)
        while True:
            entered_vars, stop = self.enter_new_var(y, X, entered_vars, dropped_vars)
            if stop == True:
                ols = self.linear_regression(y, X[entered_vars])
                return ols
                # ** break
            new_entered_vars, dropped_vars = self.vars_drop(y, X, entered_vars, p_criteria)