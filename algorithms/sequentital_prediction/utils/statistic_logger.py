
from utils.profile import profile
from utils.algorithm import algorithm

class statistic_logger:

    def __init__(self, stats_name, algorithm_names):
        self.stats_name = stats_name
        self.algorithm_name = []
        self.algorithms = []

        use_steps = False
        self.use_steps = use_steps

        for algo_name in algorithm_names:
            self.algorithms.append(algorithm(algo_name, use_steps))


    def __str__(self):
        result = ""

        if self.use_steps == False:
            # result += "\t\t"
            # for algo in self.algorithms:
            #     result += "" + algo.get_name() + "\t"

            # result += "\n"

            # for stat in self.stats_name:
            #     if len(stat) < 9:
            #         result += stat + "\t"
            #     else:
            #         result += stat[:9]

            #     for algo in self.algorithms:
            #         value = algo.get(stat) * 100
            #         result += "\t" + str(value)
                
            #     result += "\n"


            for algo in self.algorithms:
                result += f"---------  {algo.get_name()}  --------- \n"

                for stat in self.stats_name:
                    value = round(algo.get(stat) * 100, 2)

                    result += f"{stat} \t {value}% \n"


                result += "\n"

        return result
                


    def increase(self, stat_name, algo_name):
        algo = self.get_algorithm(algo_name)
        if algo != None:
            value = algo.get(stat_name)
            value = value + 1
            algo.set(stat_name, value)
            

    def get_algorithm(self, name):
        for algo in self.algorithms:
            if algo.get_name() == name:
                return algo
        
        return None

    def get(self, stat, algo_name):
        return self.get_algorithm(algo_name).get(stat)

    def set(self, stat, algo_name, value):
        algo = self.get_algorithm(algo_name)
        if algo != None:
            algo.set(stat, value)

    def cal_percent(self, stat, algo_name, sample_size):
        algo = self.get_algorithm(algo_name)
        value = algo.get(stat)
        result = value / sample_size
        algo.set(stat, result)

    

        
        