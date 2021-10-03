

parameters = {"consequent_size": 1, "window_size": 5}

class profile:

    

    @staticmethod
    def get_float_param(name):
        value = parameters.get(name, None)
        
        if value != None:
            value =  float(value)

        return value

    
    @staticmethod
    def get_int_param(name):
        value = parameters.get(name, None)
        
        if value != None:
            value =  int(value)

        return value

    
    @staticmethod
    def get_bool_param(name):
        value = parameters.get(name, None)
        
        if value != None:
            value =  bool(value)

        return value


if __name__ == "__main__":
    print(profile.get_int_param("window_size"))
