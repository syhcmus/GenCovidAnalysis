from utils.profile import profile

class paramable:

    def __init__(self) -> None:
        self.parameters = {}

    # def set_parameters(self, params):
    #     if params != None and len(params) > 0 and ":" in params:
    #         tokens = params.split("\\s")
            
    #         for param in tokens:
    #             key = param.split(":")
    #             self.parameters[key[0]] = key[1]


    def set_parameter(self, key, value):
        self.parameters[key] = value

    
    def get_float_param(self, name):
        value = self.parameters.get(name, None)

        if value != None:
            return float(value)

        return None

    def get_int_param(self, name):
        value = self.parameters.get(name, None)

        if value != None:
            return int(value)

        return None

    def get_bool_param(self, name):
        value = self.parameters.get(name, None)

        if value != None:
            return bool(value)

        return None

    def get_int_or_default_param(self, name, default_value):
        param = self.get_int_param(name)

        if param != None:
            return param

        return default_value   