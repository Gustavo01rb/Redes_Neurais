import math

class Activation_function:

  @staticmethod
  def step_function_1(*args) -> int:         
    return 1 if args[0] >= 0 else 0 
  @staticmethod
  def step_function_2(*args) -> int:         
    return 1 if args[0] >= 0 else -1
  @staticmethod
  def step_function_3(*args) -> int:         
    return 1 if args[0] > 0 else 0 if args[0] == 0 else -1 
  @staticmethod
  def linear_function(*args) -> float:      
    return 1 if args[0] > 1 else 0 if args[0] < 0 else args[0]
  @staticmethod
  def linear_function_no_saturation(*args) -> float:
    return args[1] * args[0]
  @staticmethod
  def sigmoidal_functions_unipolar(*args) -> float:
    return 1 /( 1 + math.exp( (-args[1])*args[0] ) )
  @staticmethod
  def sigmoidal_functions_bipolar(*args) -> float:
    return (2 /( 1 + math.exp( (-args[1])*args[0] ) )) - 1 