import math

almost_pi: float = 22/7
almost_e: float = 19/7
error_pi: float = math.pi - almost_pi
error_e: float = math.e - almost_e

print(f"pi is {math.pi}")
print(f"almost_pi is {almost_pi}")
print(f"error_pi is {error_pi}")
print(f"e is {math.e}")
print(f"almost_e is {almost_e}")
print(f"error_e is {error_e}")