import math

almost_pi: float = 22/7
error_pi: float = math.pi - almost_pi

almost_e: float = 19/7
error_e: float = math.e - almost_e

e_to_i_pi: float = math.e ** (math.pi * 1j)
almost_e_to_i_pi: float = almost_e ** (almost_pi * 1j)
error_e_to_i_pi: float = e_to_i_pi - almost_e_to_i_pi

print(f"pi is {math.pi}")
print(f"almost_pi is {almost_pi}")
print(f"error_pi is {error_pi}")
print(f"e is {math.e}")
print(f"almost_e is {almost_e}")
print(f"error_e is {error_e}")
print(f"e_to_i_pi is {e_to_i_pi}")
print(f"almost_e_to_i_pi is {almost_e_to_i_pi}")