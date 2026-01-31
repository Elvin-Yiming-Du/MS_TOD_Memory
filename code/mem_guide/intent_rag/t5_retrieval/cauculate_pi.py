#给定一个随机数生成函数，估算圆周率pi
import random
def calculate_pi():
    a = random.random()
    b = random.random()
    inside = 0

    for i in range(1000000):
        a = random.random()
        b = random.random()
        if a*a + b*b <=1:
            inside += 1
    
    s = inside/1000000
    return 4*s


main = calculate_pi()
print("Estimated value of pi:", main)