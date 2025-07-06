import matplotlib.pyplot as plt
import numpy as np

print("Matplotlib 기본 그래프 그리기")

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("사인 함수 그래프")
plt.xlabel("x 값")
plt.ylabel("y = sin(x)")
plt.grid()
plt.show()
