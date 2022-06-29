from matplotlib import pyplot as plt
import numpy as np

# Нарисовать и сохранить матрицу ошибок
def save_matrix(matrix, name, path):
    matrix.plot()
    plt.title(name)

    plt.savefig(path)

# Нарисовать и сохранить столбчатую диаграмму
def save_hist(data, name, path):
    data.plot(y=list(data), kind="bar", rot=10)
    plt.title(name)
    plt.legend(loc='upper right')
    plt.ylim(top=1.2)
    plt.yticks(np.arange(0, 1.05, 0.05))

    plt.savefig(path)
