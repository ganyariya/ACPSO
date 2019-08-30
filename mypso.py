import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# 目的関数
# def objective_function(position):
#     return np.sum(position * position)

def objective_function(position):
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / len(position) * np.sum(position ** 2, axis=0)))
    t3 = np.e
    t4 = -np.exp(1.0 / len(position) * np.sum(np.cos(2 * np.pi * position), axis=0))
    return t1 + t2 + t3 + t4


# 描画のための初期化
def init_plot(xy_min, xy_max):
    matplot_x = np.arange(xy_min, xy_max, 1.0)
    matplot_y = np.arange(xy_min, xy_max, 1.0)

    matplot_mesh_X, matplot_mesh_Y = np.meshgrid(matplot_x, matplot_y)

    Z = []
    for i in range(len(matplot_mesh_X)):
        z = []
        for j in range(len(matplot_mesh_X[0])):
            result = objective_function(np.array([matplot_mesh_X[i][j], matplot_mesh_Y[i][j]]))
            z.append(result)
        Z.append(z)
    Z = np.array(Z)

    plt.ion()
    plt.title("PSO")
    fig = plt.figure()
    axes = Axes3D(fig)

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('f(x, y)')

    mesh_XYZ = {
        "X": matplot_mesh_X,
        "Y": matplot_mesh_Y,
        "Z": Z
    }

    return axes, mesh_XYZ


# 描画
def play_plot(axes, mesh_XYZ, positions, personal_best_positions, personal_best_scores, global_best_particle_position, velocities):
    axes.plot_wireframe(mesh_XYZ['X'], mesh_XYZ['Y'], mesh_XYZ['Z'], alpha=0.3)
    axes.scatter(positions[:, 0], positions[:, 1], np.apply_along_axis(objective_function, 1, positions), marker='^',
                 c="red", linewidths=5)
    axes.scatter(personal_best_positions[:, 0], personal_best_positions[:, 1], personal_best_scores, linewidths=5,
                 marker='x', c='blue')
    axes.scatter(global_best_particle_position[0], global_best_particle_position[1],
                 objective_function(global_best_particle_position), linewidths=8, marker='o', c='green')

    axes.quiver(positions[:,0], positions[:,1], np.apply_along_axis(objective_function, 1, positions), velocities[:, 0], velocities[:, 1], np.zeros(len(velocities)), color='gray')
    plt.draw()
    plt.pause(1)
    plt.cla()


# 各粒子の位置更新
def update_positions(positions, velocities):
    positions += velocities
    return positions


# 各粒子の速度更新
def update_velocities(positions, velocities, personal_best_positions, global_best_particle_position, w=0.5,
                      ro_max=0.14):
    rc1 = random.uniform(0, ro_max)
    rc2 = random.uniform(0, ro_max)

    velocities = velocities * w + rc1 * (personal_best_positions - positions) + rc2 * (
            global_best_particle_position - positions)
    return velocities


def main():
    print("Particles: ")
    number_of_particles = int(input())

    print("Dimensions: ")
    dimensions = int(input())

    print("LimitTimes: ")
    limit_times = int(input())

    xy_min, xy_max = -32, 32

    # グラフの初期化
    axes, mesh_XYZ = init_plot(xy_min, xy_max)

    # 各粒子の位置
    positions = np.array(
        [[random.uniform(xy_min, xy_max) for _ in range(dimensions)] for _ in range(number_of_particles)])

    # 各粒子の速度
    velocities = np.zeros(positions.shape)

    # 各粒子ごとのパーソナルベスト位置
    personal_best_positions = np.copy(positions)

    # 各粒子ごとのパーソナルベストの値
    personal_best_scores = np.apply_along_axis(objective_function, 1, personal_best_positions)

    # グローバルベストの粒子ID
    global_best_particle_id = np.argmin(personal_best_scores)

    # グローバルベスト位置
    global_best_particle_position = personal_best_positions[global_best_particle_id]

    # 規定回数
    for T in range(limit_times):

        # 速度更新
        velocities = update_velocities(positions, velocities, personal_best_positions,
                                       global_best_particle_position)
        # 位置更新
        positions = update_positions(positions, velocities)

        # パーソナルベストの更新
        for i in range(number_of_particles):
            score = objective_function(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]

        # グローバルベストの更新
        global_best_particle_id = np.argmin(personal_best_scores)
        global_best_particle_position = personal_best_positions[global_best_particle_id]

        # グラフ描画
        play_plot(axes, mesh_XYZ, positions, personal_best_positions, personal_best_scores,
                  global_best_particle_position, velocities)


if __name__ == '__main__':
    main()
