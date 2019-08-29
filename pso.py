import numpy as np
import random


# z = x^2 + y^2
def criterion(x, y):
    z = x * x + y * y
    return z


def update_position(x, y, vx, vy):
    new_x = x + vx
    new_y = y + vy
    return new_x, new_y


def update_velocity(x, y, vx, vy, per, glo, w=0.5, ro_max=0.14):
    ro1 = random.uniform(0, ro_max)
    ro2 = random.uniform(0, ro_max)

    new_vx = w * vx + ro1 * (per["x"] - x) + ro2 * (glo["x"] - x)
    new_vy = w * vy + ro2 * (per["y"] - y) + ro2 * (glo["y"] - y)
    return new_vx, new_vy


def main():
    particles = 100
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5

    ps = [{"x": random.uniform(x_min, x_max), "y": random.uniform(y_min, y_max)} for i in range(particles)]
    vs = [{"x": 0.0, "y": 0.0} for i in range(particles)]

    personal_best_positions = list(ps)
    personal_best_scores = [criterion(p["x"], p["y"]) for p in ps]
    best_particle = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[best_particle]

    LOOP = 30

    for t in range(LOOP):

        for i in range(particles):
            x, y = ps[i]['x'], ps[i]['y']
            vx, vy = vs[i]['x'], vs[i]['y']

            p = personal_best_positions[i]

            new_x, new_y = update_position(x, y, vx, vy)
            ps[i] = {"x": new_x, "y": new_y}

            new_vx, new_vy = update_velocity(new_x, new_y, vx, vy, p, global_best_position)
            vs[i] = {"x": new_vx, "y": new_vy}

            score = criterion(new_x, new_y)

            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = ps[i]

        best_particle = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[best_particle]

        print(t, global_best_position)


if __name__ == '__main__':
    main()
