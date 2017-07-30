import numpy as np
import numpy.linalg as npl
import math
from scipy.spatial.distance import pdist, squareform
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools


class Universe:
    def __init__(self, size=np.array([100, 100]), count=5, over=False, dt=0.1, tol=False, con=1, seed=False):
        self.count = count
        self.size = size
        self.m = np.zeros(self.count)
        self.r = np.zeros(self.count)
        self.labels = []
        self.type = []

        self.tol = tol
        self.constant = con

        self.dt = dt
        self.elapsed_time = 0
        self.step_count = 0
        self.init_state = np.zeros([self.count, 6])
        self.state = self.init_state.copy()

        if not over:
            self.obj_generator(seed)
        else:
            self.state_overwrite()

        self.plot_uni("random_state")
        np.savetxt("random.out", self.m)
        self.pos_check()
        self.force()
        self.init_state = self.state.copy()
        np.savetxt("init.out", self.m)
        self.state_sum("initial_state")
        self.plot_uni("initial_state")

    def state_overwrite(self):
        file = open("input.txt", "r")
        c = 0
        for i in file:
            mrlt = i.split(",")
            self.m[c] = float(mrlt[0])
            self.r[c] = float(mrlt[1])
            self.labels.append(mrlt[2])
            self.type.append(mrlt[3])
            self.state[c, :4] = [float(f) for f in mrlt[4:]]
            c += 1
        file.close()

    def obj_generator(self, seed=False):
        if seed:
            rnd.seed(seed)

        counts = np.zeros(4)
        for i in range(self.count):
            dice = rnd.randrange(0, 1000001, 1) / 10000
            if dice <= 90:  # Normal Planet
                pm = rnd.randrange(2, 5002, 5) / 100
                pr = .01  # rnd.randrange(1,101,5)/10
                obj_type = 0
                counts[0] += 1
                label = "P" + str(int(counts[0]))

            elif dice <= 99.99:  # Star
                pm = rnd.randrange(10, 30, 1)
                pr = 0.5  # rnd.randrange(0,100,5) + 50
                obj_type = 1
                counts[1] += 1
                label = "S" + str(int(counts[1]))

            elif dice <= 99.9999:  # Neutron Star
                pm = rnd.randrange(30, 55, 5)
                pr = 2  # rnd.randrange(0,1500,20) + 500
                obj_type = 2
                counts[2] += 1
                label = "NS" + str(int(counts[2]))

            else:  # Black Hole
                pm = rnd.randrange(55, 100)
                pr = 3  # rnd.randrange(0,100,5) + 50
                obj_type = 3
                counts[3] += 1
                label = "BH" + str(int(counts[3]))

            x = (rnd.randint(1000, 9001) / 10000) * self.size[0]
            y = (rnd.randint(1000, 9001) / 10000) * self.size[1]

            v_x = 0 # (rnd.randint(0, 10001) / 1000 - 5) * self.size[0] / 1000
            v_y = 0 # (rnd.randint(0, 10001) / 1000 - 5) * self.size[1] / 1000

            self.labels.append(label)
            self.type.append(obj_type)
            self.m[i] = pm
            self.r[i] = pr
            self.state[i, :] = [x, y, v_x, v_y, 0, 0]

    def pos_check(self):

        if not self.tol:
            self.tol = self.size[0] / 40
        dups = 5
        while dups > 0:
            d = squareform(pdist(self.state[:, :2]))
            ind1, ind2 = np.where(d < self.tol)
            unique = (ind1 < ind2)
            ind1 = ind1[unique]
            ind2 = ind2[unique]
            dups = len(ind1)
            to_be_deleted = []
            to_be_deleted_labels = []
            for i, j in zip(ind1, ind2):
                m0 = self.m[i]
                r0 = self.r[i]
                u0 = self.state[i, :2]
                v0 = self.state[i, 2:4]
                l0 = self.labels[i]

                m1 = self.m[j]
                r1 = self.r[j]
                v1 = self.state[j, 2:4]
                l1 = self.labels[j]

                mnew = m0 + m1 + 0.01
                rnew = (r0 ** 2 + r1 ** 2) ** 0.5
                unew = u0
                vnew = (m0 * v0 + m1 * v1) / mnew
                lnew = l0 + l1



                self.state[i, :2] = unew
                self.state[i, 2:4] = vnew
                self.state[i, 4:] = 0
                self.m[i] = mnew
                self.r[i] = rnew
                self.labels[i] = lnew

                self.state[j, :2] = 0
                self.state[j, 2:4] = 0
                self.m[j] = 0
                self.r[j] = 0
                self.labels[j] = ""

                to_be_deleted.append(j)
                to_be_deleted_labels.append(self.labels[j])

            to_be_deleted = list(set(to_be_deleted))

            self.state = np.delete(self.state, to_be_deleted, 0)

            self.m = np.delete(self.m, to_be_deleted, 0)

            self.r = np.delete(self.r, to_be_deleted, 0)
            self.labels = [l for l in self.labels if l not in to_be_deleted_labels]
            self.count -= len(to_be_deleted)

        for n, i in enumerate(self.state):
            x = i[0]
            y = i[1]

            if x >= self.size[0]:
                self.state[n, 2] = -i[2]
                self.state[n, 0] = (2 * self.size[0] - x)

            if y >= self.size[1]:
                self.state[n, 3] = -i[3]
                self.state[n, 1] = (2 * self.size[1] - y)

            if x < 0:
                self.state[n, 2] = -i[2]
                self.state[n, 0] = - x

            if y < 0:
                self.state[n, 3] = -i[3]
                self.state[n, 1] = - y

    def force(self):
        self.state[:, 4:6] = [0, 0]
        for w in itertools.combinations(range(self.count), 2):
            i, j = w
            m0 = self.m[i]
            u0 = self.state[i, :2]
            p0 = self.state[i, 4:6]

            m1 = self.m[j]
            u1 = self.state[j, :2]
            p1 = self.state[j, 4:6]

            du = u1 - u0
            d = npl.norm(du)
            p = self.constant * m0 * m1 / d / d
            angle = math.atan2(du[1], du[0])
            while angle < 0.0:
                angle += math.pi * 2

            self.state[i, 4:6] = p0 + np.array([math.cos(angle), math.sin(angle)]) * p
            self.state[j, 4:6] = p1 - np.array([math.cos(angle), math.sin(angle)]) * p

    def update(self):
        if self.step_count % 1000 == 0:
            print("Elapsed time = %d" % self.elapsed_time)
        for n, i in enumerate(self.state):
            m_o = self.m[n]
            ux = i[0]
            uy = i[1]
            vx = i[2]
            vy = i[3]
            px = i[4]
            py = i[5]
            a_x = px / m_o
            a_y = py / m_o

            vx_n = vx + self.dt * a_x
            vy_n = vy + self.dt * a_y

            ux_n = ux + self.dt * vx + self.dt * self.dt * a_x / 2
            uy_n = uy + self.dt * vy + self.dt * self.dt * a_y / 2

            self.state[n, :4] = [ux_n, uy_n, vx_n, vy_n]

            if self.step_count % 1000 == 0:
                print(n,m_o, vx, vy)

    def step(self):
        self.elapsed_time += self.dt
        self.step_count += 1

        Galaxy.update()
        # self.state[0] = [0,0,0,0,0,0]
        # self.m[0] = 1000
        Galaxy.force()
        Galaxy.pos_check()
        Galaxy.force()

    def plot_uni(self, file_name="fig"):
        fig = plt.figure(figsize=(15, 15), )
        ax = plt.axes()
        ax.set_facecolor("black")
        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])

        for la, i, j in zip(self.labels, self.state, self.r):
            x = i[0]
            y = i[1]
            xn = x + i[2]
            yn = y + i[3]
            xna = x + i[4] / 10
            yna = y + i[5] / 10
            size_ = j * 300
            ax.scatter(x, y, s=size_, color="white")
            ax.annotate(la, (x, y), color="white")
            ax.plot([x, xn], [y, yn])
            ax.plot([x, xna], [y, yna])

        plt.savefig(file_name)
        plt.close()


    def state_sum(self, tag):
        pass

Galaxy = Universe(count=40, size=np.array([15000, 15000]), con=1000, dt=0.1, tol=50, over=False)


fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, fc="black",
                     xlim=(-0.1*Galaxy.size[0], 1.1 * Galaxy.size[0]),
                     ylim=(-0.1*Galaxy.size[1], 1.1 * Galaxy.size[1]))

rect = plt.Rectangle([0, 0],
                     Galaxy.size[0],
                     Galaxy.size[1],
                     ec='white', lw=4, fc='none')

planets, = ax.plot([], [], 'bo', ms=2)

ax.add_patch(rect)

def animate(i):
    Galaxy.step()
    planets.set_data(Galaxy.state[:, 0], Galaxy.state[:, 1])
    return planets,


def init():
    planets.set_data([], [])
    return planets,


ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=2, blit=True)
plt.show()
