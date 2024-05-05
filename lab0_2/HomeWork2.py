import pygame
import numpy as np



pygame.init()
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))

GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
BROWN = (165, 42, 42)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)

def draw_points(points, colors):
    for point, color in zip(points, colors):
        pygame.draw.circle(win, color, (point[0], point[1]), 3)


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def range_query(data, point_index, eps):
    neighbors = []
    for i in range(len(data)):
        if euclidean_distance(data[point_index], data[i]) <= eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(data, cluster_result, point_index, neighbors, cluster_label, eps, min_pts):
    cluster_result[point_index] = cluster_label
    i = 0
    while i < len(neighbors):
        neighbor_index = neighbors[i]
        if cluster_result[neighbor_index] == -1:
            cluster_result[neighbor_index] = cluster_label
            new_neighbors = range_query(data, neighbor_index, eps)
            if len(new_neighbors) >= min_pts:
                neighbors += [n for n in new_neighbors if n not in neighbors]
        elif cluster_result[neighbor_index] == 0:
            cluster_result[neighbor_index] = cluster_label
        i += 1

def dbscan(data, eps, min_pts):
    cluster_label = 0
    cluster_result = [-1] * len(data)
    for i in range(len(data)):
        if cluster_result[i] != -1:
            continue
        neighbors = range_query(data, i, eps)
        if len(neighbors) < min_pts:
            cluster_result[i] = 0
        else:
            cluster_label += 1
            expand_cluster(data, cluster_result, i, neighbors, cluster_label, eps, min_pts)
    return cluster_result

points = []
colors = [BLACK]

running = True

cluster_colors = {}

while running:
    win.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                points.append(event.pos)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                eps = 50
                min_samples = 3
                data = np.array(points)
                cluster_result = dbscan(data, eps, min_samples)
                print(cluster_result)
                colors = []

                for i in range(len(data)):
                    if cluster_result[i] == 0:
                        colors.append(BLACK)
                    else:
                        cluster_id = cluster_result[i]
                        if cluster_id not in cluster_colors:
                            cluster_colors[cluster_id] = np.random.randint(25, 256, size=3)
                        colors.append(cluster_colors[cluster_id])

    for point in points:
        pygame.draw.circle(win, (0, 0, 0), point, 5)
    draw_points(points, colors)
    pygame.display.update()


pygame.quit()