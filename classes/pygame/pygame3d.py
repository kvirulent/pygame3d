# pygame3d.py

# import libraries
import math
from dataclasses import dataclass
import numpy
import pygame
import csv

import pygame.gfxdraw

def gcolor(c) -> pygame.Color:
    match c:
        case "red":
            return pygame.Color(255, 25, 25)
        case "green":
            return pygame.Color(25, 255, 25)
        case "blue":
            return pygame.Color(25, 25, 255)
        case "yellow":
            return pygame.Color(255, 255, 0)
        case "purple":
            return pygame.Color(255, 25, 255)
        case "cyan":
            return pygame.Color(25, 255, 255)

# instance of pygame for 3d
class Pygame3d:
    def __init__(self, running = True, fps = 60): # -> None: sets constants and sets up pygame
        # pygame setup
        pygame.init()
        self.display = pygame.display.Info().current_w, pygame.display.Info().current_h
        self.running = running        
        self.font = pygame.font.SysFont('Comic Sans', 12)
        self.screen = pygame.display.set_mode(self.display, pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.fps = fps
        # camera constants
        self.focal_length = 85
        self.scale = 5
        self.center = self.to_screen_coordinates(0, 0)
        self.camera = Camera(numpy.array([0., 0., 0.]), numpy.array([0., 90., 0.]))
        self.camera_clipping_planes = [
            ClippingPlane((0, 0, 1), self.focal_length),
            ClippingPlane((1 / math.sqrt(2), 0, 1 / math.sqrt(2)), 0),
            ClippingPlane((-1 / math.sqrt(2), 0, 1 / math.sqrt(2)), 0),
            ClippingPlane((0, 1 / math.sqrt(2), 1 / math.sqrt(2)), 0),
            ClippingPlane((0, -1 / math.sqrt(2), 1 / math.sqrt(2)), 0),
        ]
        self.render_order = []

    @staticmethod
    def load_model(filename: str) -> dataclass: # -> Model: reads model csv data and converts it to a Model
        with open(filename) as model_file:
            reader = list(csv.DictReader(model_file))
            triangle = [Triangle(*[int(i) for i in list(row.values())[:3]], row["color"], row["normal"]) for row in reader if row["color"]]
            vertices = numpy.array([[float(i) for i in list(row.values())[:3]] for row in reader if not row["color"]])
        return Model(vertices, triangle, [-2, 0, 0], numpy.array([0,0,0]))

    def check_for_quit(self): # -> None: quits program if QUIT event exists 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False 

    def to_screen_coordinates(self, x: float, y: float) -> tuple[float, float]: # ->    (float,float): converts 2d coordinates to screen coordinates (origin in
        return x + self.display[0] / 2, y + self.display[1] / 2                 #                      the center of the screen)

    def to_rotation_coordinates(self, x: float, y: float) -> tuple[float, float]:# -> (float,float): converts 2d coordinates to rotation coordinates (origin in
        return x - self.display[0] / 2, y - self.display[1] / 2                  #                   the top left of the screen)
    
    def project_mesh(self, model: dataclass):# -> Tuple(float,float,float,string): Projects the 3D scene in the view of the camera's clipping planes
        vertices = self.rotate(model.vertices, rotation=model.rotation)          # onto a 2D plane
        vertices = self.translate(vertices, translation=model.position)
        vertices = self.translate(vertices, translation=self.camera.position)
        vertices = self.rotate(vertices, rotation=self.camera.rotation)
        triangles = [t for t in model.triangles if all([self.clip_triangle(p, t, vertices) for p in self.camera_clipping_planes])]
        projected_x = [(vertex[0] * self.focal_length) / vertex[-1] for vertex in vertices]
        projected_y = [(vertex[1] * self.focal_length) / vertex[-1] for vertex in vertices]
        x = ((self.to_screen_coordinates(projected_x[triangle.a] * self.scale, projected_y[triangle.a] * self.scale),
                 self.to_screen_coordinates(projected_x[triangle.b] * self.scale, projected_y[triangle.b] * self.scale),
                 self.to_screen_coordinates(projected_x[triangle.c] * self.scale, projected_y[triangle.c] * self.scale),
                 triangle.color) for triangle in triangles)
        return x
    
    def project_scene(self, scene): # -> List[Tuple(float,float,float,string)]: Runs project_mesh() on each model in scene
        return [self.project_mesh(mesh) for mesh in scene]

    def render_mesh(self, projected_mesh): # -> None: Draws the projected scene on the screen
        def merge(arr: dict):
            if len(arr) > 1:
                arr = arr
                r = len(arr) // 2
                L = arr[:r]
                M = arr[r:]

                merge(L)
                merge(M)

                i = j = k = 0

                while i < len(L) and j < len(M):
                    if L[i] < M[j]:
                        arr[k] = L[i]
                    else:
                        arr[k] = M[j]
                    k += 1
                
                while i < len(L):
                    arr[k] = L[i]
                    i += 1
                    k += 1
                
                while j < len(M):
                    arr[k] = M[j]
                    j += 1
                    k += 1

                sx = []

                for key in arr:
                    sx.append(x[key])

                return sx
            
        x = {}

        for t in projected_mesh:
            dist = self.get_vert_distance((t[0], t[1], t[2]), self.camera.position)
            x[int(dist[0])] = t
            x = merge(x)

        for t in x:
            pygame.gfxdraw.filled_polygon(self.screen, (x[t][0], x[t][1], x[t][2]), gcolor(x[t][-1]))

        #for triangle in projected_mesh:
        #    pygame.gfxdraw.filled_polygon(self.screen, (triangle[0], triangle[1], triangle[2]), gcolor(triangle[-1]))

    def render_scene(self, projected_scene): # -> None: Runs render_mesh() on each projection in projected_scene
        for projected_mesh in projected_scene:
            self.render_mesh(projected_mesh)

    @staticmethod
    def translate(vertices, translation): # -> Tuple(int): Calculates the translation of vertices
        return vertices + translation

    @staticmethod
    def rotate(vertices, rotation: numpy.ndarray): # -> NDArray[float,float,float]: Calculates the rotation of vertices by rotation matrix
        rotation = rotation * math.pi / 180
        rotation_z_matrix = numpy.array([
            [math.cos(rotation[2]), -math.sin(rotation[2]), 0],
            [math.sin(rotation[2]), math.cos(rotation[2]), 0],
            [0, 0, 1],
        ])
        rotation_y_matrix = numpy.array([
            [math.cos(rotation[1]), 0, math.sin(rotation[1])],
            [0, 1, 0],
            [-math.sin(rotation[1]), 0, math.cos(rotation[1])],
        ])
        rotation_x_matrix = numpy.array([
            [1, 0, 0],
            [0, math.cos(rotation[0]), -math.sin(rotation[0])],
            [0, math.sin(rotation[0]), math.cos(rotation[0])],
        ])
        x_rotated = numpy.tensordot(rotation_x_matrix, vertices, axes=(1, 1)).T
        xy_rotated = numpy.tensordot(rotation_y_matrix, x_rotated, axes=(1, 1)).T
        xyz_rotated = numpy.tensordot(rotation_z_matrix, xy_rotated, axes=(1, 1)).T
        return xyz_rotated

    def clip_triangle(self, plane, triangle, vertices): # -> Boolean: Check if vertices intercept the clipping plane
        distances = numpy.array([
            self.get_signed_distance(plane, vertices[triangle.a]),
            self.get_signed_distance(plane, vertices[triangle.b]),
            self.get_signed_distance(plane, vertices[triangle.c])
        ])

        if all(distances > 0):
            return True
        elif all(distances < 0):
            return False
        else:
            return True

    @staticmethod
    def get_signed_distance(plane, vertex):# -> float: Calculates the distance between plane and vertex
        normal_x, normal_y, normal_z = plane.normal
        vertex_x, vertex_y, vertex_z = vertex
        return vertex_x * normal_x + (vertex_y * normal_y) + (vertex_z * normal_z) + plane.distance_to_origin
    
    @staticmethod
    def get_vert_distance(v1, v2):
        v1_x, v1_y, v1_z = v1
        v2_x, v2_y, v2_z = v2
        return numpy.sqrt((v2_x - v1_x)**2 + (v2_y - v1_y)**2 + (v2_z - v1_z)**2)

    def flip(self): # -> None: Renders the viewport
        pygame.display.flip()
        self.screen.fill('black')
        self.clock.tick(self.fps)

# simple triangle class
@dataclass
class Triangle:
    a: int
    b: int
    c: int
    color: str
    normal: 1 | 0

# simple camera class
# TODO: modularize camera behavior
@dataclass
class Camera:
    position: numpy.ndarray
    rotation: numpy.ndarray

# simple class for complex geometry
@dataclass
class Model:
    vertices: numpy.ndarray
    triangles: list
    position: list
    rotation: numpy.ndarray

# class for a plane that represents what is in the view of the camera:
# TODO: achieve higher render distance
@dataclass
class ClippingPlane:
    normal: tuple
    distance_to_origin: float