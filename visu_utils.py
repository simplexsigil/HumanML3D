import numpy as np
import trimesh
import pyglet
from pyglet.gl import *
import time


import numpy as np
import trimesh
import pyglet
from pyglet.gl import *


def visualize_motion_trimesh(vertices_seq, faces, fps=30):
    vertices_seq = np.asarray(vertices_seq)
    num_frames = len(vertices_seq)
    current_frame = 0
    width, height = 800, 600

    mesh = trimesh.Trimesh(vertices=vertices_seq[0], faces=faces, process=False)

    config = pyglet.gl.Config(double_buffer=True, depth_size=24)
    window = pyglet.window.Window(width=width, height=height, config=config, caption="SMPL Motion")

    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glClearColor(0.95, 0.95, 0.95, 1.0)

    light_pos = (GLfloat * 4)(2.0, 2.0, 2.0, 1.0)
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos)

    # Camera setup
    bbox = mesh.bounding_box.bounds
    center = np.mean(bbox, axis=0)
    size = np.linalg.norm(bbox[1] - bbox[0])
    cam_distance = size * 2.5

    def update(dt):
        nonlocal current_frame
        current_frame = (current_frame + 1) % num_frames
        mesh.vertices = vertices_seq[current_frame]

    def draw_mesh():
        glColor3f(0.2, 0.3, 0.4)
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for idx in face:
                glVertex3f(*mesh.vertices[idx])
        glEnd()

    @window.event
    def on_draw():
        window.clear()
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / float(height), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(center[0], center[1], cam_distance, center[0], center[1], center[2], 0, 1, 0)
        draw_mesh()

    pyglet.clock.schedule_interval(update, 1.0 / fps)
    pyglet.app.run()


import numpy as np
import trimesh
import pyglet
from pyglet.gl import *
import math


# import trimesh

# mesh = trimesh.creation.icosphere(radius=0.5)
# scene = trimesh.Scene(mesh)
# scene.show()


def visualize_static_trimesh():
    width, height = 800, 600
    mesh = trimesh.creation.icosphere(radius=0.5, subdivisions=3)

    config = pyglet.gl.Config(double_buffer=True, depth_size=24)
    window = pyglet.window.Window(width=width, height=height, config=config, caption="Static Trimesh Example")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glClearColor(0.9, 0.9, 0.9, 1.0)

    light_pos = (GLfloat * 4)(2.0, 3.0, 2.0, 1.0)
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos)

    center = mesh.centroid
    size = np.linalg.norm(mesh.bounding_box.extents)
    cam_distance = size * 2.5

    rotation = [0.0]

    def draw_mesh():
        glColor3f(0.8, 0.8, 0.95)
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for idx in face:
                glVertex3f(*mesh.vertices[idx])
        glEnd()

    def update(dt):
        rotation[0] += 30 * dt  # degrees per second

    @window.event
    def on_draw():
        window.clear()
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / float(height), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        eye_x = cam_distance * math.sin(math.radians(rotation[0]))
        eye_z = cam_distance * math.cos(math.radians(rotation[0]))
        gluLookAt(eye_x, 0.5, eye_z, center[0], center[1], center[2], 0, 1, 0)

        draw_mesh()

    pyglet.clock.schedule_interval(update, 1 / 60.0)
    pyglet.app.run()


# visualize_static_trimesh()
