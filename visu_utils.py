import numpy as np
import trimesh
import pyglet
from pyglet.gl import *
import math


def rotate_sequence_zup_to_yup(vertices_seq):
    # -90 degrees around X-axis to map Z-up → Y-up
    Rx = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
    rotated_seq = np.einsum("ij,tvj->tvi", Rx[:3, :3], vertices_seq)
    return rotated_seq


def visualize_motion_trimesh(vertices_seq, faces, title="SMPL Motion", fps=30):
    print(f"Visualizing {title}")
    # the original amass samples have z up orientation
    vertices_seq = rotate_sequence_zup_to_yup(vertices_seq)

    vertices_seq = np.asarray(vertices_seq)
    num_frames = len(vertices_seq)
    current_frame = 0
    width, height = 800, 600

    mesh = trimesh.Trimesh(vertices=vertices_seq[0], faces=faces, process=False)

    config = pyglet.gl.Config(double_buffer=True, depth_size=24, alpha_size=8)
    window = pyglet.window.Window(width=width, height=height, config=config, caption=title)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glClearColor(0.9, 0.9, 0.9, 1.0)

    light_pos = (GLfloat * 4)(2.0, 2.0, 2.0, 1.0)
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos)

    bbox = mesh.bounding_box.bounds
    center = np.mean(bbox, axis=0)
    size = np.linalg.norm(bbox[1] - bbox[0])
    cam_distance = size * 2.5

    yaw, pitch = [0.0], [0.0]
    dragging = [False]

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

    def draw_axes(length=0.5):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)  # X - red
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)
        glColor3f(0, 1, 0)  # Y - green
        glVertex3f(0, 0, 0)
        glVertex3f(0, length, 0)
        glColor3f(0, 0, 1)  # Z - blue
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        glEnd()

    def draw_ground(size=1.0):
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.5, 0.5, 0.5, 0.2)  # last value < 1 enables transparency
        glBegin(GL_QUADS)
        glVertex3f(-size, 0, -size)
        glVertex3f(size, 0, -size)
        glVertex3f(size, 0, size)
        glVertex3f(-size, 0, size)
        glEnd()
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if dragging[0]:
            yaw[0] += dx * 0.3
            pitch[0] += dy * 0.3
            pitch[0] = max(-89, min(89, pitch[0]))

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        dragging[0] = True

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        dragging[0] = False

    @window.event
    def on_draw():
        window.clear()
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / float(height), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        yaw_rad = math.radians(yaw[0])
        pitch_rad = math.radians(pitch[0])
        x = cam_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = cam_distance * math.sin(pitch_rad)
        z = cam_distance * math.cos(pitch_rad) * math.cos(yaw_rad)

        gluLookAt(center[0] + x, center[1] + y, center[2] + z, center[0], center[1], center[2], 0, 1, 0)

        draw_ground(size=1.5 * size)
        draw_axes(length=0.5 * size)
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
    window.set_caption("Your New Title")

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
