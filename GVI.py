"""
    McMaster University
    Computer Engineering Capstone Project
    Gesture-based Virtual Interaction System (GVIS)
    Arthor: Zhenhuan Sun
"""

from multiprocessing import Process
from gesture_event import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import wx 
import cv2
import glfw
import pyrr
import math
import random
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram
from OpenGL.GL.shaders import compileShader
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController


import argparse
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized




# window width and height for 3D simulation
WIDTH, HEIGHT = 1280, 720

# shader
#=================================================================================================
#=================================================================================================

vertex_shader = """
    # version 330 core

    layout(location = 0) in vec3 a_position;     // vertex position attribute
    layout(location = 1) in vec2 a_texture;     // vertex texture attribute
    layout(location = 2) in vec3 a_normal;       // vertex normal attribute (light tracking)
    //layout(location = 3) in mat4 a_instanceMatrix;      // instance matrix

    uniform mat4 projection;     // projection matrix
    uniform mat4 view;       // view matrix
    uniform mat4 model;      // translation and rotation matrix

    out vec2 v_texture;     // texture vertex pass to fragment shader

    void main()
    {
        //vec3 final_position = a_position + a_offset;
        gl_Position = projection * view * model  * vec4(a_position, 1.0);
        v_texture = a_texture;
    }
"""

fragment_shader = """
    # version 330 core

    in vec2 v_texture;      // texture vertex passed from vertex shader

    uniform sampler2D s_texture;        // texture for vertex

    out vec4 out_color;     // output color of all vectices

    void main()
    {
        // map texture using texture vertex
        out_color = texture(s_texture, v_texture);
    }
"""

#=================================================================================================
#=================================================================================================




# ObjectLoader class
#=================================================================================================
#=================================================================================================

# 3D obejct file loader class that is capable of loading .obj file and texture file exported from blender
class ObjectLoader:
    buffer = []

    @staticmethod
    # find the data in one line
    def search_data(data_values, coordinates, skip, data_type):
        for d in data_values:
            if d == skip:
                continue
            # vertex, vertex texture, vertex normal
            if data_type == 'float':
                coordinates.append(float(d))
            # face value
            elif data_type == 'int':
                # the value in obj file start at 1
                # python use 0 as the first index, so we subtract 1
                coordinates.append(int(d)-1)

    @staticmethod
    # sorted vertex buffer for use with glDrawArrays function
    def create_sorted_vertex_buffer(indices_data, vertices, textures, normals):
        for i, ind in enumerate(indices_data):
            if i % 3 == 0: # sort the vertex coordinates
                start = ind * 3
                end = start + 3
                ObjectLoader.buffer.extend(vertices[start:end])
            elif i % 3 == 1: # sort the texture coordinates
                start = ind * 2
                end = start + 2
                ObjectLoader.buffer.extend(textures[start:end])
            elif i % 3 == 2: # sort the normal coordinates
                start = ind * 3
                end = start + 3
                ObjectLoader.buffer.extend(normals[start:end])

    @staticmethod
    def load_obj(file):
        vertex_coords = [] # buffer that contains vertex coordinates
        texture_coords = [] # buffer that contains texture coordinates (u and v)
        normal_coords = [] # buffer that contains normal coordinates
        all_indices = [] # buffer that contains all the vertex, texture and normal indices (face)
        indices = [] # buffer that contains vertex indices for indexed drawing

        with open(file, 'r') as f:
            line = f.readline()
            while line:
                values = line.split()
                if values[0] == 'v': # vertex coordinates
                    ObjectLoader.search_data(values, vertex_coords, 'v', 'float')
                elif values[0] == 'vt': # vertex texture coordinates
                    ObjectLoader.search_data(values, texture_coords, 'vt', 'float')
                elif values[0] == 'vn': # vertex normal coordinates
                    ObjectLoader.search_data(values, normal_coords, 'vn', 'float')
                elif values[0] == 'f': # face (each face is a triangle which contains all vertex, texture and normal indices)
                    for value in values[1:]:
                        val = value.split('/')
                        ObjectLoader.search_data(val, all_indices, 'f', 'int')
                        indices.append(int(val[0])-1) # python index starts at 0
                line = f.readline() # read another line of obj file

        # used for glDrawArrays function
        ObjectLoader.create_sorted_vertex_buffer(all_indices, vertex_coords, texture_coords, normal_coords)

        # vertex for drawing
        vertex = ObjectLoader.buffer.copy()
        ObjectLoader.buffer = [] # free up memory

        return np.array(indices, dtype=np.uint32), np.array(vertex, dtype=np.float32)

#=================================================================================================
#=================================================================================================


# Camera class
#=================================================================================================
#================================================================================================= 
 
# Camera class that controls the camera's movement using mouse and keyboard
class Camera:
    def __init__(self):
        # camera attributes
        self.camera_position = pyrr.Vector3([0.0, 0.0, 10.0]) # where is camera located
        self.camera_front = pyrr.Vector3([0.0, 0.0, -1.0]) # the direction of where camera looks
        self.camera_up = pyrr.Vector3([0.0, 1.0, 0.0])
        self.camera_right = pyrr.Vector3([1.0, 0.0, 0.0])
        self.mouse_sensitivity = 0.1   # mouse sensitivity
        self.camera_yaw = -90  # camera yaw left and right
        self.camera_pitch = 0  # camera pitch up and down

    def get_view_matrix(self):
        return pyrr.matrix44.create_look_at(self.camera_position, 
                                            self.camera_position + self.camera_front,
                                            self.camera_up)

    def process_mouse_movement(self, x_offset, y_offset, constrain_pitch=True):
        x_offset *= self.mouse_sensitivity
        y_offset *= self.mouse_sensitivity

        self.camera_yaw += x_offset
        self.camera_pitch += y_offset
        
        # give constrain to the camera pitch range
        if constrain_pitch:
            if self.camera_pitch > 45:
                self.camera_pitch = 45
            if self.camera_pitch < -45:
                self.camera_pitch = -45

        # update the camera vector
        self.update_camera_vector()

    # process keyboard press action
    def process_keyboard(self, direction, velocity):
        if direction == "FORWARD":
            if self.camera_position.z <= 3:
                self.camera_position += self.camera_front * 0
            else:
                self.camera_position += self.camera_front * velocity
                
        if direction == "BACKWARD":
            if self.camera_position.z >= 30:
                self.camera_position -= self.camera_front * 0
            else:
                self.camera_position -= self.camera_front * velocity
        if direction == "LEFT":
            self.camera_position -= self.camera_right * velocity
        if direction == "RIGHT":
            self.camera_position += self.camera_right * velocity             

    def update_camera_vector(self):
        front = pyrr.Vector3([0.0, 0.0, 0.0])
        front.x = math.cos(math.radians(self.camera_yaw)) * math.cos(math.radians(self.camera_pitch))
        front.y = math.sin(math.radians(self.camera_pitch))
        front.z = math.sin(math.radians(self.camera_yaw)) * math.cos(math.radians(self.camera_pitch))

        # normalise the camera vector
        self.camera_front = pyrr.vector.normalise(front)
        self.camera_right = pyrr.vector.normalise(pyrr.vector3.cross(self.camera_front, pyrr.Vector3([0.0, 1.0, 0.0])))
        self.camera_up = pyrr.vector.normalise(pyrr.vector3.cross(self.camera_right, self.camera_front))

#=================================================================================================
#=================================================================================================


# camera setup
#=================================================================================================
#=================================================================================================

camera = Camera()
last_x, last_y = WIDTH / 2, HEIGHT / 2
mouse_first_enter = True
left, right, forward, backward = False, False, False, False

#=================================================================================================
#=================================================================================================



# peripharal device handler
#=================================================================================================
#=================================================================================================

# this function will be called everytime we move the mouse inside the glfw window
def mouse_look_callback(window, x_pos, y_pos):
    global mouse_first_enter, last_x, last_y

    # if mouse just entered the window, the mouse position will be reset to mouse's x and y position in glfw window
    if mouse_first_enter:
        last_x = x_pos
        last_y = y_pos
        mouse_first_enter = False

    x_offset = x_pos - last_x
    y_offset = last_y - y_pos # mouse y-axis start from top to bottom and OpenGL y-axis starts from bottom to top

    last_x = x_pos
    last_y = y_pos

    camera.process_mouse_movement(x_offset, y_offset, True) # limit mouse range

# this function will be called everytime the mouse is entered the window or leave the window
def mouse_enter_callback(window, entered):
    global mouse_first_enter

    if entered:
        mouse_first_enter = False
    else:
        mouse_first_enter = True

# this function will be called everytime there is a key pressed in the keyboard
def keyboard_input_callback(window, key, scancode, action, mode):
    global left, right, forward, backward

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        #glfw.set_window_should_close(window, True)
        exit()
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False
    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False

    # reset the key status once the key has been released
    # if key in [glfw.KEY_A, glfw.KEY_D, glfw.KEY_W, glfw.KEY_S] and action == glfw.RELEASE:
    #     left, right, forward, backward = False, False, False, False

# enable continuous movement while key is pressed
def do_movement():
    if left:
        camera.process_keyboard("LEFT", 0.5)
    if right:
        camera.process_keyboard("RIGHT", 0.5)
    if forward:
        camera.process_keyboard("FORWARD", 0.8)
    if backward:
        camera.process_keyboard("BACKWARD", 0.8)

# this function will be called everytime when we resize the window
def resize_window(window, width, height):
    glViewport(0, 0, width, height)
    # generate a new projection matrix everytime window is resized
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width/height, 0.1, 10000)
    # pass the matrix to shader
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

#=================================================================================================
#=================================================================================================



# virtual mouse setup
#=================================================================================================
#=================================================================================================

# simulate mouse event and keyboard event
keyboard = KeyboardController()
GOForward = False

mouse = MouseController()
app = wx.App(False)
# screen resolution
screen_x, screen_y = wx.GetDisplaySize()
window_x, window_y = 500, 500

# use cvtColor() method to convert rgb value to a range of hsv value
# as there are lots of variation of one particular color
green = np.uint8([[[0, 0, 255]]])
hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
# our program will only care about green color object and the rest of the
# colors are ignored
lowerLimit = hsvGreen[0][0][0] - 10, 100, 100
upperLimit = hsvGreen[0][0][0] + 10, 255, 255
#lowerLimit = np.array(lowerLimit)
#upperLimit = np.array(upperLimit)

# color detection setup
#===================================================================
""" John """
lowerLimit = np.array([21,68,68])
upperLimit = np.array([41,255,255])
#===================================================================
""" Steven """
# lowerLimit = np.array([20,50,120])
# upperLimit = np.array([100,255,255])
#===================================================================
""" Wanga """
# lowerLimit = np.array([20,50,120])
# upperLimit = np.array([100,255,255])
#===================================================================

# initialize camera object
# todo
# capture = cv2.VideoCapture(0)

# this flag is used to avoid performing mouse click operation continuouly
# and avoid continuous multiple clicks while dragging
clicked = False

#=================================================================================================
#=================================================================================================



# virtual keyboard setup
#=================================================================================================
#=================================================================================================



#=================================================================================================
#=================================================================================================



# window creation
#=================================================================================================
#=================================================================================================

# initialize the glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized")

# create the window
window = glfw.create_window(WIDTH, HEIGHT, "Demo", None, None)
# free up the memory allocated by glfw if window can not be created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created")

# change window's position
glfw.set_window_pos(window, 600 ,200)

# resize window using callback function
glfw.set_window_size_callback(window, resize_window)

# move mouse using callback function
glfw.set_cursor_pos_callback(window, mouse_look_callback)
# enter the window using callback function
glfw.set_cursor_enter_callback(window, mouse_enter_callback)

# keyboard press callback function
glfw.set_key_callback(window, keyboard_input_callback)

# make mouse cursor invisible when it is inside the window client area
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

# before start drawing we need to initialize OpenGL and this is done by creating an OpenGL context
glfw.make_context_current(window)

#=================================================================================================
#=================================================================================================



# loading and view 3D object
#=================================================================================================
#=================================================================================================

# load 3d object file
earth_indices, earth_vertices = ObjectLoader.load_obj('assets/3dObject/earth/earth.obj')
moon_indices, moon_vertices = ObjectLoader.load_obj('assets/3dObject/moon/moon.obj')
mars_indices, mars_vertices = ObjectLoader.load_obj('assets/3dObject/mars/mars.obj')
asteroids_indices, asteroids_vertices = ObjectLoader.load_obj('assets/3dObject/asteroids/asteroids.obj')

# compile shader program
shader = compileProgram(compileShader(vertex_shader, GL_VERTEX_SHADER), compileShader(fragment_shader, GL_FRAGMENT_SHADER))

# vertex array object (VAO)
# for earth and moon
VAO = glGenVertexArrays(4)
# vertex buffer object (VBO)
VBO = glGenBuffers(4)
# element buffer object (EBO)
EBO = glGenBuffers(4)

# earth VAO
# any subsequent vertex attribute calls after glBindVertexArray will be stored inside the VAO
glBindVertexArray(VAO[0])
# earth VBO
glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
glBufferData(GL_ARRAY_BUFFER, earth_vertices.nbytes, earth_vertices, GL_STATIC_DRAW)
# earth EBO
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[0])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, earth_indices.nbytes, earth_indices, GL_STATIC_DRAW)

# put the vertex data to position attribute of the shader program
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, earth_vertices.itemsize * 8, ctypes.c_void_p(0))
# put the texture data to texture attribute of the shader program
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, earth_vertices.itemsize * 8, ctypes.c_void_p(12))
# put the normal data to normal attribute of the shader program
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, earth_vertices.itemsize * 8, ctypes.c_void_p(20))

# moon VAO
glBindVertexArray(VAO[1])
# moon VBO
glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
glBufferData(GL_ARRAY_BUFFER, moon_vertices.nbytes, moon_vertices, GL_STATIC_DRAW)
# moon EBO
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[1])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, moon_indices.nbytes, moon_indices, GL_STATIC_DRAW)

# put the vertex data to position attribute of the shader program
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, moon_vertices.itemsize * 8, ctypes.c_void_p(0))
# put the texture data to texture attribute of the shader program
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, moon_vertices.itemsize * 8, ctypes.c_void_p(12))
# put the normal data to normal attribute of the shader program
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, moon_vertices.itemsize * 8, ctypes.c_void_p(20))

# mars VAO
glBindVertexArray(VAO[2])
# mars VBO
glBindBuffer(GL_ARRAY_BUFFER, VBO[2])
glBufferData(GL_ARRAY_BUFFER, mars_vertices.nbytes, mars_vertices, GL_STATIC_DRAW)
# mars EBO
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[2])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, mars_indices.nbytes, mars_indices, GL_STATIC_DRAW)

# put the vertex data to position attribute of the shader program
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, mars_vertices.itemsize * 8, ctypes.c_void_p(0))
# put the texture data to texture attribute of the shader program
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, mars_vertices.itemsize * 8, ctypes.c_void_p(12))
# put the normal data to normal attribute of the shader program
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, mars_vertices.itemsize * 8, ctypes.c_void_p(20))

# asteroids VAO
glBindVertexArray(VAO[3])
# asteroids VBO
glBindBuffer(GL_ARRAY_BUFFER, VBO[3])
glBufferData(GL_ARRAY_BUFFER, asteroids_vertices.nbytes, asteroids_vertices, GL_STATIC_DRAW)
# asteroids EBO
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[3])
glBufferData(GL_ELEMENT_ARRAY_BUFFER, asteroids_indices.nbytes, asteroids_indices, GL_STATIC_DRAW)

# put the vertex data to position attribute of the shader program
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, asteroids_vertices.itemsize * 8, ctypes.c_void_p(0))
# put the texture data to texture attribute of the shader program
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, asteroids_vertices.itemsize * 8, ctypes.c_void_p(12))
# put the normal data to normal attribute of the shader program
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, asteroids_vertices.itemsize * 8, ctypes.c_void_p(20))


# create texture object
# for earth and moon
texture = glGenTextures(5)

# earth texture
glBindTexture(GL_TEXTURE_2D, texture[0])
# set the texture wrapping parameters (U and V coordinates)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# load image
earth_image = Image.open("assets/3dObject/earth/Diffuse_2K.png")
earth_image = earth_image.transpose(Image.FLIP_TOP_BOTTOM)
earth_img_data = earth_image.convert("RGBA").tobytes()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, earth_image.width, earth_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, earth_img_data)

# moon texture
glBindTexture(GL_TEXTURE_2D, texture[1])
# set the texture wrapping parameters (U and V coordinates)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# load image
moon_image = Image.open("assets/3dObject/moon/Diffuse_2K.png")
moon_image = moon_image.transpose(Image.FLIP_TOP_BOTTOM)
moon_img_data = moon_image.convert("RGBA").tobytes()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, moon_image.width, moon_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, moon_img_data)

# mars texture
glBindTexture(GL_TEXTURE_2D, texture[2])
# set the texture wrapping parameters (U and V coordinates)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# load image
mars_image = Image.open("assets/3dObject/mars/mars.png")
mars_image = mars_image.transpose(Image.FLIP_TOP_BOTTOM)
mars_img_data = mars_image.convert("RGBA").tobytes()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mars_image.width, mars_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, mars_img_data)

# asteroids texture
glBindTexture(GL_TEXTURE_2D, texture[3])
# set the texture wrapping parameters (U and V coordinates)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
# load image
asteroids_image = Image.open("assets/3dObject/asteroids/asteroids.png")
asteroids_image = asteroids_image.transpose(Image.FLIP_TOP_BOTTOM)
asteroids_img_data = asteroids_image.convert("RGBA").tobytes()
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, asteroids_image.width, asteroids_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, asteroids_img_data)

# use the shader program
glUseProgram(shader)

# window clear color
glClearColor(0, 0, 0, 1)

# enable the depth buffer and transparency
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# projection matrix
projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH/HEIGHT, 0.1, 10000)
# tanslation matrix for 3d object's postion
earth_position = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -8]))
#moon_position = pyrr.matrix44.create_from_translation(pyrr.Vector3([50, 0, -20]))
mars_position = pyrr.matrix44.create_from_translation(pyrr.Vector3([-100, 0, -100]))
mars_scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([5, 5, 5]))

# in order to pass matrix to shader program we need to first locate where we should put the matrix
projection_loc = glGetUniformLocation(shader, "projection")
model_loc = glGetUniformLocation(shader, "model")
view_loc = glGetUniformLocation(shader, "view")

# pass the matrix to the shader program
# we only need to pass the projection and view matrix once
glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

#=================================================================================================
#=================================================================================================
# GVIS
GVIS = GestureEvent()


# yolov5 setup
###############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='./weights/best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
opt = parser.parse_args()
print(opt)
check_requirements()

# get webcam
source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://'))

# Directories
save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Initialize
set_logging()
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

# Set Dataloader
vid_path, vid_writer = None, None
if webcam:
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
# else:
#     save_img = True
#     dataset = LoadImages(source, img_size=imgsz, stride=stride)

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
t0 = time.time()

# window loop
#=================================================================================================
#=================================================================================================
for path, img, im0s, vid_cap in dataset:
    # print("iteration")

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            img_c = im0s[i].copy() # copy for GVI
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        print(len(det), end='   ')
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # if save_img or view_img:  # Add bbox to image
                if view_img:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    # print(((int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1])+int(xyxy[3]))/2), end='    ')

                    # GVIS
                    GVIS.update(len(det), xyxy, int(cls))

        # if no gesture found
        else:
            GVIS.no_detect()

        # debug
        cv2.line(im0, (round(GVIS.center_line), 0), (round(GVIS.center_line), 512), (0, 255, 0), thickness=2)

        # Print positions
        GVIS.output()
        

        # Stream results
        if view_img:
            cv2.imshow(str(p), cv2.flip(im0, 1))
            key = cv2.waitKey(1)  # 1 millisecond
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
    

    # todo while loop
    # while not glfw.window_should_close(window)

    # simulate mouse event
    #=================================================================================================
    # read a frame from the camera
    # ret, img_c = capture.read()

    # resize the image frame to a small size for faster processing
    img_c = cv2.resize(img_c, (window_x, window_y))

    # convert the image to hsv format
    imgHSV = cv2.cvtColor(img_c, cv2.COLOR_BGR2HSV)

    # create a binary image of same size of original image
    # only those pixels that are in the hsv range will be 
    # displayed in this mask
    # create a mask for green color
    mask = cv2.inRange(imgHSV, lowerLimit, upperLimit)

    # filtering the mask to eliminate the noise
    # morphological operation: opening
    # remove all the dots randomly poping 
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))

    # morphological operation: closing
    # close the small holes that are present in the actual object
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, np.ones((20, 20)))

    # draw contours from mask
    maskFinal = maskClose
    # RETR_EXTERNAL flag to get the ourter most contour of the shape
    # CHAIN_APPROX_NONE stoes every pixel and CHAIN_APPROX_SIMPLE store only endpoints of the line
    # that forms the contour
    conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # draw all contours in an image
    #cv2.drawContours(img, conts, -1, (255, 0, 0), 3)


    # moving leftward if left hand is g0 and right hand is g1
    if GVIS.cam_move_left == True:
        left = True
    else:
        left = False

    # moving rightward if left hand is g0 and right hand is g1
    if GVIS.cam_move_right == True:
        right = True
    else:
        right = False


    # if there are two contours captured
    if len(conts) == 2:
        if clicked == True:
            clicked = False
            mouse.release(Button.left)
        # finger open gesture, move mouse without click
        x1, y1, w1, h1 = cv2.boundingRect(conts[0])
        x2, y2, w2, h2 = cv2.boundingRect(conts[1])

        #cv2.drawContours(img, conts, -1, (255, 0, 0), 3)

        # draw ractangle 
        cv2.rectangle(img_c, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
        cv2.rectangle(img_c, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 2)

        # center point of two contours
        cx1 = int(x1+w1/2)
        cy1 = int(y1+h1/2)
        cx2 = int(x2+w2/2)
        cy2 = int(y2+h2/2)

        # center point between two fingers
        cx = int((cx1 + cx2) / 2)
        cy = int((cy1 + cy2) / 2)

        # for i in range(len(conts)):
        #     x,y,w,h = cv2.boundingRect(conts[i])
        #     cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

        # handle keyboard event
        # distance between two contours
        distance = abs(cx2 - cx1)

        # moving forward if distance between two hands is larger than 150
        if distance > 200:
            forward = True
        else:
            forward = False

        # moving backward if distance between two hands is larger than 50 and smaller than 120
        if distance >= 0 and distance < 150:
            backward = True
        else:
            backward = False

        # draw line between two fingers
        cv2.line(img_c, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        # draw circle to the mid point
        cv2.circle(img_c, (cx, cy), 2, (0, 0, 255), 2)

        # match the capture window resolution to screen resolution
        mouseLoc = (int(screen_x-(cx*screen_x/window_x)), int(cy*screen_y/window_y))
        mouse.position = mouseLoc
        # move mouse to finger location
        while mouse.position != mouseLoc:
            pass
    
    # if there is only one contour captured
    elif len(conts) == 1:
        # reset keyboard states when there is only one hand detected
        #forward = False
        #backward = False

        if clicked == False:
            clicked = True
            # todo mouse.press(Button.left)
            
        # finger close gesture
        x,y,w,h=cv2.boundingRect(conts[0])

         # for i in range(len(conts)):
         #     x,y,w,h = cv2.boundingRect(conts[i])
         #     cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

        # draw rectangle
        cv2.rectangle(img_c, (x,y), (x+w, y+h), (0, 0, 255), 2)

        # draw circle at center point
        cx = int(x + w/2)
        cy = int(y + h/2)
        cv2.circle(img_c, (cx, cy), int((w+h)/4), (0, 255, 255), 2)
        # For illustration purpose only
        #cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

        # match the capture window resolution to screen resolution
        mouseLoc = (int(screen_x-(cx*screen_x/window_x)), int(cy*screen_y/window_y))
        mouse.position = mouseLoc
        # move mouse to finger location
        while mouse.position != mouseLoc:
            pass
    
    cv2.imshow("capture", cv2.flip(img_c, 1))
    cv2.moveWindow("capture", 10, 10)
    #cv2.imshow("mask", mask)
    #cv2.imshow("maskOpen", maskOpen)
    #cv2.imshow("maskClose", maskClose)
    #cv2.waitKey(10)
    #=================================================================================================


    # simulate keyboard event
    #=================================================================================================
    

    #=================================================================================================


    # handle keyboard and mouse event and draw to the window
    #=================================================================================================
    # enable interaction with window using mouse and keyboard
    glfw.poll_events()

    do_movement() # enable keyboard camera movement

    # clear buffers to preset color value
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # the view matrix will be re-calculated in every frame since camera will be moving
    view = camera.get_view_matrix()
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # rotation matrix
    # it needs to be updated in every iteration
    rotation_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
    rotation_y = pyrr.Matrix44.from_y_rotation(0.3 * glfw.get_time())
    rotation_z = pyrr.Matrix44.from_z_rotation(0.4 * glfw.get_time())
    rotation = pyrr.matrix44.multiply(rotation_x, rotation_y)
    rotation = pyrr.matrix44.multiply(rotation_z, rotation)

    # model matrix combined rotation matrix and translation matrix for earth
    model_G = pyrr.matrix44.multiply(rotation_y, earth_position)

    # Draw earth
    glBindVertexArray(VAO[0])
    glBindTexture(GL_TEXTURE_2D, texture[0])
    # pass model matrix to shader program
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_G)
    glDrawArrays(GL_TRIANGLES, 0, len(earth_indices))
    #glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    moon_position = pyrr.matrix44.create_from_translation(
        pyrr.Vector3([30 * math.cos(0.3 * glfw.get_time()) + 20 * math.sin(0.3 * glfw.get_time()), 
                        0, 
                        -20 * math.cos(0.3 * glfw.get_time()) + 30 * math.sin(0.3 * glfw.get_time())]))
    # model matrix combined rotation matrix and translation matrix for moon
    model_G = pyrr.matrix44.multiply(rotation, moon_position) 

    # Draw moon
    glBindVertexArray(VAO[1])
    glBindTexture(GL_TEXTURE_2D, texture[1])
    # pass model matrix to shader program
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_G)
    glDrawArrays(GL_TRIANGLES, 0, len(moon_indices))

    # model matrix combined rotation matrix and translation matrix for moon
    model_G = pyrr.matrix44.multiply(rotation, mars_scale)
    model_G = pyrr.matrix44.multiply(model_G, mars_position)

    # Draw mars
    glBindVertexArray(VAO[2])
    glBindTexture(GL_TEXTURE_2D, texture[2])
    # pass model matrix to shader program
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_G)
    glDrawArrays(GL_TRIANGLES, 0, len(mars_indices))

    # OpenGL use double buffer, swapping between frot and back buffer is necessary
    glfw.swap_buffers(window)
    #=================================================================================================

    #event check
    if GVIS.exit_check > 5:
        exit()

    # Print time (inference + NMS)
    t2 = time_synchronized()
    print(f'{s}Done. ({t2 - t1:.3f}s)')

#=================================================================================================
#=================================================================================================

# when window is terminated, glfw library need to be closed and free up the allocated memory
glfw.terminate()