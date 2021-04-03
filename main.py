from tkinter import *
from tkinter import filedialog
import os

root = Tk()

root.title("GVIS")

# resize root window
root.geometry("500x500")

def open_render_program():
    #open the program
    os.system('python 3dRender.py')

def open_simulation_program():
    #open the program
    os.system('python 3dSimulation.py')

def enable_gesture_control_level_1():
    #open the program
    os.system('python GVI_Color.py')

def enable_gesture_control_level_2():
    #open the program
    os.system('python GVI.py')

def enable_virtual_mouse():
    #open the program
    os.system('python virtualMouse.py')

my_button = Button(root, text="Rendering Program", command=open_render_program)
my_button.pack(pady=20)
my_label = Label(root, text="")
my_label.pack(pady=20)

my_button = Button(root, text="Simulation Program", command=open_simulation_program)
my_button.pack(pady=10)
my_label = Label(root, text="")
my_label.pack(pady=20)

my_button = Button(root, text="Enable gesture control (Color detection)", command=enable_gesture_control_level_1)
my_button.pack(pady=10)
my_label = Label(root, text="")
my_label.pack(pady=20)

my_button = Button(root, text="Enable gesture control (GVI)", command=enable_gesture_control_level_2)
my_button.pack(pady=10)
my_label = Label(root, text="")
my_label.pack(pady=20)

my_button = Button(root, text="Virtual Mouse", command=enable_virtual_mouse)
my_button.pack(pady=10)
my_label = Label(root, text="")
my_label.pack(pady=20)

root.mainloop()