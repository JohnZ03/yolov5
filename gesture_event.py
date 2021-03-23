from math import sqrt

cam_x_size = 1024
cam_y_size = 512
lost_thresh = 3



class GestureEvent:
    def __init__(self):
        self.event_resolved = False
        self.hand_R = GestureObject((-1,-1,-1,-1), 0)
        self.hand_L = GestureObject((-1,-1,-1,-1), 1024)
        self.R_lost_track = 0
        self.L_lost_track = 0
        self.hand_count = 0
        self.center_line = cam_x_size/2
        self.exit_check = 0
        

    def __del__(self):
        print("")

    def update(self, number_of_gestures, x, gesture_type):
        # note: more than 0.07ms*3frams lost track will be registered as new
        if number_of_gestures < 3:                
            get_cx = (int(x[0])+int(x[2]))/2
            get_cy = (int(x[1])+int(x[3]))/2

            # todo: use centre of both hands as threshold
            if 0<= get_cx and get_cx < self.center_line:
                self.hand_R.update(get_cx, get_cy, gesture_type)
                # if only right hand detected
                if number_of_gestures == 1:
                    self.hand_L.lost()
            else:
                self.hand_L.update(get_cx, get_cy, gesture_type)
                # if only left hand detected
                if number_of_gestures == 1:
                    self.hand_R.lost()
            
            # update centerline
            self.center_line = (self.hand_L.cx + self.hand_R.cx)/2

            # check for quit
            # quit both g0 y>380; both move dis>100
            if self.hand_L.gesture_type == 0 and self.hand_R.gesture_type == 0:
                self.exit_check += 1
            else:
                if self.exit_check > 0:
                    self.exit_check -= 1


    def output(self):
        print((self.hand_L.cx,self.hand_L.cy,self.hand_L.gesture_type), end='    ')
        print(self.center_line, end='     ')
        print((self.hand_R.cx,self.hand_R.cy,self.hand_R.gesture_type), end='    ')

    def no_detect(self):
        self.hand_L.lost()
        self.hand_R.lost()
        self.center_line = cam_x_size/2


class GestureObject:
    def __init__(self, x, def_x):
        self.cx = def_x
        self.cy = (int(x[1])+int(x[3]))/2
        self.old_cx = def_x
        self.old_cy = (int(x[1])+int(x[3]))/2
        self.lost_track = 0
        self.missing = True
        self.def_x = def_x
        self.gesture_type = -1

    def update(self, new_cx, new_cy, gesture_type):
        self.lost_track = 0
        self.old_cx = self.cx
        self.old_cy = self.cy
        self.cx = new_cx
        self.cy = new_cy
        self.missing = False
        self.gesture_type = gesture_type

    def lost(self):
        self.lost_track += 1
        if self.lost_track > lost_thresh:
            # todo: reset self
            self.cx = self.def_x
            self.cy = -1
            self.missing = True
            self.type = -1


def cal_distance(x1, x2, y1, y2):
    return sqrt((x1-x2)**2+(y1-y2)**2)