#!/usr/bin/env python3

from flask import Flask, make_response, render_template
import cv2
import numpy as np
from random import randint, choice

app = Flask(__name__)
app.debug = True

# coordinates

width = 3000
height = width

first_line = 400
second_line = first_line + 200
inner_border = second_line + 400

border = 10
thin_line = 2

left_position = inner_border
middle_position = width // 2
right_position = width - inner_border

narrow_radius = 350
narrow_color = (255, 0, 0)
narrow_thickness = 20

start_section = (192, 192, 192)
red_obstacle = (55,39,238)
green_obstacle = (44, 214, 68)

obstacles_coords = [
    (second_line, right_position),
    (second_line, middle_position),
    (second_line, left_position),
    (first_line, right_position),    
    (first_line, middle_position),
    (first_line, left_position)
]

obstacle_size = 100

obstacles_sets = [
    [0],    # 1
    [1],    # 2
    [2],    # 3
    [0, 2], # 4
    [3],    # 5
    [4],    # 6
    [5],    # 7
    [3, 5], # 8
    [0, 5], # 9
    [2, 3], # 10
    []      # 11
]
mandatory_obstacles_set = 5

car_positions_in_qualif = [
    ((0, left_position), (first_line, middle_position)),
    ((0, middle_position), (first_line, right_position)),
    ((first_line, left_position), (second_line, middle_position)),
    ((first_line, middle_position), (second_line, right_position)),
    ((second_line, left_position), (inner_border, middle_position)),
    ((second_line, middle_position), (inner_border, right_position))    
]

car_positions_in_final = [
    ((first_line, left_position), (second_line, middle_position)),
    ((first_line, middle_position), (second_line, right_position))
]

img_center = (width // 2 + border, height // 2 + border)

font = cv2.FONT_HERSHEY_SIMPLEX
text_position = (width // 2 - 285 + border, height // 2 + 50 + border)
fontScale = 6

# game field template

template = np.zeros((height+border*2,width+border*2,3), np.uint8)
template[border:height+border,border:width+border] = (255,255,255)

template[first_line-(thin_line//2)+border:first_line+(thin_line//2)+border,border+inner_border:width-inner_border+border] = (0,0,0)
template[second_line-(thin_line//2)+border:second_line+(thin_line//2)+border,border+inner_border:width-inner_border+border] = (0,0,0)
template[inner_border-(thin_line//2)+border:inner_border+(thin_line//2)+border,border:width+border] = (0,0,0)

template[height-first_line-(thin_line//2)+border:height-first_line+(thin_line//2)+border,border+inner_border:width-inner_border+border] = (0,0,0)
template[height-second_line-(thin_line//2)+border:height-second_line+(thin_line//2)+border,border+inner_border:width-inner_border+border] = (0,0,0)
template[height-inner_border-(thin_line//2)+border:height-inner_border+(thin_line//2)+border,border:width+border] = (0,0,0)

template[border+inner_border:height-inner_border+border,first_line-(thin_line//2)+border:first_line+(thin_line//2)+border] = (0,0,0)
template[border+inner_border:height-inner_border+border,second_line-(thin_line//2)+border:second_line+(thin_line//2)+border] = (0,0,0)
template[border:height+border,inner_border-(thin_line//2)+border:inner_border+(thin_line//2)+border] = (0,0,0)

template[border+inner_border:height-inner_border+border,width-first_line-(thin_line//2)+border:width-first_line+(thin_line//2)+border] = (0,0,0)
template[border+inner_border:height-inner_border+border,width-second_line-(thin_line//2)+border:width-second_line+(thin_line//2)+border] = (0,0,0)
template[border:height+border,width-inner_border-(thin_line//2)+border:width-inner_border+(thin_line//2)+border] = (0,0,0)

template[border:inner_border+border,(width//2)-(thin_line//2)+border:(width//2)+(thin_line//2)+border] = (0,0,0)
template[height-inner_border+border:height+border,(width//2)-(thin_line//2)+border:(width//2)+(thin_line//2)+border] = (0,0,0)
template[(height//2)-(thin_line//2)+border:(height//2)+(thin_line//2)+border,border:inner_border+border] = (0,0,0)
template[(height//2)-(thin_line//2)+border:(height//2)+(thin_line//2)+border,width-inner_border+border:width+border] = (0,0,0)

def on_north(img, h1, w1, h2, w2, c):
    img[min(h1,h2)+border:max(h1,h2)+border, min(w1,w2)+border:max(w1,w2)+border] = c

def on_south(img, h1, w1, h2, w2, c):
    img[height-max(h1,h2)+border:height-min(h1,h2)+border, width-max(w1,w2)+border:width-min(w1,w2)+border] = c
    
def on_west(img, h1, w1, h2, w2, c):
    img[height-max(w1,w2)+border:height-min(w1,w2)+border, min(h1,h2)+border:max(h1,h2)+border] = c

def on_east(img, h1, w1, h2, w2, c):
    img[min(w1,w2)+border:max(w1,w2)+border, width-max(h1,h2)+border:width-min(h1,h2)+border] = c
    
def draw_obstacle(img, f, h, w, c):
    f(img,
      h-(obstacle_size//2), w-(obstacle_size//2),
      h+(obstacle_size//2), w+(obstacle_size//2),
      c)

def draw_obstacles_set(img, f, s, o):
    closer_color = green_obstacle if o == 'cw' else red_obstacle
    further_color = red_obstacle if o == 'cw' else green_obstacle
    
    for obstacle in s:
        if (obstacle < 3):
            draw_obstacle(img, f, obstacles_coords[obstacle][0], obstacles_coords[obstacle][1], closer_color)
        else: 
            draw_obstacle(img, f, obstacles_coords[obstacle][0], obstacles_coords[obstacle][1], further_color)

def draw_inner_walls(img, code):    
    h_n = inner_border
    w_w = inner_border
    h_s = height - inner_border
    w_e = width - inner_border    
    
    if code[3] == '1':
        h_n = second_line
    if code[2] == '1':
        w_w = second_line
    if code[1] == '1':
        h_s = height - second_line
    if code[0] == '1':
        w_e = width - second_line

    # north
    img[h_n-(border//2)+border:h_n+(border//2)+border,w_w-(border//2)+border:w_e+(border//2)+border] = (0,0,0)
    # west
    img[h_n-(border//2)+border:h_s+(border//2)+border,w_w-(border//2)+border:w_w+(border//2)+border] = (0,0,0)
    # south
    img[h_s-(border//2)+border:h_s+(border//2)+border,w_w-(border//2)+border:w_e+(border//2)+border] = (0,0,0)
    # east
    img[h_n-(border//2)+border:h_s+(border//2)+border,w_e-(border//2)+border:w_e+(border//2)+border] = (0,0,0)

def draw_narrow(img, o):
    axes = (narrow_radius, narrow_radius)
    startA = 180
    endA = -90
    img = cv2.ellipse(img, img_center, axes, 0, startA, endA, narrow_color, narrow_thickness)

    if o == 'cw':
        startP = (img_center[0] - narrow_radius, img_center[1])
        endP = (img_center[0] - narrow_radius - 30, img_center[1] + 80)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)
        endP = (img_center[0] - narrow_radius + 50, img_center[1] + 75)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)
    elif o == 'ccw':
        startP = (img_center[0], img_center[1] - narrow_radius)
        endP = (img_center[0] + 80, img_center[1] - narrow_radius - 30)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)
        endP = (img_center[0] + 75, img_center[1] - narrow_radius + 50)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)

def draw_scheme_for_final(scheme):
    image = template.copy()
    
    f_list = [on_north, on_west, on_south, on_east]
    
    car_pos = scheme['start_position']
    car_side = scheme['start_side']
    f_list[car_side](image, 
                     car_positions_in_final[car_pos][0][0], car_positions_in_final[car_pos][0][1], 
                     car_positions_in_final[car_pos][1][0], car_positions_in_final[car_pos][1][1],
                     start_section)
    
    direct = scheme['driving_direction']    
    north = scheme['obstacles_on_north']
    west = scheme['obstacles_on_west']
    south = scheme['obstacles_on_south']
    east = scheme['obstacles_on_east']
    
    draw_obstacles_set(image, on_north, north, direct)
    draw_obstacles_set(image, on_west, west, direct)
    draw_obstacles_set(image, on_south, south, direct)
    draw_obstacles_set(image, on_east, east, direct)

    image[inner_border-(border//2)+border:height-inner_border+(border//2)+border,inner_border-(border//2)+border:inner_border+(border//2)+border] = (0,0,0)
    image[inner_border-(border//2)+border:height-inner_border+(border//2)+border,width-inner_border-(border//2)+border:width-inner_border+(border//2)+border] = (0,0,0)

    image[inner_border-(border//2)+border:inner_border+(border//2)+border,inner_border-(border//2)+border:width-inner_border+(border//2)+border] = (0,0,0)
    image[height-inner_border-(border//2)+border:height-inner_border+(border//2)+border,inner_border-(border//2)+border:width-inner_border+(border//2)+border] = (0,0,0)

    return image

def rotate90ccw(img):
    M = cv2.getRotationMatrix2D((img_center), 90, 1)
    image = cv2.warpAffine(img, M, (width+border*2, height+border*2))
    return image

def generate_layout(layout_type='qualification', direction='cw'):
    
    if layout_type == 'qualification':
        f_list = [on_north, on_west, on_south, on_east]
        
        walls_config = randint(0,15)
        side = randint(0,3)

        code = bin(walls_config)[2:]
        code = '0' * (4 - len(code)) + code
        
        if code[3-side] == '1':
            allowed_pos = 3
        else:
            allowed_pos = 5

        car_pos = randint(0, allowed_pos)

        image = template.copy()

        f_list[side](image, 
                     car_positions_in_qualif[car_pos][0][0], car_positions_in_qualif[car_pos][0][1], 
                     car_positions_in_qualif[car_pos][1][0], car_positions_in_qualif[car_pos][1][1],
                     start_section)

        draw_inner_walls(image, code)

        draw_narrow(image, direction)

    elif layout_type == 'final':
        
        r1 = 0
        r2 = 0
        r3 = 0
        while (r1 == r2) or (r2 == r3) or (r1 == r3):
            r1 = randint(0, len(obstacles_sets)-1)
            r2 = randint(0, len(obstacles_sets)-1)
            r3 = randint(0, len(obstacles_sets)-1)
        
        m = randint(0,3)
        if m == 0:
            obs_sets = [obstacles_sets[mandatory_obstacles_set],
                        obstacles_sets[r1],
                        obstacles_sets[r2],
                        obstacles_sets[r3]
                       ]
        elif m == 1:
            obs_sets = [obstacles_sets[r1],
                        obstacles_sets[mandatory_obstacles_set],
                        obstacles_sets[r2],
                        obstacles_sets[r3]
                       ]
        elif m == 2:
            obs_sets = [obstacles_sets[r1],
                        obstacles_sets[r2],
                        obstacles_sets[mandatory_obstacles_set],
                        obstacles_sets[r3]
                       ]
        elif m == 3:
            obs_sets = [obstacles_sets[r1],
                        obstacles_sets[r2],
                        obstacles_sets[r3],
                        obstacles_sets[mandatory_obstacles_set]
                       ]
        
        sp = randint(0, 1)
        side = randint(0, 3)

        scheme = {
            'start_side': side,
            'start_position': sp,
            'driving_direction': direction,
            'obstacles_on_north': obs_sets[0],
            'obstacles_on_west': obs_sets[1],
            'obstacles_on_south': obs_sets[2],
            'obstacles_on_east': obs_sets[3]
        }
        image = draw_scheme_for_final(scheme)
        
        draw_narrow(image, scheme['driving_direction'])
        
    return image


#### HTTP Content related

def generate_image(img):
    res, im_png = cv2.imencode('.png', img)
    image = im_png.tobytes()
    response = make_response(image)
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Cache-Control', 'no-store')
    return response

#### HTTP endpoints

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/qualification/cw')
def generate_qualification_cw():
    layout = generate_layout()
    response = generate_image(layout)
    return response

@app.route('/qualification/ccw')
def generate_qualification_ccw():
    layout = generate_layout(direction='ccw')
    response = generate_image(layout)
    return response

@app.route('/final/cw')
def generate_final_cw():
    layout = generate_layout(layout_type='final')
    response = generate_image(layout)
    return response

@app.route('/final/ccw')
def generate_final_ccw():
    layout = generate_layout(layout_type='final', direction='ccw')
    response = generate_image(layout)
    return response

if __name__ == '__main__':
    app.run()