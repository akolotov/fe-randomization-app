#!/usr/bin/env python3
"""
WRO Future Engineers Randomization app

This app randomizes the vehicle starting position, inner walls configuration and obstacles positions for WRO Future Engineers game.

It is a Flask app that generates images of the game mat with randomized elements.

The type of the challenge as well as the challenge driving direction are defined through the URL endpoints:

- `/qualification/cw` - the open challenge round with clockwise driving direction
- `/qualification/ccw` - the open challenge round with counter-clockwise driving direction
- `/final/cw` - the obstacle challenge round with clockwise driving direction
- `/final/ccw` - the obstacle challenge round with counter-clockwise driving direction
"""

from flask import Flask, make_response, render_template
import cv2
import numpy as np
from random import randint, choice, sample
from enum import Enum

app = Flask(__name__)
app.debug = True

"""
### General Game Mat description

The game mat is a square with dimensions of 3 meters by 3 meters, divided into 9 equal sections, each sized 1 meter by 1 meter. The sections are organized as follows:

1. **Corner Sections**:
    - There are 4 corner sections, located in the four outer corners of the mat. These sections are referred to as "C" in the schematic below.
2. **Straightforward Sections**:
    - There are 4 straightforward sections, which are linear and connect opposite corners. These sections are positioned as:
        - 2 horizontal straightforward sections (one above and one below the central section).
        - 2 vertical straightforward sections (one to the left and one to the right of the central section).
    - These sections are referred to as "S" in the schematic.
3. **Central Section**:
    - The middle section is non-functional and serves as the **central section**. This area is referred to as "###" in the schematic and remains unused during gameplay.

**Pseudographic Representation of the Game Mat**

To visually aid the understanding of the layout, here’s a pseudographic representation of the game mat:

```
+---+---+---+
|   |   |   |
| C | S | C |
|   |   |   |
+---+---+---+
| S |###| S |
|   |###|   |  Central Section (###)
|   |###|   |  (Unused during gameplay)
+---+---+---+
|   |   |   |
| C | S | C |
|   |   |   |
+---+---+---+
```

**Important Notes:**

- The game mat consists only of the sections described above, with no built-in walls or obstacles.
- **Walls**: Outer and inner walls are additional game elements that are placed around or on the mat during the game, but they are not part of the mat itself.
- The **outer walls** typically run along the edges of the mat, while **inner walls** can be configured in various ways, potentially enclosing the central section, but these walls are separate from the mat structure.

**Labels for the straightforward sections**

- The section at the top is labeled "Section N"
- The section at the bottom is labeled "Section S"
- The section on the left is labeled "Section W"
- The section on the right is labeled "Section E"

```
+---+---+---+
|   |   |   |
|   | N |   |
|   |   |   |
+---+---+---+
| W |###| E |
|   |###|   |
|   |###|   |
+---+---+---+
|   |   |   |
|   | S |   |
|   |   |   |
+---+---+---+
```

### Straightforward Section Layout Description

Each straightforward section on the game mat has a structured layout, featuring the following elements:

1. **Radiuses**:
    - There are **three radiuses** within each straightforward section:
        - **Two radiuses** run vertically along the **left and right edges** of the section. They extend from the central section (top) toward the outer part of the field (bottom).
        - **One central radius** is positioned in the **middle** of the section, running vertically from the top to the bottom of the section.
    - These radiuses divide the section into two columns of zones.
2. **Arcs**:
    - There are **two arcs** that run horizontally across the section:
        - The arcs are positioned at two levels, creating divisions across the section horizontally.
        - These arcs do not connect the corner sections but instead divide the section into multiple zones horizontally.
3. **Intersections**:
    - The layout of radiuses and arcs forms multiple intersections, which can be used to place obstacles such as traffic signs:
        - **4 T-intersections**: These occur where the arcs meet the radiuses.
        - **2 X-intersections**: These occur where the central radius intersects with the arcs.
    - Each intersection is a potential location for placing obstacles that vehicles must navigate during the game.
4. **Zones**:
    - The combination of the two arcs and three radiuses divides the straightforward section into **6 zones**:
        - **Top and bottom zones** are taller and provide more space.
        - **Middle zones** are shorter, as they are compressed by the arcs.
    - These zones serve as key areas for gameplay, where obstacles can be placed or specific actions may occur.

**Pseudographic Representation of the Straightforward Section:**

```
radius  radius
v       v
+---+---+
|   |   |
|   |   |
|   |   |
+---+---+ - arc
|   |   |
|   |   |
+---+---+ - arc
|   |   |
|   |   |
|   |   |
+---+---+
    ^
    radius
```

In this schematic:

- The three radiuses (left, right, and middle) run vertically from the outer part of the game field (at the top) to the central section (at the bottom).
- The two arcs divide the section horizontally into three rows of zones, with the middle row being shorter than the top and bottom rows.
- The intersections formed by the radiuses and arcs serve as key locations for obstacle placement.

**Labels for the Intersection Points**

- Intersection points in the top row are labeled "T4", "X2", and "T3" from left to right.
- Intersection points in the bottom row are labeled "T2", "X1", and "T1" from left to right.

**Labels for the zones**

- Zones in the top row are labeled "Z6" and "Z5"
- Zones in the middle row are labeled "Z4" and "Z3"
- Zones in the bottom row are labeled "Z2" and "Z1"

```
+----+----+
|    |    |
| Z6 | Z5 |
|    |    |
T4---X2---T3
| Z4 | Z3 |
|    |    |
T2---X1---T1
|    |    |
| Z2 | Z1 |
|    |    |
+----+----+
```
"""

# coordinates

# In order to be able to randomize the inner walls configuration as well as the obstacles positions,
# it is necessary to define the main points of the game mat in pixels.

# The size of the game mat is 3 meters by 3 meters, which is 3000 pixels by 3000 pixels.
# The width and height do not include the outerwalls.
width = 3000
height = width

# The straightforward section is 1000x1000 pixels.
# The field has 4 straightforward sections with the identical layout.
# Due to the symmetry, it is enough to define coordinates of the elements for one section relative to the top left corner of the game mat (0,0).
# The coordinates below are for the section labeled "Section N".

# Arcs:
# The first line is the first arc in the straightforward section.
first_line = 400
# The second line is the second arc in the straightforward section.
second_line = first_line + 200

# The inner border is the border between the straightforward section and the central section.
inner_border = second_line + 400

# Radiuses:
# The left radius is the left radius in the straightforward section.
left_position = inner_border
# The middle radius is the middle radius in the straightforward section.
middle_position = width // 2
# The right radius is the right radius in the straightforward section.
right_position = width - inner_border

# Thickness of lines representing the walls
border = 10

# Thickness of the lines representing the radiuses and arcs
thin_line = 2

# The presentation of the challenge driving direction is a narrow arc
# in the central section of the game mat.
narrow_radius = 350
narrow_color = (255, 0, 0)
narrow_thickness = 20

# The color to mark the starting zone
start_section_color = (192, 192, 192)

# The obstacle will be represented as a square with the side of 100 pixels.
obstacle_size = 100

def on_north(img, h1, w1, h2, w2, c):
    """
    Draw a square with the given color and the given relative coordinates in the Section N
    """

    img[min(h1,h2)+border:max(h1,h2)+border, min(w1,w2)+border:max(w1,w2)+border] = c

def on_south(img, h1, w1, h2, w2, c):
    """
    Draw a square with the given color and the given relative coordinates in the Section S
    """

    img[height-max(h1,h2)+border:height-min(h1,h2)+border, width-max(w1,w2)+border:width-min(w1,w2)+border] = c

def on_west(img, h1, w1, h2, w2, c):
    """
    Draw a square with the given color and the given relative coordinates in the Section W
    """

    img[height-max(w1,w2)+border:height-min(w1,w2)+border, min(h1,h2)+border:max(h1,h2)+border] = c

def on_east(img, h1, w1, h2, w2, c):
    """
    Draw a square with the given color and the given relative coordinates in the Section E
    """

    img[min(w1,w2)+border:max(w1,w2)+border, width-max(h1,h2)+border:width-min(h1,h2)+border] = c

class Section(Enum):
    """
    Represents a straightforward section.

    Value is a function to draw a square in the corresponding straightforward section.
    """
    NORTH = on_north
    SOUTH = on_south
    WEST = on_west
    EAST = on_east

class Direction(Enum):
    CW = 'cw'
    CCW = 'ccw'

    @classmethod
    def is_cw(cls, direction):
        return direction == cls.CW
    
    @classmethod
    def is_ccw(cls, direction):
        return direction == cls.CCW

class ChallengeType(Enum):
    OPEN = 'open'
    OBSTACLE = 'obstacle'

# Intersection points in the straightforward sections
class Intersection(Enum):
    TopLeft = (0, left_position)
    TopMiddle = (0, middle_position)
    TopRight = (0, right_position)
    T4 = (first_line, left_position)
    X2 = (first_line, middle_position)
    T3 = (first_line, right_position)
    T2 = (second_line, left_position)
    X1 = (second_line, middle_position)
    T1 = (second_line, right_position)
    BottomLeft = (inner_border, left_position)
    BottomMiddle = (inner_border, middle_position)
    BottomRight = (inner_border, right_position)

class Color(Enum):
    RED = (55,39,238)
    GREEN = (44, 214, 68)
    UNDEFINED = (0,0,0)

class Obstacle:
    """
    Represents an obstacle on the game mat.

    The obstacle is defined by the position and the color.
    """

    def __init__(self, position: Intersection, color: Color):
        self.position = position
        self.color = color

    def set_color(self, color: Color):
        self.color = color

    def is_red(self):
        return self.color == Color.RED

    def is_green(self):
        return self.color == Color.GREEN

    def _x(self):
        return self.position.value[0]

    def _y(self):
        return self.position.value[1]

    def _color(self):
        return self.color.value

    def draw(self, img: np.ndarray, section: Section):
        """
        Draw a square obstacle in the straightforward section defined by the given function `section`
        """

        section(img,
          self._x()-(obstacle_size//2), self._y()-(obstacle_size//2),
          self._x()+(obstacle_size//2), self._y()+(obstacle_size//2),
          self._color())

class StartZone(Enum):
    Z1 = (Intersection.X1, Intersection.BottomRight)
    Z2 = (Intersection.T2, Intersection.BottomMiddle)
    Z3 = (Intersection.X2, Intersection.T1)
    Z4 = (Intersection.T4, Intersection.X1)
    Z5 = (Intersection.TopMiddle, Intersection.T3)
    Z6 = (Intersection.TopLeft, Intersection.X2)

class VehiclePosition:
    """
    Represents a vehicle starting position on the game mat.
    """

    def __init__(self, start_zone: StartZone):
        self.start_zone = start_zone

    def _top_left_x(self):
        return self.start_zone.value[0].value[0]

    def _top_left_y(self):
        return self.start_zone.value[0].value[1]

    def _bottom_right_x(self):
        return self.start_zone.value[1].value[0]

    def _bottom_right_y(self):
        return self.start_zone.value[1].value[1]

    def draw(self, img: np.ndarray, section: Section):
        """
        Draw a vehicle starting zone in the straightforward section defined by the given function `section`
        """

        section(img,
          self._top_left_x(), self._top_left_y(),
          self._bottom_right_x(), self._bottom_right_y(),
          start_section_color)

class InnerWall:
    """
    Represents an inner wall on the game mat.
    """

    def __init__(self, sides: list[Section] = []):
        self._north = Section.NORTH in sides
        self._west = Section.WEST in sides
        self._south = Section.SOUTH in sides
        self._east = Section.EAST in sides

    def on_north(self):
        """
        Check if the inner wall is closer to the outer wall on the northern side of the game mat.
        """
        return self._north

    def on_west(self):
        """
        Check if the inner wall is closer to the outer wall on the western side of the game mat.
        """
        return self._west

    def on_south(self):
        """
        Check if the inner wall is closer to the outer wall on the southern side of the game mat.
        """
        return self._south

    def on_east(self):
        """
        Check if the inner wall is closer to the outer wall on the eastern side of the game mat.
        """
        return self._east
    
    def on_side(self, side: Section):
        """
        Check if the inner wall is closer to the outer wall on the given side of the game mat.
        """
        if side == Section.NORTH:
            return self.on_north()
        elif side == Section.WEST:
            return self.on_west()
        elif side == Section.SOUTH:
            return self.on_south()
        elif side == Section.EAST:
            return self.on_east()

    def draw(self, img: np.ndarray):
        """
        Draw the inner walls of the game mat.
        """

        # default position of the inner walls
        h_n = inner_border # Y coordinate of the northern inner wall
        w_w = inner_border # X coordinate of the western inner wall
        h_s = height - inner_border # Y coordinate of the southern inner wall
        w_e = width - inner_border # X coordinate of the eastern inner wall
        
        # Adjust the position of the inner walls based on which side of the game mat
        # the inner wall should be drawn closer to the outer walls - the wall is
        # positioned along the second arc of the corresponding straightforward section.
        if self.on_north():
            h_n = second_line
        if self.on_west():
            w_w = second_line
        if self.on_south():
            h_s = height - second_line
        if self.on_east():
            w_e = width - second_line

        # north
        img[h_n-(border//2)+border:h_n+(border//2)+border,w_w-(border//2)+border:w_e+(border//2)+border] = (0,0,0)
        # west
        img[h_n-(border//2)+border:h_s+(border//2)+border,w_w-(border//2)+border:w_w+(border//2)+border] = (0,0,0)
        # south
        img[h_s-(border//2)+border:h_s+(border//2)+border,w_w-(border//2)+border:w_e+(border//2)+border] = (0,0,0)
        # east
        img[h_n-(border//2)+border:h_s+(border//2)+border,w_e-(border//2)+border:w_e+(border//2)+border] = (0,0,0)

# Randomization process operates with sets of obstacles.
# Each element of the list defines relative positions of the obstacles in the straightforward section.
obstacles_sets = [
    [Obstacle(Intersection.T1, Color.UNDEFINED)],                                             # 0, T1
    [Obstacle(Intersection.X1, Color.UNDEFINED)],                                             # 1, X1
    [Obstacle(Intersection.T2, Color.UNDEFINED)],                                             # 2, T2
    [Obstacle(Intersection.T1, Color.UNDEFINED), Obstacle(Intersection.T2, Color.UNDEFINED)], # 3, T1, T2
    [Obstacle(Intersection.T3, Color.UNDEFINED)],                                             # 4, T3
    [Obstacle(Intersection.X2, Color.UNDEFINED)],                                             # 5, X2
    [Obstacle(Intersection.T3, Color.UNDEFINED), Obstacle(Intersection.T4, Color.UNDEFINED)], # 6, T3, T4
    [Obstacle(Intersection.T1, Color.UNDEFINED), Obstacle(Intersection.T4, Color.UNDEFINED)], # 7, T1, T4
    [Obstacle(Intersection.T2, Color.UNDEFINED), Obstacle(Intersection.T3, Color.UNDEFINED)], # 8, T2, T3
    []      # 9, empty
]
# The randomization process says that at least one of the straightforward sections must
# have at least one obstacle in the intersection labeled as "X2".
mandatory_obstacles_set = 5

# Relative coordinates of the vehicle starting zones in the straightforward sections 
# for the Open challenge rounds.
vehicle_positions_in_open = [
    VehiclePosition(StartZone.Z6),
    VehiclePosition(StartZone.Z5),
    VehiclePosition(StartZone.Z4),
    VehiclePosition(StartZone.Z3),
    VehiclePosition(StartZone.Z2),
    VehiclePosition(StartZone.Z1)
]

# Relative coordinates of the vehicle starting zones in the straightforward sections 
# for the Obstacle challenge rounds.
vehicle_positions_in_obstacle = [
    VehiclePosition(StartZone.Z4),
    VehiclePosition(StartZone.Z3)
]

# game field image template
template = np.zeros((height+border*2,width+border*2,3), np.uint8)

# The game field is white square with the border which represents the outer walls.
# The color of the border is black.
template[border:height+border,border:width+border] = (255,255,255)

# The lines representing arcs in the "Section N"
template[first_line-(thin_line//2)+border:first_line+(thin_line//2)+border,border+inner_border:width-inner_border+border] = (0,0,0)
template[second_line-(thin_line//2)+border:second_line+(thin_line//2)+border,border+inner_border:width-inner_border+border] = (0,0,0)

# One line representing the left radiuse of the "Section W", the border of the "Section N" with the central section and the right radius of the "Section E".
template[inner_border-(thin_line//2)+border:inner_border+(thin_line//2)+border,border:width+border] = (0,0,0)

# The lines representing arcs in the "Section S" 
template[height-first_line-(thin_line//2)+border:height-first_line+(thin_line//2)+border,border+inner_border:width-inner_border+border] = (0,0,0)
template[height-second_line-(thin_line//2)+border:height-second_line+(thin_line//2)+border,border+inner_border:width-inner_border+border] = (0,0,0)

# One line representing the right radius of the "Section W", the border of the "Section S" with the central section and the left radius of the "Section E".
template[height-inner_border-(thin_line//2)+border:height-inner_border+(thin_line//2)+border,border:width+border] = (0,0,0)

# The lines representing arcs in the "Section W"
template[border+inner_border:height-inner_border+border,first_line-(thin_line//2)+border:first_line+(thin_line//2)+border] = (0,0,0)
template[border+inner_border:height-inner_border+border,second_line-(thin_line//2)+border:second_line+(thin_line//2)+border] = (0,0,0)

# One line representing the left radius of the "Section N", the border of the "Section E" with the central section and the right radius of the "Section S".
template[border:height+border,inner_border-(thin_line//2)+border:inner_border+(thin_line//2)+border] = (0,0,0)

# The lines representing arcs in the "Section E"
template[border+inner_border:height-inner_border+border,width-first_line-(thin_line//2)+border:width-first_line+(thin_line//2)+border] = (0,0,0)
template[border+inner_border:height-inner_border+border,width-second_line-(thin_line//2)+border:width-second_line+(thin_line//2)+border] = (0,0,0)

# One line representing the right radius of the "Section N", the border of the "Section W" with the central section and the left radius of the "Section S".
template[border:height+border,width-inner_border-(thin_line//2)+border:width-inner_border+(thin_line//2)+border] = (0,0,0)

# The line representing the central radius of the "Section N"
template[border:inner_border+border,(width//2)-(thin_line//2)+border:(width//2)+(thin_line//2)+border] = (0,0,0)
# The line representing the central radius of the "Section S"
template[height-inner_border+border:height+border,(width//2)-(thin_line//2)+border:(width//2)+(thin_line//2)+border] = (0,0,0)
# The line representing the central radius of the "Section W"
template[(height//2)-(thin_line//2)+border:(height//2)+(thin_line//2)+border,border:inner_border+border] = (0,0,0)
# The line representing the central radius of the "Section E"
template[(height//2)-(thin_line//2)+border:(height//2)+(thin_line//2)+border,width-inner_border+border:width+border] = (0,0,0)

def draw_obstacles_set(img, f, s, o):
    """
    Draw a set of obstacles defined by the elements of the obstacles set `s`
    in the straightforward section defined by the given function `f`.
    `o` contains the driving direction of the vehicle.

    In the season 2021 it was assumed that the obstacle closer to the central section
    is green and the one further from the central section is red for the clockwise direction.
    For the counterclockwise direction the situation is opposite.
    """

    closer_color = Color.GREEN if Direction.is_cw(o) else Color.RED
    further_color = Color.RED if Direction.is_cw(o) else Color.GREEN
    
    for obstacle in s:
        if (obstacle.position in [Intersection.T1, Intersection.T2, Intersection.X1]):
            obstacle.set_color(closer_color)
        else: 
            obstacle.set_color(further_color)

        obstacle.draw(img, f)

def draw_narrow(img, direction: Direction):
    """
    Draw the narrow arc in the central section of the game mat.
    `direction` contains the driving direction of the vehicle.
    """

    # The center of the game mat is calculated with taking into account the outer walls.
    img_center = (width // 2 + border, height // 2 + border)

    axes = (narrow_radius, narrow_radius)
    startA = 180
    endA = -90
    # Draw the arc
    img = cv2.ellipse(img, img_center, axes, 0, startA, endA, narrow_color, narrow_thickness)

    # Draw the arrow at the end of the arc
    if Direction.is_cw(direction):
        startP = (img_center[0] - narrow_radius, img_center[1])
        endP = (img_center[0] - narrow_radius - 30, img_center[1] + 80)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)
        endP = (img_center[0] - narrow_radius + 50, img_center[1] + 75)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)
    elif Direction.is_ccw(direction):
        startP = (img_center[0], img_center[1] - narrow_radius)
        endP = (img_center[0] + 80, img_center[1] - narrow_radius - 30)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)
        endP = (img_center[0] + 75, img_center[1] - narrow_radius + 50)
        img = cv2.line(img, startP, endP, narrow_color, narrow_thickness)

def draw_scheme_for_final(scheme):
    """
    Draw the game mat for the Obstacle challenge rounds.

    The scheme is a dictionary with the following keys:
    - start_section: the straightforward section where the starting zone is located
    - start_zone: the position of the starting zone in the chosen straightforward section
    - driving_direction: the challenge driving direction
    - obstacles_on_north: the set of obstacles in the Section N
    - obstacles_on_west: the set of obstacles in the Section W
    - obstacles_on_south: the set of obstacles in the Section S
    - obstacles_on_east: the set of obstacles in the Section E
    """

    image = template.copy()
        
    # Create the vehicle starting position object for the given zone
    # and draw it in the chosen straightforward section
    VehiclePosition(scheme['start_zone']).draw(image, scheme['start_section'])
    
    direction = scheme['driving_direction']    
    section_north = scheme['obstacles_on_north']
    section_west = scheme['obstacles_on_west']
    section_south = scheme['obstacles_on_south']
    section_east = scheme['obstacles_on_east']
    
    # Draw the obstacles in the corresponding sections
    draw_obstacles_set(image, Section.NORTH, section_north, direction)
    draw_obstacles_set(image, Section.WEST, section_west, direction)
    draw_obstacles_set(image, Section.SOUTH, section_south, direction)
    draw_obstacles_set(image, Section.EAST, section_east, direction)

    return image

def randomize_and_draw_layout_for_open(direction: Direction) -> np.ndarray:
    """
    Generate the game mat for the Open challenge rounds.

    Returns a 3-dimensional NumPy array (matrix) representing the game mat where
    every pixel is represented by three numbers corresponding to the BGR color.
    """

    # Cannot use list(Section) because elements of Section are functions.
    sections = [Section.NORTH, Section.WEST, Section.SOUTH, Section.EAST]

    # Choose on which sides of the game mat the inner walls should be drawn 
    # closer to the outer walls.
    inner_walls_config = sample(sections, randint(0, 4))
    inner_walls = InnerWall(inner_walls_config)

    # Choose the straightforward section where the starting zone is located.
    starting_section = choice(sections)

    # If the inner wall in the starting section is closer to the outer wall,
    # the starting zone could be only one of the four available zones in
    # the starting section. That is why number of zones used for randomization
    # must be limited.
    if inner_walls.on_side(starting_section):
        allowed_zones = [StartZone.Z6, StartZone.Z5, StartZone.Z4, StartZone.Z3]
    else:
        allowed_zones = list(StartZone)
    # Choose the starting zone within the allowed zones.
    starting_zone = choice(allowed_zones)

    image = template.copy()

    # Create the vehicle starting position object for the given zone
    # and draw it in the chosen straightforward section
    VehiclePosition(starting_zone).draw(image, starting_section)

    # Draw the inner walls
    inner_walls.draw(image)

    # Draw the narrow arc in the central section
    draw_narrow(image, direction)

    return image
        
def randomize_and_draw_layout_for_obstacle(direction: Direction) -> np.ndarray:
    """
    Generate the game mat for the Obstacle challenge rounds.

    Returns a 3-dimensional NumPy array (matrix) representing the game mat where
    every pixel is represented by three numbers corresponding to the BGR color.
    """

    r1 = 0
    r2 = 0
    r3 = 0
    four_or_six = False
    # The proecess to choose the obstacles sets for the straightforward sections
    # must be repeated until the following conditions are met:
    # - all three chosen sets are different
    # - number of obstacles closer to the central section equals
    #   number of the obstacles closer to the outer walls
    # - total number of obstacles is 4 or 6
    while (r1 == r2) or (r2 == r3) or (r1 == r3) or four_or_six:
        # Choose the indces in the obstacle sets for three straightforward sections.
        r1 = randint(0, len(obstacles_sets)-1)
        r2 = randint(0, len(obstacles_sets)-1)
        r3 = randint(0, len(obstacles_sets)-1)

        # With the chosen obstacles sets and the mandatory set calculate the total
        # number of obstacles, the number of obstacles that are closer to the
        # central section and the number of obstacles that are closer to the outer
        # walls.
        obstacles_amount = 0
        inner_amount = 0
        outer_amount = 0
        for one_obstacles_set in [obstacles_sets[mandatory_obstacles_set],
                                    obstacles_sets[r1],
                                    obstacles_sets[r2],
                                    obstacles_sets[r3]
                                    ]:
            obstacles_amount = obstacles_amount + len(one_obstacles_set)
            for one_obstacle in one_obstacles_set:
                if one_obstacle.position in [Intersection.T1, Intersection.T2, Intersection.X1]:
                    inner_amount = inner_amount + 1
                else:
                    outer_amount = outer_amount + 1
        
        # Although the randomiziation process allows completely random sets of
        # obstacles, it makes sense to consider only the cases that are worth
        # to evaluate solutions of the participants:
        # - number of obstacles closer to the central section equals
        #   to amount of the obstacles closer to the outer walls
        # - total number of obstacles is not 2
        four_or_six = False
        if (obstacles_amount != 4) and (obstacles_amount != 6):
            four_or_six = True
        if inner_amount != outer_amount:
            four_or_six = True
    
    # Choose the straightforward section where the mandatory set of obstacles is located.
    m = randint(0,3)

    # Create the list of obstacles sets for the straightforward sections.
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
    
    # Cannot use list(Section) because elements of Section are functions.
    sections = [Section.NORTH, Section.WEST, Section.SOUTH, Section.EAST]

    scheme = {
        'start_section': choice(sections),
        'start_zone': choice([StartZone.Z3, StartZone.Z4]),
        'driving_direction': direction,
        'obstacles_on_north': obs_sets[0],
        'obstacles_on_west': obs_sets[1],
        'obstacles_on_south': obs_sets[2],
        'obstacles_on_east': obs_sets[3]
    }
    image = draw_scheme_for_final(scheme)

    # Draw the inner walls
    InnerWall().draw(image)
    
    # Draw the narrow arc in the central section
    draw_narrow(image, scheme['driving_direction'])

    return image

#### HTTP Content related

def generate_image(img):
    """
    Encode the image from a 3-dimensional NumPy array to the PNG format
    and return it as a HTTP response.
    """

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
    layout = randomize_and_draw_layout_for_open(Direction.CW)
    response = generate_image(layout)
    return response

@app.route('/qualification/ccw')
def generate_qualification_ccw():
    layout = randomize_and_draw_layout_for_open(Direction.CCW)
    response = generate_image(layout)
    return response

@app.route('/final/cw')
def generate_final_cw():
    layout = randomize_and_draw_layout_for_obstacle(Direction.CW)
    response = generate_image(layout)
    return response

@app.route('/final/ccw')
def generate_final_ccw():
    layout = randomize_and_draw_layout_for_obstacle(Direction.CCW)
    response = generate_image(layout)
    return response

if __name__ == '__main__':
    app.run()