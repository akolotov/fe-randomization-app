#!/usr/bin/env python3
"""
WRO Future Engineers Randomization app

This app randomizes the vehicle starting position, inner walls configuration
and obstacles positions for WRO Future Engineers game.

It is a Flask app that generates images of the game mat with randomized
elements.

The type of the challenge as well as the challenge driving direction are
defined through the URL endpoints:

- `/qualification/cw` - the open challenge round with clockwise driving
direction

- `/qualification/ccw` - the open challenge round with counter-clockwise
driving direction

- `/final/cw` - the obstacle challenge round with clockwise driving direction

- `/final/ccw` - the obstacle challenge round with counter-clockwise driving
direction
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

The game mat is a square with dimensions of 3 meters by 3 meters, divided
into 9 equal sections, each sized 1 meter by 1 meter. The sections are
organized as follows:

1. **Corner Sections**:
    - There are 4 corner sections, located in the four outer corners of
    the mat. These sections are referred to as "C" in the schematic below.

2. **Straightforward Sections**:
    - There are 4 straightforward sections, which are linear and connect
    opposite corners. These sections are positioned as:

        - 2 horizontal straightforward sections (one above and one below the
        central section).
        - 2 vertical straightforward sections (one to the left and one to the
        right of the central section).

    - These sections are referred to as "S" in the schematic.

3. **Central Section**:
    - The middle section is non-functional and serves as the
    **central section**. This area is referred to as "###" in the schematic
    and remains unused during gameplay.

**Pseudographic Representation of the Game Mat**

To visually aid the understanding of the layout, here’s a pseudographic
representation of the game mat:

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

- The game mat consists only of the sections described above, with no built-in
walls or obstacles.

- **Walls**: Outer and inner walls are additional game elements that are
placed around or on the mat during the game, but they are not part of the mat
itself.

- The **outer walls** typically run along the edges of the mat, while
**inner walls** can be configured in various ways, potentially enclosing the
central section, but these walls are separate from the mat structure.

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

Each straightforward section on the game mat has a structured layout,
featuring the following elements:

1. **Radiuses**:
    - There are **three radiuses** within each straightforward section:

        - **Two radiuses** run vertically along the **left and right edges**
        of the section. They extend from the central section (top) toward the
        outer part of the field (bottom).

        - **One central radius** is positioned in the **middle** of the
        section, running vertically from the top to the bottom of the section.

    - These radiuses divide the section into two columns of zones.

2. **Arcs**:
    - There are **two arcs** that run horizontally across the section:

        - The arcs are positioned at two levels, creating divisions across the
        section horizontally.

        - These arcs do not connect the corner sections but instead divide the
        section into multiple zones horizontally.

3. **Intersections**:
    - The layout of radiuses and arcs forms multiple intersections, which can
    be used to place obstacles such as traffic signs:

        - **4 T-intersections**: These occur where the arcs meet the radiuses.

        - **2 X-intersections**: These occur where the central radius
        intersects with the arcs.

    - Each intersection is a potential location for placing obstacles that
    vehicles must navigate during the game.

4. **Zones**:

    - The combination of the two arcs and three radiuses divides the
    straightforward section into **6 zones**:

        - **Top and bottom zones** are taller and provide more space.

        - **Middle zones** are shorter, as they are compressed by the arcs.

    - These zones serve as key areas for gameplay, where obstacles can be
    placed or specific actions may occur.

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

- The three radiuses (left, right, and middle) run vertically from the outer
part of the game field (at the top) to the central section (at the bottom).

- The two arcs divide the section horizontally into three rows of zones, with
the middle row being shorter than the top and bottom rows.

- The intersections formed by the radiuses and arcs serve as key locations for
obstacle placement.

**Labels for the Intersection Points**

- Intersection points in the top row are labeled "T4", "X2", and "T3" from
left to right.

- Intersection points in the bottom row are labeled "T2", "X1", and "T1" from
left to right.

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

# In order to be able to randomize the inner walls configuration
# as well as the obstacles positions, it is necessary to define
# the main points of the game mat in pixels.

# The size of the game mat is 3 meters by 3 meters,
# which is 3000 pixels by 3000 pixels.
# The width and height do not include the outerwalls.
width = 3000
height = width

# The straightforward section is 1000x1000 pixels.
# The field has 4 straightforward sections with the identical layout.
# Due to the symmetry, it is enough to define coordinates of the elements for
# one section relative to the top left corner of the game mat (0,0).
# The coordinates below are for the section labeled "Section N".

# Arcs:
# The first line is the first arc in the straightforward section.
first_line = 400
# The second line is the second arc in the straightforward section.
second_line = first_line + 200

# The inner border is the border between the straightforward section and the
# central section.
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
narrow_color = (255, 0, 0)  # The color is blue
narrow_thickness = 20

# The color to mark the starting zone is grey
start_section_color = (192, 192, 192)

# The color of parking lot barriers is magenta
parking_lot_color = (255, 0, 255)
parking_barrier_thickness = 20
parking_barrier_length = 200
distance_between_parking_barriers = 300

# The obstacle will be represented as a square with the side of 100 pixels.
obstacle_size = 100


def on_north(img, h1, w1, h2, w2, c):
    """
    Draw a square with the given color and the given relative coordinates
    in the Section N
    """

    img[
        min(h1, h2) + border:max(h1, h2)+border,
        min(w1, w2) + border:max(w1, w2)+border
        ] = c


def on_south(img, h1, w1, h2, w2, c):
    """
    Draw a square with the given color and the given relative coordinates
    in the Section S
    """

    img[
        height-max(h1, h2) + border:height-min(h1, h2)+border,
        width-max(w1, w2) + border:width-min(w1, w2)+border
        ] = c


def on_west(img, h1, w1, h2, w2, c):
    """
    Draw a square with the given color and the given relative coordinates
    in the Section W
    """

    img[
        height-max(w1, w2)+border:height-min(w1, w2) + border,
        min(h1, h2)+border:max(h1, h2) + border
        ] = c


def on_east(img, h1, w1, h2, w2, c):
    """
    Draw a square with the given color and the given relative coordinates
    in the Section E
    """

    img[
        min(w1, w2) + border:max(w1, w2) + border,
        width-max(h1, h2)+border:width-min(h1, h2)+border
        ] = c


class Section(Enum):
    """
    Represents a straightforward section.

    Value is a function to draw a square in the
    corresponding straightforward section.
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
    RED = (55, 39, 238)
    GREEN = (44, 214, 68)
    UNDEFINED = (0, 0, 0)


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
        Draw a square obstacle in the straightforward section defined by the
        given function `section`
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

# Intersections that will be in front of the vehicle for the given direction
# and the given start zone.
#
# Direction clockwise:
#   Z3: T1, T3
#   Z4: X1, X2
#
# Direction counter-clockwise:
#   Z3: X1, X2
#   Z4: T2, T4


forbidden_intersections_in_start_zone = {
    Direction.CW: {
        StartZone.Z3: [Intersection.T1, Intersection.T3],
        StartZone.Z4: [Intersection.X1, Intersection.X2]
    },
    Direction.CCW: {
        StartZone.Z3: [Intersection.X1, Intersection.X2],
        StartZone.Z4: [Intersection.T2, Intersection.T4]
    }
}

# According to the rules, these intersections in the straightforward section
# containing the parking lot cannot be used for the obstacle placement.
forbidden_intersections_in_parking_section = [
    Intersection.T3,
    Intersection.T4,
    Intersection.X2
]


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
        Draw a vehicle starting zone in the straightforward section defined
        by the given function `section`
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
        Check if the inner wall is closer to the outer wall on the
        northern side of the game mat.
        """
        return self._north

    def on_west(self):
        """
        Check if the inner wall is closer to the outer wall on the
        western side of the game mat.
        """
        return self._west

    def on_south(self):
        """
        Check if the inner wall is closer to the outer wall on the
        southern side of the game mat.
        """
        return self._south

    def on_east(self):
        """
        Check if the inner wall is closer to the outer wall on the eastern
        side of the game mat.
        """
        return self._east

    def on_side(self, side: Section):
        """
        Check if the inner wall is closer to the outer wall on the given side
        of the game mat.
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
        h_n = inner_border           # Y coordinate of the northern inner wall
        w_w = inner_border           # X coordinate of the western inner wall
        h_s = height - inner_border  # Y coordinate of the southern inner wall
        w_e = width - inner_border   # X coordinate of the eastern inner wall

        # Adjust the position of the inner walls based on which side of the
        # game mat the inner wall should be drawn closer to the outer walls
        # - the wall is positioned along the second arc of the
        # corresponding straightforward section.

        if self.on_north():
            h_n = second_line
        if self.on_west():
            w_w = second_line
        if self.on_south():
            h_s = height - second_line
        if self.on_east():
            w_e = width - second_line

        # north
        img[
            h_n - (border//2) + border:h_n + (border//2) + border,
            w_w - (border//2) + border:w_e + (border//2) + border
            ] = (0, 0, 0)
        # west
        img[
            h_n - (border//2) + border:h_s + (border//2) + border,
            w_w - (border//2) + border:w_w + (border//2) + border
            ] = (0, 0, 0)
        # south
        img[
            h_s - (border//2) + border:h_s + (border//2) + border,
            w_w - (border//2) + border:w_e + (border//2) + border
            ] = (0, 0, 0)
        # east
        img[
            h_n - (border//2) + border:h_s + (border//2) + border,
            w_e - (border//2) + border:w_e + (border//2) + border
            ] = (0, 0, 0)

# Randomization process operates with sets of obstacles.
# Each element of the list defines relative positions of the obstacles in the
# straightforward section.
# The duplicates for Card 14, 15, 20, 21 are removed to have results
# more valuable from evaluation point of view.


obstacles_sets = [
    # Single intersection obstacles (T1)
    [Obstacle(Intersection.T1, Color.GREEN)],                # 0, Card 1
    [Obstacle(Intersection.T1, Color.RED)],                  # 1, Card 2

    # Single intersection obstacles (X1)
    [Obstacle(Intersection.X1, Color.GREEN)],                # 2, Card 3
    [Obstacle(Intersection.X1, Color.RED)],                  # 3, Card 4

    # Single intersection obstacles (T2)
    [Obstacle(Intersection.T2, Color.GREEN)],                # 4, Card 5
    [Obstacle(Intersection.T2, Color.RED)],                  # 5, Card 6

    # Single intersection obstacles (T3)
    [Obstacle(Intersection.T3, Color.GREEN)],                # 6, Card 7
    [Obstacle(Intersection.T3, Color.RED)],                  # 7, Card 8

    # Single intersection obstacles (X2)
    [Obstacle(Intersection.X2, Color.GREEN)],                # 8, Card 9
    [Obstacle(Intersection.X2, Color.RED)],                  # 9, Card 10

    # Single intersection obstacles (T4)
    [Obstacle(Intersection.T4, Color.GREEN)],                # 10, Card 11
    [Obstacle(Intersection.T4, Color.RED)],                  # 11, Card 12

    # T3 and T2 combinations
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T2, Color.GREEN)],                # 12, Card 13
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T2, Color.RED)],                  # 13, Card 14/16
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T2, Color.GREEN)],                # 14, Card 15/17
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T2, Color.RED)],                  # 15, Card 18

    # T1 and T4 combinations
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T4, Color.GREEN)],                # 16, Card 19
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T4, Color.RED)],                  # 17, Card 20/22
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T4, Color.GREEN)],                # 18, Card 21/23
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T4, Color.RED)],                  # 19, Card 24

    # T1 and T2 combinations
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T2, Color.GREEN)],                # 20, Card 25
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T2, Color.RED)],                  # 21, Card 26
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T2, Color.GREEN)],                # 22, Card 27
    [Obstacle(Intersection.T1, Color.GREEN),
     Obstacle(Intersection.T2, Color.RED)],                  # 23, Card 28
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T2, Color.GREEN)],                # 24, Card 29
    [Obstacle(Intersection.T1, Color.RED),
     Obstacle(Intersection.T2, Color.RED)],                  # 25, Card 30

    # T3 and T4 combinations
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T4, Color.GREEN)],                # 26, Card 31
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T4, Color.RED)],                  # 27, Card 32
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T4, Color.GREEN)],                # 28, Card 33
    [Obstacle(Intersection.T3, Color.GREEN),
     Obstacle(Intersection.T4, Color.RED)],                  # 29, Card 34
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T4, Color.GREEN)],                # 30, Card 35
    [Obstacle(Intersection.T3, Color.RED),
     Obstacle(Intersection.T4, Color.RED)],                  # 31, Card 36
]

# The randomization process says that at least one of the
# straightforward sections must have at least one obstacle
# in the intersection labeled as "X2". The map contains indices
# of the corresponding obstacle sets for the green and red obstacles.

mandatory_obstacles_sets = {
    Color.GREEN: 8,
    Color.RED: 9
}

# One of the obstacles sets from this list must be present on the game field to
# make to reduce risk when incomplete solutions solve the challenge.

required_obstacles_sets = [21, 22, 27, 28]

# Relative coordinates of the vehicle starting zones in the
# straightforward sections for the Open challenge rounds.

vehicle_positions_in_open = [
    VehiclePosition(StartZone.Z6),
    VehiclePosition(StartZone.Z5),
    VehiclePosition(StartZone.Z4),
    VehiclePosition(StartZone.Z3),
    VehiclePosition(StartZone.Z2),
    VehiclePosition(StartZone.Z1)
]

# Relative coordinates of the vehicle starting zones in the
# straightforward sections
# for the Obstacle challenge rounds.

vehicle_positions_in_obstacle = [
    VehiclePosition(StartZone.Z4),
    VehiclePosition(StartZone.Z3)
]

# game field image template
template = np.zeros((height+border * 2,
                     width+border * 2, 3), np.uint8)

# The game field is white square with the border which
# represents the outer walls.
# The color of the border is black.

template[
    border:height + border,
    border:width+border
    ] = (255, 255, 255)

# The lines representing arcs in the "Section N"
template[
    first_line - (thin_line//2) + border:
    first_line + (thin_line//2) + border,
    border + inner_border:width - inner_border + border
    ] = (0, 0, 0)

template[
    second_line - (thin_line//2) + border:
    second_line + (thin_line//2) + border,
    border + inner_border: width - inner_border + border
    ] = (0, 0, 0)

# One line representing the left radiuse of the "Section W", the border of the
# "Section N" with the central section and the right radius of the "Section E".
template[
    inner_border - (thin_line//2) + border:
    inner_border + (thin_line//2) + border,
    border:width + border
    ] = (0, 0, 0)

# The lines representing arcs in the "Section S"
template[
    height - first_line - (thin_line//2) + border:
    height - first_line + (thin_line//2) + border,
    border + inner_border: width - inner_border + border
    ] = (0, 0, 0)

template[
    height - second_line - (thin_line//2) + border:
    height - second_line + (thin_line//2) + border,
    border + inner_border: width - inner_border + border
    ] = (0, 0, 0)

# One line representing the right radius of the "Section W", the border of the
# "Section S" with the central section and the left radius of the "Section E".
template[
    height - inner_border - (thin_line//2) + border:
    height - inner_border + (thin_line//2) + border,
    border: width + border
    ] = (0, 0, 0)

# The lines representing arcs in the "Section W"
template[
    border+inner_border:
    height-inner_border+border,
    first_line-(thin_line//2)+border:
    first_line+(thin_line//2)+border
    ] = (0, 0, 0)
template[
    border + inner_border:
    height - inner_border + border,
    second_line - (thin_line//2) + border:
    second_line + (thin_line//2) + border
    ] = (0, 0, 0)

# One line representing the left radius of the "Section N", the border of the
# "Section E" with the central section and the right radius of the "Section S".
template[
    border:height + border,
    inner_border - (thin_line//2) + border:
    inner_border + (thin_line//2) + border
    ] = (0, 0, 0)

# The lines representing arcs in the "Section E"
template[
    border + inner_border:
    height - inner_border+border,
    width - first_line - (thin_line//2) + border:
    width - first_line + (thin_line//2) + border
    ] = (0, 0, 0)

template[
    border + inner_border:
    height - inner_border+border,
    width - second_line - (thin_line//2) + border:
    width - second_line + (thin_line//2) + border
    ] = (0, 0, 0)

# One line representing the right radius of the "Section N", the border of the
# "Section W" with the central section and the left radius of the "Section S".
template[
    border: height + border,
    width - inner_border - (thin_line//2) + border:
    width - inner_border + (thin_line//2) + border
    ] = (0, 0, 0)

# The line representing the central radius of the "Section N"
template[
    border: inner_border + border,
    (width//2) - (thin_line//2) + border:
    (width//2) + (thin_line//2) + border
    ] = (0, 0, 0)

# The line representing the central radius of the "Section S"
template[
    height - inner_border + border:
    height + border,
    (width//2) - (thin_line//2) + border:
    (width//2) + (thin_line//2) + border
    ] = (0, 0, 0)

# The line representing the central radius of the "Section W"
template[
    (height//2) - (thin_line//2) + border:
    (height//2) + (thin_line//2) + border,
    border: inner_border + border
    ] = (0, 0, 0)

# The line representing the central radius of the "Section E"
template[
    (height//2) - (thin_line//2) + border:
    (height//2) + (thin_line//2) + border,
    width - inner_border + border: width + border
    ] = (0, 0, 0)


def draw_parking_lot_barriers(img, section: Section):
    """
    Draw the parking lot barriers in the given section.
    """
    # coordinates of the top left corner of the first barrier
    # relatively to the section:

    first_barrier_top_left = (
        left_position,  # x coordinate
        0               # y coordinate - aligned with top edge
        )

    # coordinates of the bottom right corner of the first barrier
    # relatively to the section:

    first_barrier_bottom_right = (
        left_position + parking_barrier_thickness,  # x coordinate
        parking_barrier_length  # y coordinate - extends down by barrier length
        )

    # coordinates of the top left corner of the second barrier
    # relatively to the section:

    second_barrier_top_left = (
        left_position + parking_barrier_thickness +
        distance_between_parking_barriers,  # x coordinate
        0           # y coordinate - aligned with top edge
        )

    # coordinates of the bottom right corner of the second barrier
    # relatively to the section:

    second_barrier_bottom_right = (
        left_position +
        parking_barrier_thickness +
        distance_between_parking_barriers +
        parking_barrier_thickness,  # x coordinate
        parking_barrier_length  # y coordinate - extends down by barrier length
        )

    # Draw both barriers

    section(img,
            first_barrier_top_left[1],
            first_barrier_top_left[0],
            first_barrier_bottom_right[1],
            first_barrier_bottom_right[0],
            parking_lot_color
            )

    section(img,
            second_barrier_top_left[1],
            second_barrier_top_left[0],
            second_barrier_bottom_right[1],
            second_barrier_bottom_right[0],
            parking_lot_color
            )


def draw_obstacles_set(img, section: Section, obstacles_set: list[Obstacle]):
    """
    Draw a set of obstacles defined by the elements of the
    obstacles set `obstacles_set` in the straightforward
    section defined by the given function `section`.
    """

    for obstacle in obstacles_set:
        obstacle.draw(img, section)


def draw_narrow(img, direction: Direction):
    """
    Draw the narrow arc in the central section of the game mat.
    `direction` contains the driving direction of the vehicle.
    """

    # The center of the game mat is calculated with taking into
    # account the outer walls.
    img_center = (width // 2 + border, height // 2 + border)

    axes = (narrow_radius, narrow_radius)
    startA = 180
    endA = -90
    # Draw the arc
    img = cv2.ellipse(img, img_center, axes, 0, startA, endA,
                      narrow_color, narrow_thickness)

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
    Draw the game field for the Obstacle challenge rounds.

    The scheme is a dictionary with the following keys:

    - start_section: the straightforward section where the starting
    zone is located

    - start_zone: the position of the starting zone in the chosen
    straightforward section

    - obstacles: a dictionary where keys are indices of the obstacles
    sets and values are sections where the obstacles are located

    - parking_section: the section where the parking lot is located

    Returns a 3-dimensional NumPy array (matrix) representing the game
    field where every pixel is represented by three numbers corresponding
    to the BGR color.
    """

    image = template.copy()

    # Create the vehicle starting position object for the given zone
    # and draw it in the chosen straightforward section
    VehiclePosition(scheme['start_zone']).draw(image, scheme['start_section'])

    # Draw the parking lot barriers in the parking section
    draw_parking_lot_barriers(image, scheme['parking_section'])

    # Draw the obstacles in the corresponding sections
    obstacles_configuration = scheme['obstacles']
    for obstacles_set_index in obstacles_configuration:
        draw_obstacles_set(
            image, obstacles_configuration[obstacles_set_index],
            obstacles_sets[obstacles_set_index]
            )

    return image


def randomize_and_draw_layout_for_open(direction: Direction) -> np.ndarray:
    """
    Generate the game field for the Open challenge rounds.

    Returns a 3-dimensional NumPy array (matrix) representing the game
    field where every pixel is represented by three numbers corresponding
    to the BGR color.
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
        allowed_zones = [
            StartZone.Z6, StartZone.Z5,
            StartZone.Z4, StartZone.Z3
            ]

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
    Generate the game field for the Obstacle challenge rounds.

    Returns a 3-dimensional NumPy array (matrix) representing the game
    field where every pixel is represented by three numbers corresponding
    to the BGR color.
    """

    # The set of intersections that will be in front of the vehicle
    # in the start zone for the given driving direction.
    forbidden_intersections = forbidden_intersections_in_start_zone[direction]

    # Look for the obstacles sets that satisfy the conditions:
    #
    # - the difference between the number of green and red obstacles is
    # not greater than one
    #
    # - the total number of obstacles is at least 5
    #
    # - there is at least one valid start zone for the given combination
    # of obstacles
    satisfied = False

    while not satisfied:
        # Choose the index of the mandatory obstacles set.
        mandatory_set_color = choice([Color.GREEN, Color.RED])
        mandatory_obstacles_set = mandatory_obstacles_sets[mandatory_set_color]

        # Choose the index of the required obstacles set.
        required_obstacles_set = choice(required_obstacles_sets)

        # Choose the index of the obstacles set for one of two remaining
        # sections.
        os1 = mandatory_obstacles_set

        while os1 == required_obstacles_set or os1 == mandatory_obstacles_set:
            os1 = randint(0, len(obstacles_sets) - 1)

        # Choose the index of the obstacles set for the last remaining section.
        os2 = mandatory_obstacles_set

        while os2 == required_obstacles_set or  \
            os2 == mandatory_obstacles_set or \
                os2 == os1:
            os2 = randint(0, len(obstacles_sets) - 1)

        chosen_obstacles_sets_indices = [
            mandatory_obstacles_set,
            required_obstacles_set,
            os1,
            os2
            ]

        # Calculate the number of obstacles, the number of green and
        # red obstacles and the forbidden start zones for choosen
        # obstacles sets.
        forbidden_start_zones = {}
        obstacles_set_conflicting_with_parking_section = set()
        obstacles_amount = 0
        green_amount = 0
        red_amount = 0

        for obstacles_set_index in chosen_obstacles_sets_indices:
            one_obstacles_set = obstacles_sets[obstacles_set_index]

            obstacles_amount = obstacles_amount + len(one_obstacles_set)

            forbidden_start_zones[obstacles_set_index] = set()

            for one_obstacle in one_obstacles_set:
                if one_obstacle.is_green():
                    green_amount = green_amount + 1

                elif one_obstacle.is_red():
                    red_amount = red_amount + 1

                else:
                    raise ValueError("Unknown obstacle color")

                # Check if the current obstacle would be in front of the
                # vehicle for each possible starting zone in this section
                for zone in forbidden_intersections:
                    if one_obstacle.position in forbidden_intersections[zone]:
                        forbidden_start_zones[obstacles_set_index].add(zone)

                # Check if the current obstacle's position is suitable for the
                # section where the parking lot is located.
                for intersection in forbidden_intersections_in_parking_section:
                    if one_obstacle.position == intersection:
                        obstacles_set_conflicting_with_parking_section.add(
                            obstacles_set_index
                            )

        # Remove obstacle sets where both possible start zones are forbidden,
        # keeping only sets that have at least one valid start zone.
        for obstacles_set_index in forbidden_start_zones:
            if len(forbidden_start_zones[obstacles_set_index]) == 2:
                del forbidden_start_zones[obstacles_set_index]

        # Get all obstacle sets that are suitable for the parking section.
        obstacles_set_suitable_for_parking_section = set(
            chosen_obstacles_sets_indices
            ) - \
            obstacles_set_conflicting_with_parking_section

        # Stops to look for the obstacles sets if the conditions are satisfied:
        #
        # - the difference between the number of green and red obstacles is
        # not greater than one
        #
        # - the total number of obstacles is at least 5
        #
        # - there is at least one valid start zone for the given combination
        # of obstacles
        #
        # - there is at least one obstacle set that is suitable for the parking
        # section
        satisfied = (abs(green_amount - red_amount) <= 1) and \
            (obstacles_amount > 4) and \
            (len(forbidden_start_zones) > 0) and \
            (len(obstacles_set_suitable_for_parking_section) > 0)

    # Cannot use list(Section) because elements of Section are functions.
    sections = [Section.NORTH, Section.WEST, Section.SOUTH, Section.EAST]

    # Randomly assign each obstacle set to a unique section of the game field
    shuffled_sections = sample(sections, 4)
    sections_for_obstacles_sets = {}
    for obstacles_set_index in chosen_obstacles_sets_indices:
        sections_for_obstacles_sets[obstacles_set_index] =  \
            shuffled_sections.pop()

    # Choose one of the obstacle sets that has at least one valid start zone.
    obstacles_set_in_start_section = choice(list(forbidden_start_zones.keys()))

    # Choose the section where the chosen obstacle set is located.
    start_section = sections_for_obstacles_sets[obstacles_set_in_start_section]

    # Choose one of the obstacle sets that is suitable for the parking section.
    obstacles_set_in_parking_section = choice(
        list(obstacles_set_suitable_for_parking_section)
        )

    # Choose the section where the chosen obstacle set is located.
    parking_section = sections_for_obstacles_sets[
        obstacles_set_in_parking_section
        ]

    # Choose one of the valid start zones for the chosen obstacle set.
    start_zone = choice(
        list(set([StartZone.Z3, StartZone.Z4]) -
             forbidden_start_zones[obstacles_set_in_start_section]
             ))

    scheme = {
        'start_section': start_section,
        'start_zone': start_zone,
        'obstacles': sections_for_obstacles_sets,
        'parking_section': parking_section
    }
    image = draw_scheme_for_final(scheme)

    # Draw the inner walls
    InnerWall().draw(image)

    # Draw the narrow arc in the central section
    draw_narrow(image, direction)

    return image


# HTTP Content related


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


# HTTP endpoints


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
