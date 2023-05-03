import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import dataclasses
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from metabox.utils import CoordType, ParameterType


@dataclasses.dataclass
class Shape:
    value: ParameterType

    def __post_init__(self):
        self.use_complex = _is_complex(self.value)


@dataclasses.dataclass
class Polygon(Shape):
    points: List[CoordType]

    def __post_init__(self):
        Shape.__post_init__(self)
        if not isinstance(self.points, list):
            raise TypeError("points must be a list.")
        if len(self.points) < 3:
            raise ValueError("Polygon must have at least 3 points.")
        self.points = [_floatt(p) for p in self.points]


@dataclasses.dataclass
class Circle(Shape):
    """Defines a circle.

    Attributes:
        center: The center of the circle.
        radius: The radius of the circle.
    """

    center: CoordType
    radius: ParameterType

    def __post_init__(self):
        Shape.__post_init__(self)
        self.center = _floatt(self.center)


@dataclasses.dataclass
class Rectangle(Shape):
    """Defines a rectangle.

    Attributes:
        center: The center of the rectangle.
        x_width: The width of the rectangle in the x direction.
        y_width: The width of the rectangle in the y direction.
        rotation_deg: The rotation of the rectangle in degrees.
    """

    center: CoordType
    x_width: ParameterType
    y_width: ParameterType
    rotation_deg: ParameterType = 0.0

    def __post_init__(self):
        Shape.__post_init__(self)
        self.center = _floatt(self.center)


@dataclasses.dataclass
class Canvas:
    x_width: ParameterType
    y_width: ParameterType
    spacing: ParameterType = 1.0
    background_value: ParameterType = 0.0
    enforce_4fold_symmetry: bool = False

    """A class for drawing on a canvas.

    Attributes:
        x_width: The width of the canvas in the x direction.
        y_width: The width of the canvas in the y direction.
        spacing: The spacing between pixels. Defaults to 1.
        background_value: The value of the background. Defaults to 0.
        enforce_4fold_symmetry: whether to make the layer 4-fold symmetric.
            If true, then the layer will be mirrored along the x and y axes,
            then added to its transpose.
    """

    def __post_init__(self):
        self.x_pixels = int(self.x_width / self.spacing)
        self.y_pixels = int(self.y_width / self.spacing)
        xx = tf.linspace(-self.x_pixels / 2, self.x_pixels / 2, self.x_pixels)
        yy = tf.linspace(-self.y_pixels / 2, self.y_pixels / 2, self.y_pixels)
        self.xx, self.yy = tf.meshgrid(xx, yy)
        self.map = tf.zeros([self.y_pixels, self.x_pixels])

    def rasterize(self, shape_list: List[Shape]) -> tf.Tensor:
        """Rasterizes a list of shapes.

        High level API for rasterizing a list of shapes.

        Args:
            shape_list: The list of shapes to rasterize.
        """
        if not isinstance(shape_list, list):
            raise TypeError("shape_list must be a list.")

        use_complex = False
        if _is_complex(self.background_value):
            use_complex = True
        else:
            for shape in shape_list:
                if shape.use_complex:
                    use_complex = True
                    break

        if use_complex:
            self.map = tf.cast(self.map, tf.complex64)
            self.background_value = tf.cast(self.background_value, tf.complex64)
        self.use_complex = use_complex

        for shape in shape_list:
            self.merge_shape(shape, enforce_4fold_symmetry=self.enforce_4fold_symmetry)

        if use_complex:
            return self.map + tf.cast(self.background_value, tf.complex64)
        return self.map + self.background_value

    def merge_shape(self, shape: Shape, enforce_4fold_symmetry: bool) -> tf.Tensor:
        """Adds a shape onto the canvas.

        Args:
            shape: The shape to rasterize.
        """
        tabula_rasa = _blank_canvas_like(self)
        if isinstance(shape, Polygon):
            tabula_rasa.add_polygon(shape.points)
        elif isinstance(shape, Rectangle):
            tabula_rasa.add_rectangle(
                shape.center, shape.x_width, shape.y_width, shape.rotation_deg
            )
        elif isinstance(shape, Circle):
            tabula_rasa.add_circle(shape.center, shape.radius)
        else:
            raise TypeError("Unsupported shape type {}.".format(type(shape)))

        if enforce_4fold_symmetry:
            image = tabula_rasa.map[..., tf.newaxis]
            image += tf.image.rot90(image, k=1)
            image += tf.image.rot90(image, k=2)
            image += tf.image.flip_left_right(image)
            tabula_rasa.map = image[..., 0]
            tabula_rasa = _apply_threshold(tabula_rasa)

        if self.use_complex:
            tabula_rasa.map = tf.cast(tabula_rasa.map, tf.complex64)
        tabula_rasa.map *= shape.value - self.background_value
        self.merge_with(tabula_rasa)

    def merge_with(self, other: "Canvas") -> None:
        """Merges the canvas with another canvas.

        Args:
            other: The other canvas to merge with.
        """
        if self.use_complex:
            self.map = tf.where(
                tf.abs(other.map) > tf.abs(self.map), other.map, self.map
            )
        else:
            self.map = tf.where(other.map > self.map, other.map, self.map)

    def draw(self):
        """Draws the canvas."""
        figure = plt.figure()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.imshow(
            self.map.numpy(),
            extent=[
                -self.x_width / 2,
                self.x_width / 2,
                -self.y_width / 2,
                self.y_width / 2,
            ],
        )
        plt.colorbar()

    def __add__(self, other):
        self.map += other.map
        return self

    def __sub__(self, other):
        self.map -= other.map
        return self

    def __mul__(self, other):
        self.map *= other.map
        return self

    def __truediv__(self, other):
        self.map /= other.map
        return self

    def add_point(self, p: Tuple[float, float], radius=0.5) -> None:
        """Adds a point to the canvas.

        Args:
            p: The point to add.
            radius: The radius of the point.
        """
        self.map = _add_point(self, p, radius).map

    def add_triangle(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> None:
        """Adds a triangle to the canvas.

        Args:
            p0: The first point of the triangle.
            p1: The second point of the triangle.
            p2: The third point of the triangle.

        If the points are in clockwise order, the triangle will be rasterized
        as a positive shape. If the points are in counter-clockwise order, the
        triangle will be rasterized as a negative shape.
        """
        self.map = _add_triangle(self, p0, p1, p2).map

    def add_polygon(
        self,
        points: CoordType,
        keep_positive: bool = True,
    ) -> None:
        """Adds a polygon to the canvas.

        Args:
            points (List[Tuple[float, float]]): _description_
            keep_positive (bool, optional): keeps the values positive. Defaults to True.
        """
        self.map = _add_polygon(self, points, keep_positive=keep_positive).map

    def add_regular_polygon(
        self,
        center: Tuple[float, float],
        radius: float,
        n: int,
        keep_positive: bool = True,
        apply_threshold: bool = True,
    ) -> None:
        """Adds a regular polygon to the canvas.

        Args:
            center: The center of the polygon.
            radius: The radius of the polygon.
            n: The number of sides of the polygon.
            keep_positive: Keeps the values positive. Defaults to True.
            apply_threshold: Applies threshold from 0 to 1. Defaults to True.
        """
        self.map = _add_regular_polygon(
            self,
            center,
            radius,
            n,
        ).map

    def add_regular_star(
        self,
        center: CoordType,
        radius: ParameterType,
        n: int,
        keep_positive: bool = True,
        apply_threshold: bool = True,
    ) -> None:
        """Addes a regular star to the canvas.

        Args:
            center: The center of the star.
            radius: The radius of the star.
            n: The number of points of the star.
            keep_positive: Keeps the values positive. Defaults to True.
            apply_threshold: Applies threshold from 0 to 1. Defaults to True.
        """
        self.map = _add_regular_star(
            self,
            center,
            radius,
            n,
            keep_positive=keep_positive,
            apply_threshold=apply_threshold,
        ).map

    def add_rectangle(
        self,
        center: CoordType,
        x_width: ParameterType,
        y_width: ParameterType,
        rotation_deg: ParameterType = 0.0,
    ) -> None:
        """Adds a rectangle to the canvas.

        Args:
            center: The center of the rectangle.
            x_width: The width of the rectangle in the x direction.
            y_width: The width of the rectangle in the y direction.
            rotation_deg: The rotation of the rectangle in degrees. Defaults to 0.
        """
        vertices = rectangle_to_vertices(center, x_width, y_width, rotation_deg)
        self.map = _add_polygon(self, vertices).map

    def add_circle(
        self,
        center: CoordType,
        radius: ParameterType,
    ) -> None:
        """Adds a circle to the canvas.

        Args:
            center: The center of the circle.
            radius: The radius of the circle.
            keep_positive: Keeps the values positive. Defaults to True.
            apply_threshold: Applies threshold from 0 to 1. Defaults to True.
        """
        self.map = _add_circle(
            self,
            center,
            radius,
        ).map


def _is_complex(x: ParameterType):
    """Determine if a ParameterType is complex-valued.

    Args:
        x: A ParameterType.
    """
    x = tf.convert_to_tensor(x)
    return x.dtype.is_complex


def _floatt(x: CoordType) -> CoordType:
    """Clean up the input coordinates.

    Args:
        x: The coordinates to clean up.

    Returns:
        The cleaned up coordinates.
    """
    a, b = tf.math.real(x[0]), tf.math.real(x[1])
    return (tf.cast(a, tf.float32), tf.cast(b, tf.float32))


def _blank_canvas_like(canvas) -> Canvas:
    """Generates a blank canvas with the same properties as the input canvas.

    Args:
        canvas: The canvas to copy.

    Returns:
        A blank canvas with the same properties as the input canvas.
    """
    return Canvas(
        x_width=canvas.x_width,
        y_width=canvas.y_width,
        spacing=canvas.spacing,
        background_value=canvas.background_value,
    )


def _add_point(canvas, p: CoordType, radius: ParameterType = 0.5) -> Canvas:
    """Adds a point to the canvas.

    Args:
        canvas: The canvas to add the point to.
        p: The point to add.
        radius: The radius of the point.

    Returns:
        The canvas with the point added.
    """
    p = _floatt(p)
    new_xx = canvas.xx - p[0] / canvas.spacing
    new_yy = canvas.yy + p[1] / canvas.spacing
    canvas.map += tf.exp((-tf.square(new_xx) - tf.square(new_yy)) / (radius**2))
    return canvas


def _add_circle(canvas, center: CoordType, radius: ParameterType) -> Canvas:
    """Adds a circle to the canvas.

    Args:
        canvas: The canvas to add the circle to.
        center: The center of the circle.
        radius: The radius of the circle.

    Returns:
        The canvas with the circle added.
    """
    center = _floatt(center)
    new_xx = canvas.xx - center[0] / canvas.spacing
    new_yy = canvas.yy + center[1] / canvas.spacing
    canvas.map += radius / canvas.spacing - tf.sqrt((new_xx**2) + (new_yy**2))
    return _apply_threshold(canvas)


def _add_line_ramp(
    canvas,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
) -> Canvas:
    """Adds a line ramp to the canvas.

    Args:
        canvas: The canvas to add the line ramp to.
        p0: The first point of the line.
        p1: The second point of the line.
    Returns:
        The canvas with the line ramp added.
    """
    p0, p1 = _floatt(p0), _floatt(p1)
    new_xx = canvas.xx - p0[0] / canvas.spacing
    new_yy = canvas.yy + p0[1] / canvas.spacing
    angle = tf.atan2(p1[1] - p0[1], p1[0] - p0[0])
    canvas.map += -tf.sin(angle) * new_xx - tf.cos(angle) * new_yy
    return canvas


def _line_function(
    canvas,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    invert: bool = False,
) -> Canvas:
    """Generates a line function.

    Args:
        canvas: The canvas to generate the line function on.
        p0: The first point of the line.
        p1: The second point of the line.
        invert: Inverts the line function. Defaults to False.
    Returns:
        The canvas with the line function added.
    """
    p0, p1 = _floatt(p0), _floatt(p1)
    canvas = _blank_canvas_like(canvas)
    canvas = _add_line_ramp(canvas, p0, p1)
    canvas = _apply_threshold(canvas)
    if invert:
        canvas.map = 1 - canvas.map
    return canvas


def _apply_threshold(
    canvas,
    high_threshold: float = 1,
    low_threshold: float = 0,
    offset: float = 0.5,
    norm: Union[float, None] = None,
) -> Canvas:
    """Applies a threshold to the canvas.

    Args:
        canvas: The canvas to apply the threshold to.
        high_threshold: The high threshold. Defaults to 1.
        low_threshold: The low threshold. Defaults to 0.
        offset: The offset to apply to the canvas. Defaults to 0.5.
        norm: The normalization factor. Defaults to None.
    Returns:
        The canvas with the threshold applied.
    """
    canvas.map += offset
    canvas.map = tf.where(canvas.map > high_threshold, high_threshold, canvas.map)
    canvas.map = tf.where(canvas.map < low_threshold, low_threshold, canvas.map)
    if norm is not None:
        canvas.map /= norm
    else:
        canvas.map /= high_threshold - low_threshold
    return canvas


def _equal_t(a: CoordType, b: CoordType) -> bool:
    """Checks if two points are equal.

    Args:
        a: The first point.
        b: The second point.
    Returns:
        True if the points are equal, False otherwise.
    """
    return (a[0] == b[0]) and (a[1] == b[1])


def _add_triangle(
    canvas,
    p0: CoordType,
    p1: CoordType,
    p2: CoordType,
) -> Canvas:
    """Adds a triangle to the canvas.

    Args:
        canvas: The canvas to add the triangle to.
        p0: The first point of the triangle.
        p1: The second point of the triangle.
        p2: The third point of the triangle.
    Returns:
        The canvas with the triangle added.
    """
    p0, p1, p2 = _floatt(p0), _floatt(p1), _floatt(p2)
    if _equal_t(p0, p1) or _equal_t(p1, p2) or _equal_t(p2, p0):
        # not a triangle
        return canvas

    # calculate cross product to determine if triangle is clockwise or counterclockwise
    vector_0 = (p1[1] - p0[1], p1[0] - p0[0])
    vector_1 = (p2[1] - p1[1], p2[0] - p1[0])
    cross_product = vector_0[0] * vector_1[1] - vector_0[1] * vector_1[0]

    if cross_product == 0:
        # not a triangle
        return canvas
    elif cross_product < 0:
        invert = False
        flip_v = 1.0
    else:
        invert = True
        flip_v = -1.0

    l0 = _line_function(canvas, p0, p1, invert)
    l1 = _line_function(canvas, p1, p2, invert)
    l2 = _line_function(canvas, p2, p0, invert)

    canvas.map += (l0 * l1 * l2).map * flip_v
    return canvas


def _is_convex(canvas, points: List[Tuple[float, float]]) -> bool:
    """Checks if a polygon is convex.

    Args:
        canvas: The canvas to check the polygon on.
        points: The points of the polygon.
    Returns:
        True if the polygon is convex, False otherwise.
    """
    chirolities = []
    for i in range(len(points)):
        p0, p1, p2 = (
            points[i],
            points[(i + 1) % len(points)],
            points[(i + 2) % len(points)],
        )
        vector_0 = (p1[1] - p0[1], p1[0] - p0[0])
        vector_1 = (p2[1] - p1[1], p2[0] - p1[0])
        cross_product = vector_0[0] * vector_1[1] - vector_0[1] * vector_1[0]
        if cross_product == 0:
            # not a triangle
            return False
        else:
            chirolities.append(cross_product > 0)
    if all(chirolities) or not any(chirolities):
        return True
    return False


def _add_convex_polygon(canvas, points: List[CoordType]) -> Canvas:
    """Adds a convex polygon to the canvas.

    Args:
        canvas: The canvas to add the polygon to.
        points: The points of the polygon.
    Returns:
        The canvas with the polygon added.
    """
    slate = _blank_canvas_like(canvas)
    slate.map += 1.0
    # check for sign
    vector_0 = (points[1][1] - points[0][1], points[1][0] - points[0][0])
    vector_1 = (points[2][1] - points[1][1], points[2][0] - points[1][0])
    cross_product = vector_0[0] * vector_1[1] - vector_0[1] * vector_1[0]
    if cross_product > 0:
        points = points[::-1]
    for i in range(len(points)):
        line = _line_function(canvas, points[i], points[(i + 1) % len(points)])
        slate *= line
    canvas += slate
    return canvas


def _add_polygon(
    canvas,
    points: List[CoordType],
    keep_positive: bool = True,
    apply_threshold: bool = True,
) -> Canvas:
    """Adds a polygon to the canvas.

    Args:
        canvas: The canvas to add the polygon to.
        points: The points of the polygon.
        keep_positive: Whether to keep the positive values of the canvas.
        apply_threshold: Whether to apply a threshold to the canvas.
    Returns:
        The canvas with the polygon added.
    """
    if _is_convex(canvas, points):
        return _add_convex_polygon(canvas, points)

    for i in range(len(points)):
        canvas = _add_triangle(canvas, (0, 0), points[i], points[(i + 1) % len(points)])
    if keep_positive:
        canvas.map = tf.abs(canvas.map)
    if apply_threshold:
        canvas = _apply_threshold(
            canvas, high_threshold=1.0, low_threshold=0.0, offset=0.0
        )
    return canvas


def rectangle_to_vertices(
    center: Tuple[float, float],
    x_width: float,
    y_width: float,
    rotation_deg: float = 0,
) -> Canvas:
    """Adds a rectangle to the canvas.

    Args:
        canvas: The canvas to add the rectangle to.
        center: The center of the rectangle.
        x_width: The width of the rectangle in the x direction.
        y_width: The width of the rectangle in the y direction.
        rotation_deg: The rotation of the rectangle in degrees.
    Returns:
        The points of the rotated rectangle.
    """
    center = _floatt(center)
    x_width, y_width = float(x_width), float(y_width)
    rotation_rad = -rotation_deg * np.pi / 180
    rotation_matrix = np.array(
        [
            [tf.cos(rotation_rad), -tf.sin(rotation_rad)],
            [tf.sin(rotation_rad), tf.cos(rotation_rad)],
        ]
    )
    translation = np.array([center[0], center[1]])
    points = [
        np.array([x_width / 2, y_width / 2]),
        np.array([-x_width / 2, y_width / 2]),
        np.array([-x_width / 2, -y_width / 2]),
        np.array([x_width / 2, -y_width / 2]),
    ]
    for i in range(len(points)):
        points[i] = rotation_matrix @ points[i] + translation
    return points


def _add_regular_polygon(
    canvas,
    center: Tuple[float, float],
    radius: float,
    n: int,
) -> Canvas:
    """Adds a regular polygon to the canvas.

    Args:
        canvas: The canvas to add the polygon to.
        center: The center of the polygon.
        radius: The radius of the polygon.
        n: The number of sides of the polygon.
    Returns:
        The canvas with the polygon added.
    """
    points = []
    for i in range(n):
        points.append(
            (
                radius * tf.cos(2 * np.pi * i / n) + center[0],
                radius * tf.sin(2 * np.pi * i / n) + center[1],
            )
        )
    return _add_polygon(canvas, points)


def _add_regular_star(
    canvas,
    center: Tuple[float, float],
    radius: float,
    n: int,
) -> Canvas:
    """Adds a regular star to the canvas.

    Args:
        canvas: The canvas to add the star to.
        center: The center of the star.
        radius: The radius of the star.
        n: The number of sides of the star.
    Returns:
        The canvas with the star added.
    """
    order = list(range(n))
    new_order = [(ii * (n // 2)) % n for ii in order]
    points = []
    for i in new_order:
        points.append(
            (
                radius * tf.cos(2 * np.pi * i / n) + center[0],
                radius * tf.sin(2 * np.pi * i / n) - center[1],
            )
        )
    return _add_polygon(canvas, points)
