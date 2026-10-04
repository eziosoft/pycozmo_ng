"""

Helper routines for working with colors and lights.

"""

from pycozmo_ng.protocol import protocol_encoder

__all__ = [
    "Color",
    "blue",
    "blue_light",
    "green",
    "green_light",
    "off",
    "off_light",
    "red",
    "red_light",
    "white",
    "white_light",
]


LED_ENC_BLUE = 0x001F
LED_ENC_GREEN = 0x03E0
LED_ENC_RED = 0x7C00
LED_ENC_IR = 0x8000

LED_ENC_BLUE_SHIFT = 0
LED_ENC_GREEN_SHIFT = 5
LED_ENC_RED_SHIFT = 10
LED_ENC_IR_SHIFT = 15


class Color:
    """
    A Color to be used with a Light.

    Either int_color or rgb may be used to specify the actual color.
    Any alpha components (from int_color) are ignored - all colors are fully opaque.

    Args:
        int_color (int): A 32 bit value holding the binary RGBA value.
        rgb (tuple): A tuple holding the integer values from 0-255 for (red, green, blue)
        name (str): A name to assign to this color
    """

    def __init__(
        self,
        int_color: int | None = None,
        rgb: tuple[int, int, int] | None = None,
        name: str | None = None,
    ) -> None:
        self.name = name
        if int_color is not None:
            self._int_color = int(int_color) | 0xFF
        elif rgb is not None:
            self._int_color = (
                (int(rgb[0]) << 24) | (int(rgb[1]) << 16) | (int(rgb[2]) << 8) | 0xFF
            )
        else:
            self._int_color = 0

    @property
    def int_color(self) -> int:
        return self._int_color

    def to_int16(self) -> int:
        r = ((self._int_color & 0xFF000000) >> 24) * 31 // 255
        g = ((self._int_color & 0x00FF0000) >> 16) * 31 // 255
        b = ((self._int_color & 0x0000FF00) >> 8) * 31 // 255
        value = (
            (r << LED_ENC_RED_SHIFT)
            | (g << LED_ENC_GREEN_SHIFT)
            | (b << LED_ENC_BLUE_SHIFT)
        )
        return value

    @classmethod
    def from_int16(cls, value: int) -> "Color":
        r = (value & LED_ENC_RED) >> LED_ENC_RED_SHIFT
        g = (value & LED_ENC_GREEN) >> LED_ENC_GREEN_SHIFT
        b = (value & LED_ENC_BLUE) >> LED_ENC_BLUE_SHIFT
        rgb = (r * 255 // 31, g * 255 // 31, b * 255 // 31)
        obj = cls(rgb=rgb)
        return obj

    def __repr__(self):
        return f"Color(name={self.name}, int_color=0x{self._int_color:08x})"


#: Green color.
green = Color(name="green", int_color=0x00FF00FF)
#: Red color.
red = Color(name="red", int_color=0xFF0000FF)
#: BLue color.
blue = Color(name="blue", int_color=0x0000FFFF)
#: White color.
white = Color(name="white", int_color=0xFFFFFFFF)  # Does not work well with cubes?
#: Off/no color.
off = Color(name="off")

#: Green light.
green_light = protocol_encoder.LightState(
    on_color=green.to_int16(), off_color=green.to_int16()
)
#: Red light.
red_light = protocol_encoder.LightState(
    on_color=red.to_int16(), off_color=red.to_int16()
)
#: Blue light.
blue_light = protocol_encoder.LightState(
    on_color=blue.to_int16(), off_color=blue.to_int16()
)
#: White light.
white_light = protocol_encoder.LightState(
    on_color=white.to_int16(), off_color=white.to_int16()
)
#: Off/no light.
off_light = protocol_encoder.LightState(
    on_color=off.to_int16(), off_color=off.to_int16()
)
