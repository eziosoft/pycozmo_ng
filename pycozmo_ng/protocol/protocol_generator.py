"""

Cozmo protocol packet encoder code generator.

"""

import os
from collections import defaultdict

from . import protocol_declaration, protocol_utils

__all__ = [
    "ProtocolGenerator",
]


def get_fmt_by_type(t: protocol_declaration.Argument) -> str:
    if isinstance(t, protocol_declaration.FloatArgument):
        fmt = "f"
    elif isinstance(t, protocol_declaration.DoubleArgument):
        fmt = "d"
    elif isinstance(t, protocol_declaration.BoolArgument):
        fmt = "b"
    elif isinstance(t, protocol_declaration.UInt8Argument):
        fmt = "B"
    elif isinstance(t, protocol_declaration.UInt16Argument):
        fmt = "H"
    elif isinstance(t, protocol_declaration.UInt32Argument):
        fmt = "L"
    elif isinstance(t, protocol_declaration.Int8Argument):
        fmt = "b"
    elif isinstance(t, protocol_declaration.Int16Argument):
        fmt = "h"
    elif isinstance(t, protocol_declaration.Int32Argument):
        fmt = "l"
    else:
        raise TypeError(f"Unexpected argument type '{t.__class__.__name__}'")
    return fmt


def get_farray_fmt(argument: protocol_declaration.FArrayArgument) -> str:
    data_fmt = get_fmt_by_type(argument.data_type)
    if not data_fmt:
        raise NotImplementedError(
            f"Unexpected farray data type '{argument.data_type.__class__.__name__}' for '{argument.name}'"
        )
    return data_fmt


def get_varray_fmts(argument: protocol_declaration.VArrayArgument) -> tuple[str, str]:
    length_fmt = get_fmt_by_type(argument.length_type)
    if not length_fmt:
        raise NotImplementedError(
            f"Unexpected varray length type '{argument.length_type.__class__.__name__}' for '{argument.name}'"
        )
    data_fmt = get_fmt_by_type(argument.data_type)
    if not data_fmt:
        raise NotImplementedError(
            f"Unexpected varray data type '{argument.data_type.__class__.__name__}' for '{argument.name}'"
        )
    return length_fmt, data_fmt


def get_string_fmt(argument: protocol_declaration.StringArgument) -> str:
    length_fmt = get_fmt_by_type(argument.length_type)
    if not length_fmt:
        raise NotImplementedError(
            f"Unexpected string length type '{argument.length_type.__class__.__name__}' for '{argument.name}'"
        )
    return length_fmt


def get_enum_fmt(argument: protocol_declaration.EnumArgument) -> str:
    data_fmt = get_fmt_by_type(argument.data_type)
    if not data_fmt:
        raise NotImplementedError(
            f"Unexpected enum data type '{argument.data_type.__class__.__name__}' for '{argument.name}'"
        )
    return data_fmt


def int_to_str(value: int, base: int = 10) -> str:
    if base == 8:
        res = f"0o{value:o}"
    elif base == 10:
        res = f"{value:d}"
    elif base == 16:
        res = f"0x{value:x}"
    else:
        raise ValueError(f"Unexpected base {base}.")
    return res


class ProtocolGenerator:
    def __init__(self, f):
        self.f = f

    def generate_packet_slots(self, struct: protocol_declaration.Struct) -> None:
        for argument in struct.arguments:
            type_hint = argument.type_hint()
            if type_hint:
                type_hint = f"  # {type_hint}"
            else:
                type_hint = ""
            self.f.write(f'        "_{argument.name}",{type_hint}\n')

    def generate_farray_validation(
        self, argument: protocol_declaration.FArrayArgument
    ) -> None:
        if isinstance(argument.data_type, (protocol_declaration.FloatArgument, protocol_declaration.DoubleArgument)):
            element_validation = (
                "lambda name, value_inner: validate_float(name, value_inner)"
            )
        elif isinstance(argument.data_type, protocol_declaration.BoolArgument):
            element_validation = (
                "lambda name, value_inner: validate_bool(name, value_inner)"
            )
        elif isinstance(argument.data_type, protocol_declaration.UInt8Argument):
            element_validation = (
                "lambda name, value_inner: validate_integer(name, value_inner, 0, 255)"
            )
        elif isinstance(argument.data_type, protocol_declaration.UInt16Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, 0, 65535)"
        elif isinstance(argument.data_type, protocol_declaration.UInt32Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, 0, 4294967295)"
        elif isinstance(argument.data_type, protocol_declaration.Int8Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, -128, 127)"
        elif isinstance(argument.data_type, protocol_declaration.Int16Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, -32768, 32767)"
        elif isinstance(argument.data_type, protocol_declaration.Int32Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, -2147483648, 2147483647)"
        elif isinstance(argument.data_type, protocol_declaration.Struct):
            element_validation = f"lambda name, value_inner: validate_object(name, value_inner, {argument.data_type.name})"
        else:
            raise NotImplementedError(
                f"Unexpected farray data type '{argument.data_type.__class__.__name__}' for '{argument.name}'"
            )

        self.f.write(
            f'validate_farray(\n            "{argument.name}", value, {argument.length}, {element_validation})\n'
        )

    def generate_varray_validation(
        self, argument: protocol_declaration.VArrayArgument
    ) -> None:
        if isinstance(argument.length_type, protocol_declaration.UInt8Argument):
            maximum_length = 255
        elif isinstance(argument.length_type, protocol_declaration.UInt16Argument):
            maximum_length = 65535
        else:
            raise NotImplementedError(
                f"Unexpected varray length type '{argument.length_type.__class__.__name__}' for '{argument.name}'"
            )

        if isinstance(argument.data_type, (protocol_declaration.FloatArgument, protocol_declaration.DoubleArgument)):
            element_validation = (
                "lambda name, value_inner: validate_float(name, value_inner)"
            )
        elif isinstance(argument.data_type, protocol_declaration.BoolArgument):
            element_validation = (
                "lambda name, value_inner: validate_bool(name, value_inner)"
            )
        elif isinstance(argument.data_type, protocol_declaration.UInt8Argument):
            element_validation = (
                "lambda name, value_inner: validate_integer(name, value_inner, 0, 255)"
            )
        elif isinstance(argument.data_type, protocol_declaration.UInt16Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, 0, 65535)"
        elif isinstance(argument.data_type, protocol_declaration.UInt32Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, 0, 4294967295)"
        elif isinstance(argument.data_type, protocol_declaration.Int8Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, -128, 127)"
        elif isinstance(argument.data_type, protocol_declaration.Int16Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, -32768, 32767)"
        elif isinstance(argument.data_type, protocol_declaration.Int32Argument):
            element_validation = "lambda name, value_inner: validate_integer(name, value_inner, -2147483648, 2147483647)"
        else:
            raise NotImplementedError(
                f"Unexpected varray data type '{argument.data_type.__class__.__name__}' for '{argument.name}'"
            )

        self.f.write(
            f'validate_varray(\n            "{argument.name}", value, {maximum_length}, {element_validation})\n'
        )

    def generate_string_validation(
        self, argument: protocol_declaration.StringArgument
    ) -> None:
        if isinstance(argument.length_type, protocol_declaration.UInt8Argument):
            maximum_length = 255
        elif isinstance(argument.length_type, protocol_declaration.UInt16Argument):
            maximum_length = 65535
        else:
            raise NotImplementedError(
                f"Unexpected string length type '{argument.length_type.__class__.__name__}' for '{argument.name}'"
            )

        self.f.write(f'validate_string("{argument.name}", value, {maximum_length})\n')

    def generate_enum_validation(
        self, argument: protocol_declaration.EnumArgument
    ) -> None:
        if isinstance(argument.data_type, protocol_declaration.UInt8Argument):
            self.f.write(f'validate_integer("{argument.name}", value.value, 0, 255)\n')
        elif isinstance(argument.data_type, protocol_declaration.UInt16Argument):
            self.f.write(
                f'validate_integer("{argument.name}", value.value, 0, 65535)\n'
            )
        elif isinstance(argument.data_type, protocol_declaration.UInt32Argument):
            self.f.write(
                f'validate_integer("{argument.name}", value.value, 0, 4294967295)\n'
            )
        elif isinstance(argument.data_type, protocol_declaration.Int8Argument):
            self.f.write(
                f'validate_integer("{argument.name}", value.value, -128, 127)\n'
            )
        elif isinstance(argument.data_type, protocol_declaration.Int16Argument):
            self.f.write(
                f'validate_integer("{argument.name}", value.value, -32768, 32767)\n'
            )
        elif isinstance(argument.data_type, protocol_declaration.Int32Argument):
            self.f.write(
                f'validate_integer("{argument.name}", value.value, -2147483648, 2147483647)\n'
            )
        else:
            raise NotImplementedError(
                f"Unexpected enum data type '{argument.data_type.__class__.__name__}' for '{argument.name}'"
            )

    def generate_argument_methods(self, struct: protocol_declaration.Struct) -> None:
        for argument in struct.arguments:
            if isinstance(argument, protocol_declaration.EnumArgument):
                self.f.write(
                    rf"""
    @property
    def {argument.name}(self) -> {argument.enum_type.name}:
        return self._{argument.name}

    @{argument.name}.setter
    def {argument.name}(self, value: {argument.enum_type.name}) -> None:
        self._{argument.name} = value
        """
                )
                self.generate_enum_validation(argument)
            else:
                self.f.write(
                    rf"""
    @property
    def {argument.name}(self):
        return self._{argument.name}

    @{argument.name}.setter
    def {argument.name}(self, value):
        self._{argument.name} = """
                )
                if isinstance(argument, (protocol_declaration.FloatArgument, protocol_declaration.DoubleArgument)):
                    self.f.write(f'validate_float("{argument.name}", value)\n')
                elif isinstance(argument, protocol_declaration.BoolArgument):
                    self.f.write(f'validate_bool("{argument.name}", value)\n')
                elif isinstance(argument, protocol_declaration.UInt8Argument):
                    self.f.write(
                        f'validate_integer("{argument.name}", value, 0, 255)\n'
                    )
                elif isinstance(argument, protocol_declaration.UInt16Argument):
                    self.f.write(
                        f'validate_integer("{argument.name}", value, 0, 65535)\n'
                    )
                elif isinstance(argument, protocol_declaration.UInt32Argument):
                    self.f.write(
                        f'validate_integer("{argument.name}", value, 0, 4294967295)\n'
                    )
                elif isinstance(argument, protocol_declaration.Int8Argument):
                    self.f.write(
                        f'validate_integer("{argument.name}", value, -128, 127)\n'
                    )
                elif isinstance(argument, protocol_declaration.Int16Argument):
                    self.f.write(
                        f'validate_integer("{argument.name}", value, -32768, 32767)\n'
                    )
                elif isinstance(argument, protocol_declaration.Int32Argument):
                    self.f.write(
                        f'validate_integer("{argument.name}", value, -2147483648, 2147483647)\n'
                    )
                elif isinstance(argument, protocol_declaration.FArrayArgument):
                    self.generate_farray_validation(argument)
                elif isinstance(argument, protocol_declaration.VArrayArgument):
                    self.generate_varray_validation(argument)
                elif isinstance(argument, protocol_declaration.StringArgument):
                    self.generate_string_validation(argument)
                else:
                    raise NotImplementedError(
                        f"Unexpected argument type '{argument.__class__.__name__}' for '{argument.name}'"
                    )

    def generate_len_method(self, struct: protocol_declaration.Struct) -> None:
        self.f.write("\n    def __len__(self):\n")
        if not struct.arguments:
            self.f.write("        return 0\n")
        else:
            self.f.write("        return \\\n")
            statements = []
            for argument in struct.arguments:
                if isinstance(argument, protocol_declaration.FloatArgument):
                    statements.append(str(protocol_utils.get_size("f")))
                elif isinstance(argument, protocol_declaration.DoubleArgument):
                    statements.append(str(protocol_utils.get_size("d")))
                elif isinstance(argument, protocol_declaration.BoolArgument):
                    statements.append(str(protocol_utils.get_size("b")))
                elif isinstance(argument, protocol_declaration.UInt8Argument):
                    statements.append(str(protocol_utils.get_size("B")))
                elif isinstance(argument, protocol_declaration.UInt16Argument):
                    statements.append(str(protocol_utils.get_size("H")))
                elif isinstance(argument, protocol_declaration.UInt32Argument):
                    statements.append(str(protocol_utils.get_size("L")))
                elif isinstance(argument, protocol_declaration.Int8Argument):
                    statements.append(str(protocol_utils.get_size("b")))
                elif isinstance(argument, protocol_declaration.Int16Argument):
                    statements.append(str(protocol_utils.get_size("h")))
                elif isinstance(argument, protocol_declaration.Int32Argument):
                    statements.append(str(protocol_utils.get_size("l")))
                elif isinstance(argument, protocol_declaration.FArrayArgument):
                    if isinstance(argument.data_type, protocol_declaration.Struct):
                        statements.append(
                            f"get_object_farray_size(self._{argument.name}, {argument.length})"
                        )
                    else:
                        data_fmt = get_farray_fmt(argument)
                        statements.append(
                            str(
                                protocol_utils.get_farray_size(
                                    data_fmt, argument.length
                                )
                            )
                        )
                elif isinstance(argument, protocol_declaration.VArrayArgument):
                    length_fmt, data_fmt = get_varray_fmts(argument)
                    statements.append(
                        f"get_varray_size(self._{argument.name}, '{length_fmt}', '{data_fmt}')"
                    )
                elif isinstance(argument, protocol_declaration.StringArgument):
                    length_fmt = get_string_fmt(argument)
                    statements.append(
                        f"get_string_size(self._{argument.name}, '{length_fmt}')"
                    )
                elif isinstance(argument, protocol_declaration.EnumArgument):
                    data_fmt = get_enum_fmt(argument)
                    statements.append(str(protocol_utils.get_size(data_fmt)))
                else:
                    raise NotImplementedError(
                        f"Unexpected argument type '{argument.__class__.__name__}' for '{argument.name}'"
                    )
            self.f.write("            ")
            self.f.write(" + \\\n            ".join(statements))
            self.f.write("\n")

    def generate_repr_method(self, struct: protocol_declaration.Struct) -> None:
        self.f.write("\n    def __repr__(self):\n")
        if struct.arguments:
            argument_strs = []
            arguments = ["type=type(self).__name__"]
            if struct.arguments:
                for argument in struct.arguments:
                    argument_strs.append(f'"{argument.name}={{{argument.name}}}')
                    arguments.append(f"{argument.name}=self._{argument.name}")
            self.f.write('        return "{type}(" \\\n               ')
            self.f.write(
                '{argument_strs})".format(\n                {arguments})\n'.format(
                    argument_strs=', " \\\n               '.join(argument_strs),
                    arguments=",\n                ".join(arguments),
                )
            )
        else:
            self.f.write('        return "{type}()".format(type=type(self).__name__)\n')

    def generate_argument_defaults(self, struct: protocol_declaration.Struct) -> None:
        for argument in struct.arguments:
            if isinstance(argument.default, str):
                self.f.write(
                    f",\n                 {argument.name}='{argument.default}'"
                )
            else:
                self.f.write(f",\n                 {argument.name}={argument.default}")

    def generate_argument_assignments(
        self, struct: protocol_declaration.Struct
    ) -> None:
        if struct.arguments:
            for argument in struct.arguments:
                if argument.description:
                    self.f.write(f"        # {argument.description}\n")
                if isinstance(argument, protocol_declaration.EnumArgument):
                    self.f.write(
                        f"        self.{argument.name} = {argument.enum_type.name}({argument.name})\n"
                    )
                else:
                    self.f.write(f"        self.{argument.name} = {argument.name}\n")
        else:
            self.f.write("        pass\n")

    def generate_packet_argument_assignments(
        self, packet: protocol_declaration.Packet
    ) -> None:
        packet_id = f"0x{packet.id:02x}" if packet.id is not None else None
        self.f.write(
            f"        super().__init__({packet.type}, packet_id={packet_id})\n"
        )
        self.generate_argument_assignments(packet)

    def generate_packet_encoding(self, struct: protocol_declaration.Struct) -> None:
        self.f.write(r"""
    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
""")
        if struct.arguments:
            for argument in struct.arguments:
                if isinstance(argument, protocol_declaration.FloatArgument):
                    self.f.write(f'        writer.write(self._{argument.name}, "f")\n')
                elif isinstance(argument, protocol_declaration.DoubleArgument):
                    self.f.write(f'        writer.write(self._{argument.name}, "d")\n')
                elif isinstance(argument, protocol_declaration.BoolArgument):
                    self.f.write(
                        f'        writer.write(int(self._{argument.name}), "b")\n'
                    )
                elif isinstance(argument, protocol_declaration.UInt8Argument):
                    self.f.write(f'        writer.write(self._{argument.name}, "B")\n')
                elif isinstance(argument, protocol_declaration.UInt16Argument):
                    self.f.write(f'        writer.write(self._{argument.name}, "H")\n')
                elif isinstance(argument, protocol_declaration.UInt32Argument):
                    self.f.write(f'        writer.write(self._{argument.name}, "L")\n')
                elif isinstance(argument, protocol_declaration.Int8Argument):
                    self.f.write(f'        writer.write(self._{argument.name}, "b")\n')
                elif isinstance(argument, protocol_declaration.Int16Argument):
                    self.f.write(f'        writer.write(self._{argument.name}, "h")\n')
                elif isinstance(argument, protocol_declaration.Int32Argument):
                    self.f.write(f'        writer.write(self._{argument.name}, "l")\n')
                elif isinstance(argument, protocol_declaration.FArrayArgument):
                    if isinstance(argument.data_type, protocol_declaration.Struct):
                        self.f.write(
                            f"        writer.write_object_farray(self._{argument.name}, {argument.length})\n"
                        )
                    else:
                        data_fmt = get_farray_fmt(argument)
                        self.f.write(
                            f'        writer.write_farray(self._{argument.name}, "{data_fmt}", {argument.length})\n'
                        )
                elif isinstance(argument, protocol_declaration.VArrayArgument):
                    length_fmt, data_fmt = get_varray_fmts(argument)
                    self.f.write(
                        f'        writer.write_varray(self._{argument.name}, "{data_fmt}", "{length_fmt}")\n'
                    )
                elif isinstance(argument, protocol_declaration.StringArgument):
                    length_fmt = get_string_fmt(argument)
                    self.f.write(
                        f'        writer.write_string(self._{argument.name}, "{length_fmt}")\n'
                    )
                elif isinstance(argument, protocol_declaration.EnumArgument):
                    data_fmt = get_enum_fmt(argument)
                    self.f.write(
                        f'        writer.write(self._{argument.name}.value, "{data_fmt}")\n'
                    )
                else:
                    raise NotImplementedError(
                        f"Unexpected argument type '{argument.__class__.__name__}' for '{argument.name}'"
                    )
        else:
            self.f.write("        pass\n")

    def generate_packet_decoding(self, struct: protocol_declaration.Struct) -> None:
        self.f.write(r"""
    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
""")
        if struct.arguments:
            for argument in struct.arguments:
                if isinstance(argument, protocol_declaration.FloatArgument):
                    self.f.write(f'        {argument.name} = reader.read("f")\n')
                elif isinstance(argument, protocol_declaration.DoubleArgument):
                    self.f.write(f'        {argument.name} = reader.read("d")\n')
                elif isinstance(argument, protocol_declaration.BoolArgument):
                    self.f.write(f'        {argument.name} = bool(reader.read("b"))\n')
                elif isinstance(argument, protocol_declaration.UInt8Argument):
                    self.f.write(f'        {argument.name} = reader.read("B")\n')
                elif isinstance(argument, protocol_declaration.UInt16Argument):
                    self.f.write(f'        {argument.name} = reader.read("H")\n')
                elif isinstance(argument, protocol_declaration.UInt32Argument):
                    self.f.write(f'        {argument.name} = reader.read("L")\n')
                elif isinstance(argument, protocol_declaration.Int8Argument):
                    self.f.write(f'        {argument.name} = reader.read("b")\n')
                elif isinstance(argument, protocol_declaration.Int16Argument):
                    self.f.write(f'        {argument.name} = reader.read("h")\n')
                elif isinstance(argument, protocol_declaration.Int32Argument):
                    self.f.write(f'        {argument.name} = reader.read("l")\n')
                elif isinstance(argument, protocol_declaration.FArrayArgument):
                    if isinstance(argument.data_type, protocol_declaration.Struct):
                        self.f.write(
                            f"        {argument.name} = reader.read_object_farray({argument.data_type.name}.from_reader, {argument.length})\n"
                        )
                    else:
                        data_fmt = get_farray_fmt(argument)
                        self.f.write(
                            f'        {argument.name} = reader.read_farray("{data_fmt}", {argument.length})\n'
                        )
                elif isinstance(argument, protocol_declaration.VArrayArgument):
                    length_fmt, data_fmt = get_varray_fmts(argument)
                    self.f.write(
                        f'        {argument.name} = reader.read_varray("{data_fmt}", "{length_fmt}")\n'
                    )
                elif isinstance(argument, protocol_declaration.StringArgument):
                    length_fmt = get_string_fmt(argument)
                    self.f.write(
                        f'        {argument.name} = reader.read_string("{length_fmt}")\n'
                    )
                elif isinstance(argument, protocol_declaration.EnumArgument):
                    data_fmt = get_enum_fmt(argument)
                    self.f.write(
                        f'        {argument.name} = reader.read("{data_fmt}")\n'
                    )
                else:
                    raise NotImplementedError(
                        f"Unexpected argument type '{argument.__class__.__name__}' for '{argument.name}'"
                    )
        else:
            self.f.write("        del reader\n")
        self.f.write("        return cls(\n")
        arguments = []
        for argument in struct.arguments:
            arguments.append(f"{argument.name}={argument.name}")
        self.f.write("            ")
        self.f.write(",\n            ".join(arguments))
        self.f.write(")\n")

    def generate_enum(self, enum_type: protocol_declaration.Enum) -> None:

        self.f.write(
            rf"""

class {enum_type.name}(enum.Enum):
"""
        )
        if enum_type.description:
            self.f.write(f'    """ {enum_type.description} """\n')
        for member in enum_type.members:
            if member.description:
                self.f.write(f"    # {member.description}\n")
            self.f.write(
                f"    {member.name} = {int_to_str(member.value, enum_type.base)}\n"
            )

    def generate_struct(self, struct: protocol_declaration.Struct) -> None:
        self.f.write(
            rf"""

class {struct.name}(Struct):
"""
        )
        if struct.description:
            self.f.write(f'    """ {struct.description} """\n')
        self.f.write("\n    __slots__ = (\n")
        self.generate_packet_slots(struct)
        self.f.write("    )\n\n    def __init__(self")
        self.generate_argument_defaults(struct)
        self.f.write("):\n")
        self.generate_argument_assignments(struct)
        self.generate_argument_methods(struct)
        self.generate_len_method(struct)
        self.generate_repr_method(struct)
        self.generate_packet_encoding(struct)
        self.generate_packet_decoding(struct)

    def generate_packet(self, packet: protocol_declaration.Packet) -> None:
        self.f.write(
            rf"""

class {packet.name}(Packet):
"""
        )
        if packet.description:
            self.f.write(f'    """ {packet.description} """\n')
        self.f.write("\n    __slots__ = (\n")
        self.generate_packet_slots(packet)
        self.f.write("    )\n\n    def __init__(self")
        self.generate_argument_defaults(packet)
        self.f.write("):\n")
        self.generate_packet_argument_assignments(packet)
        self.generate_argument_methods(packet)
        self.generate_len_method(packet)
        self.generate_repr_method(packet)
        self.generate_packet_encoding(packet)
        self.generate_packet_decoding(packet)

    def generate_id_map(self) -> None:
        packet_map = {}
        for pkt in protocol_declaration.PROTOCOL.packets:
            if pkt.id:
                packet_map[pkt.id] = pkt.name

        self.f.write("\n\nPACKETS_BY_ID = {\n")
        for k, v in sorted(packet_map.items()):
            self.f.write(f"    0x{k:02x}: {v},  # {k}\n")
        self.f.write("}\n")

    def generate_group_map(self) -> None:
        packet_map = defaultdict(set)
        for pkt in protocol_declaration.PROTOCOL.packets:
            if pkt.group and pkt.id:
                packet_map[pkt.group].add(pkt)

        self.f.write("\n\nPACKETS_BY_GROUP = {\n")
        for group, pkt_set in sorted(packet_map.items()):
            self.f.write(f'    "{group}": {{\n')
            for pkt in sorted(pkt_set, key=lambda pkt2: pkt2.id or 0):
                self.f.write(f"        0x{pkt.id:02x},  # {pkt.name}\n")
            self.f.write("    },\n")
        self.f.write("}\n")

    def generate(self) -> None:
        header = rf'''"""

Cozmo protocol packet encoder classes, based on protocol version {protocol_declaration.FIRMWARE_VERSION}.

Generated from {os.path.basename(protocol_declaration.__file__)} by {os.path.basename(__file__)} .

Do not modify.

"""

import enum

from .protocol_ast import PacketType
from .protocol_base import Struct, Packet
from .protocol_utils import \
    validate_float, validate_bool, validate_integer, validate_object, \
    validate_farray, validate_varray, validate_string, \
    get_varray_size, get_string_size, get_object_farray_size, \
    BinaryReader, BinaryWriter
'''
        self.f.write(header)

        for enum in protocol_declaration.PROTOCOL.enums:
            self.generate_enum(enum)

        for struct in protocol_declaration.PROTOCOL.structs:
            self.generate_struct(struct)

        for packet in protocol_declaration.PROTOCOL.packets:
            self.generate_packet(packet)

        self.generate_id_map()
        self.generate_group_map()
