import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import os
import cv2
from PIL import Image


class LoadedObjFile:
    """The loaded / parsed contents of a 3D Wavefront OBJ file.

    This is the intermediary step between the file on the disk, and a renderable
    3D object. It supports the subset of the OBJ file that was used in the
    Cozmo and Cube assets, and does not attempt to exhaustively support every
    possible setting.

    Args:
        filename (str): The filename of the OBJ file to load.
    """
    def __init__(self, filename):
        # list: The vertices (each vertex stored as list of 3 floats).
        self.vertices = []
        # list: The vertex normals (each normal stored as list of 3 floats).
        self.normals = []
        # list: The texture coordinates (each coordinate stored as list of 2 floats).
        self.tex_coords = []
        # dict: The faces for each mesh, indexed by mesh name.
        self.mesh_faces = {}

        # dict: A dictionary mapping named MTL attributes to values.
        self.mtl = None

        group_name = None
        material = None

        filepath = os.path.join(os.path.dirname(__file__), 'assets', filename)

        with open(filepath, 'r') as file_data:
            for line in file_data:
                if line.startswith('#'):
                    # ignore comments in the file
                    continue

                values = line.split()
                if not values:
                    # ignore empty lines
                    continue

                if values[0] == 'v':
                    # vertex position
                    v = list(map(float, values[1:4]))
                    self.vertices.append(v)
                elif values[0] == 'vn':
                    # vertex normal
                    v = list(map(float, values[1:4]))
                    self.normals.append(v)
                elif values[0] == 'vt':
                    # texture coordinate
                    self.tex_coords.append(list(map(float, values[1:3])))
                elif values[0] in ('usemtl', 'usemat'):
                    # material
                    material = values[1]
                elif values[0] == 'mtllib':
                    # material library (a filename)
                    self.mtl = self._load_mtl_file(values[1])
                elif values[0] == 'f':
                    # A face made up of 3 or 4 vertices - e.g. `f v1 v2 v3` or `f v1 v2 v3 v4`
                    # where each vertex definition is multiple indexes seperated by
                    # slashes and can follow the following formats:
                    # position_index
                    # position_index/tex_coord_index
                    # position_index/tex_coord_index/normal_index
                    # position_index//normal_index

                    positions = []
                    tex_coords = []
                    normals = []

                    for vertex in values[1:]:
                        vertex_components = vertex.split('/')

                        positions.append(int(vertex_components[0]))

                        # There's only a texture coordinate if there's at least 2 entries and the 2nd entry is non-zero length
                        if len(vertex_components) >= 2 and len(vertex_components[1]) > 0:
                            tex_coords.append(int(vertex_components[1]))
                        else:
                            # OBJ file indexing starts at 1, so use 0 to indicate no entry
                            tex_coords.append(0)

                        # There's only a normal if there's at least 2 entries and the 2nd entry is non-zero length
                        if len(vertex_components) >= 3 and len(vertex_components[2]) > 0:
                            normals.append(int(vertex_components[2]))
                        else:
                            # OBJ file indexing starts at 1, so use 0 to indicate no entry
                            normals.append(0)

                    try:
                        mesh_face = self.mesh_faces[group_name]
                    except KeyError:
                        # Create a new mesh group
                        self.mesh_faces[group_name] = []
                        mesh_face = self.mesh_faces[group_name]

                    mesh_face.append((positions, normals, tex_coords, material))
                elif values[0] == 'o':
                    # object name - ignore
                    pass
                elif values[0] == 'g':
                    # group name (for a sub-mesh)
                    group_name = values[1]
                elif values[0] == 's':
                    # smooth shading (1..20, and 'off') - ignore
                    pass
                else:
                    print(f"LoadedObjFile Ignoring unhandled type '{values[0]}' in line {line.strip()}")

    def _load_mtl_file(self, filename):
        """Load a .mtl material file, and return the contents as a dictionary.

        Supports the subset of MTL required for the Cozmo 3D viewer assets.

        Args:
            filename (str): The filename of the file to load.

        Returns:
            dict: A dictionary mapping named MTL attributes to values.
        """
        contents = {}
        current_mtl = None

        filepath = os.path.join(os.path.dirname(__file__), 'assets', filename)

        with open(filepath, 'r') as file_data:
            for line in file_data:
                if line.startswith('#'):
                    # ignore comments in the file
                    continue
                values = line.split()
                if not values:
                    # ignore empty lines
                    continue
                attribute_name = values[0]
                if attribute_name == 'newmtl':
                    # Create a new empty material
                    current_mtl = contents[values[1]] = {}
                elif current_mtl is None:
                    raise ValueError("mtl file must start with newmtl statement")
                elif attribute_name == 'map_Kd':
                    # Diffuse texture map - load the image into memory
                    image_name = values[1]
                    image_filepath = os.path.join(os.path.dirname(__file__), 'assets', image_name)

                    # Load image with PIL and convert to RGBA
                    with Image.open(image_filepath) as image:
                        image_width, image_height = image.size
                        image = image.convert("RGBA").tobytes("raw", "RGBA")

                    # Bind the image as a texture that can be used for rendering
                    texture_id = glGenTextures(1)
                    current_mtl['texture_Kd'] = texture_id

                    glBindTexture(GL_TEXTURE_2D, texture_id)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height,
                                 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
                else:
                    # Store the values for this attribute as a list of float values
                    current_mtl[attribute_name] = list(map(float, values[1:]))

        # File loaded successfully - return the contents
        return contents


class World3DViewer:
    """
    3D visualization of Cozmo's world using OpenGL
    """
    # Camera height above the floor in millimetres (Cozmo hardware spec).
    # Used when projecting detections from camera-frame to world-frame.
    CAMERA_HEIGHT_MM = 38.0  # (COZMO-28, user-measured)

    def __init__(self, window_size=(800, 600)):
        """
        Initialize 3D viewer
        :param window_size: (width, height) of the window
        """
        pygame.init()
        self.window_size = window_size
        self.display = pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Cozmo 3D World Viewer")

        # Setup OpenGL perspective
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        # Enable lighting for 3D models
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)  # Add second light
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Set up main light (front)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 200, -500, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])

        # Set up secondary light (back)
        glLightfv(GL_LIGHT1, GL_POSITION, [0, 200, 500, 1])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.6, 0.6, 0.6, 1])

        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (window_size[0] / window_size[1]), 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)

        # Camera position and orientation - side view at 45 degrees up
        # Position camera to the side and above, looking at the robot
        self.camera_pos = [200, -100, -200]  # X, Y, Z in mm - to the side and up
        self.camera_rot = [45, 135, 0]  # Pitch, Yaw, Roll - 45° up, facing the robot

        # Store detected cubes
        self.cubes = {}  # {cube_id: cube_data}

        # Store textures for cube markers
        self.textures = {}  # {cube_id: {face_type: texture_id}}

        # Load 3D cube models (OBJ files)
        self.cube_models = {}  # {cube_id: pywavefront.Wavefront}
        self._load_cube_models()

        # Colors
        self.grid_color = (0.3, 0.3, 0.3)
        self.cube_color = (0.0, 0.8, 1.0)
        self.axis_colors = {
            'x': (1.0, 0.0, 0.0),
            'y': (0.0, 1.0, 0.0),
            'z': (0.0, 0.0, 1.0)
        }

    def _load_cube_models(self):
        """Load OBJ models for the three cubes"""
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')

        if not os.path.exists(assets_dir):
            print(f"Warning: Assets directory not found at {assets_dir}")
            return

        obj_file = "cube.obj"
        obj_path = os.path.join(assets_dir, obj_file)

        if not os.path.exists(obj_path):
            print(f"OBJ file not found: {obj_path}")
            return

        # Read the base OBJ file
        with open(obj_path, 'r') as f:
            obj_content = f.read()

        # Try to load models for cube1, cube2, cube3
        for cube_id in range(3):
            mtl_file = f"cube{cube_id + 1}.mtl"
            texture_file = f"cube{cube_id + 1}.jpg"

            # Check if MTL and texture files exist
            mtl_path = os.path.join(assets_dir, mtl_file)
            texture_path = os.path.join(assets_dir, texture_file)

            if not os.path.exists(mtl_path) or not os.path.exists(texture_path):
                print(f"MTL or texture file not found for cube {cube_id + 1}")
                continue

            try:
                # Create a temporary OBJ file with the correct MTL reference
                temp_obj_content = obj_content.replace('mtllib cube1.mtl', f'mtllib {mtl_file}')
                temp_obj_path = os.path.join(assets_dir, f'cube{cube_id + 1}_temp.obj')

                with open(temp_obj_path, 'w') as f:
                    f.write(temp_obj_content)

                # Load the OBJ file with custom loader
                scene = LoadedObjFile(temp_obj_path)

                self.cube_models[cube_id] = scene
                print(f"Loaded 3D model for cube {cube_id + 1} with texture")

                # Clean up temporary file
                os.remove(temp_obj_path)

            except Exception as e:
                print(f"Failed to load model for cube {cube_id + 1}: {e}")

    def _load_texture(self, filepath):
        """Load an image file and create an OpenGL texture"""
        try:
            import cv2

            # Load image
            img = cv2.imread(filepath)
            if img is None:
                return None

            # Convert BGR to RGB and flip vertically for OpenGL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.flip(img_rgb, 0)

            # Create OpenGL texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)

            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

            # Upload texture data
            height, width = img_rgb.shape[:2]
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_rgb)

            return texture_id
        except Exception as e:
            print(f"Failed to load texture {filepath}: {e}")
            return None

    def update_cube(self, cube_id, position, rotation_matrix, cube_size=44, marker_images=None):
        """
        Update or add a cube to the world
        :param cube_id: Unique identifier for the cube
        :param position: [x, y, z] position in mm (OpenCV camera coordinates)
        :param rotation_matrix: 3x3 rotation matrix (OpenCV camera coordinates)
        :param cube_size: Size of the cube in mm
        :param marker_images: Dict of {face_type: image_array} for texturing faces (BGR format from OpenCV)
        """
        # Convert from OpenCV camera coordinates to OpenGL visualization coordinates
        # OpenCV: X right, Y down, Z forward
        # OpenGL (visualization): X right, Y up, Z forward

        # Ensure rotation_matrix is float32
        rotation_matrix = rotation_matrix.astype(np.float32)

        # Transform position: negate X and Y to correct mirror reflection, add camera height
        position_gl = np.array(
            [-position[0], -position[1] + self.CAMERA_HEIGHT_MM, position[2]],
            dtype=np.float32,
        )

        # Transform rotation using similarity transformation: R_gl = T @ R_cv @ T^T
        # where T mirrors over the YZ plane (flips X)
        T = np.array([
            [-1,  0, 0],
            [0, -1, 0],
            [0,  0, 1]
        ], dtype=np.float32)

        rotation_gl = T @ rotation_matrix @ T.T

        self.cubes[cube_id] = {
            'position': position_gl,
            'rotation': rotation_gl,
            'size': cube_size
        }

        # Update textures if marker images provided
        if marker_images:
            if cube_id not in self.textures:
                self.textures[cube_id] = {}

            for face_type, image in marker_images.items():
                if image is not None:
                    # Create or update texture for this face
                    self.textures[cube_id][face_type] = self._create_texture(image)

    def remove_cube(self, cube_id):
        """Remove a cube from the world"""
        if cube_id in self.cubes:
            del self.cubes[cube_id]
        # Clean up textures
        if cube_id in self.textures:
            for texture_id in self.textures[cube_id].values():
                glDeleteTextures([texture_id])
            del self.textures[cube_id]

    def clear_cubes(self):
        """Remove all cubes"""
        # Clean up all textures
        for cube_id in list(self.textures.keys()):
            for texture_id in self.textures[cube_id].values():
                glDeleteTextures([texture_id])
        self.textures.clear()
        self.cubes.clear()

    def _create_texture(self, image):
        """
        Create an OpenGL texture from an OpenCV image
        :param image: OpenCV image (BGR format)
        :return: OpenGL texture ID
        """
        import cv2

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Flip vertically for OpenGL texture coordinates
        image_rgb = cv2.flip(image_rgb, 0)

        # Generate texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Upload texture data
        height, width = image_rgb.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_rgb)

        return texture_id

    def _draw_grid(self, size=1000, step=100):
        """Draw a ground grid"""
        glDisable(GL_LIGHTING)
        glColor3f(*self.grid_color)
        glBegin(GL_LINES)
        for i in range(-size, size + step, step):
            # Lines parallel to X axis
            glVertex3f(i, 0, -size)
            glVertex3f(i, 0, size)
            # Lines parallel to Z axis
            glVertex3f(-size, 0, i)
            glVertex3f(size, 0, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_axes(self, length=100):
        """Draw world coordinate axes at origin"""
        glDisable(GL_LIGHTING)
        glLineWidth(3)

        # X axis (red)
        glColor3f(*self.axis_colors['x'])
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)
        glEnd()

        # Y axis (green)
        glColor3f(*self.axis_colors['y'])
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, length, 0)
        glEnd()

        # Z axis (blue)
        glColor3f(*self.axis_colors['z'])
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        glEnd()

        glLineWidth(1)
        glEnable(GL_LIGHTING)

    def _draw_cube(self, cube_id, position, rotation_matrix, size=44):
        """
        Draw a cube at specified position and orientation using 3D model or primitives
        :param cube_id: Cube identifier (e.g., "cube_1", "cube_2", "cube_3") for model/texture lookup
        :param position: [x, y, z] in mm
        :param rotation_matrix: 3x3 rotation matrix
        :param size: cube size in mm
        """
        glPushMatrix()

        # Translate to cube position
        glTranslatef(position[0], position[1], position[2])

        # Apply rotation (convert 3x3 to 4x4 matrix)
        rot_4x4 = np.eye(4, dtype=np.float32)
        rot_4x4[:3, :3] = rotation_matrix.astype(np.float32)
        glMultMatrixf(rot_4x4.T.flatten())

        # Correct OBJ model orientation to match our coordinate system
        # The OBJ model needs to be reoriented to stand upright with correct facing
        glRotatef(90, 1, 0, 0)   # Rotate 90° around X to stand upright
        glRotatef(180, 0, 1, 0)  # Rotate 180° around Y to face correct direction

        # Extract numeric cube ID from cube_id string (e.g., "cube_1" -> 0)
        numeric_id = None
        if isinstance(cube_id, str) and cube_id.startswith("cube_"):
            try:
                numeric_id = int(cube_id.split("_")[1]) - 1  # Convert to 0-indexed
            except:
                pass
        elif isinstance(cube_id, int):
            numeric_id = cube_id

        # Scale from cm (OBJ units) to mm (our units)
        # OBJ model is ~4.1cm, we want 44mm = 4.4cm
        scale_factor = 44.0 / 41.12  # Approximate cube size in OBJ
        glScalef(scale_factor * 10, scale_factor * 10, scale_factor * 10)  # cm to mm conversion

        # Try to render the 3D model
        if numeric_id is not None and numeric_id in self.cube_models:
            self._draw_obj_model(self.cube_models[numeric_id])
        else:
            # Fallback to primitive cube rendering
            self._draw_primitive_cube(cube_id, size / (scale_factor * 10))

        glPopMatrix()

    def _draw_obj_model(self, scene):
        """Render a LoadedObjFile scene"""
        glEnable(GL_TEXTURE_2D)

        for mesh_name, faces in scene.mesh_faces.items():
            for face in faces:
                positions, normals, tex_coords, material_name = face

                # Get material for this face
                material = scene.mtl.get(material_name, {}) if scene.mtl else {}

                # Bind texture if available
                if 'texture_Kd' in material:
                    glBindTexture(GL_TEXTURE_2D, material['texture_Kd'])

                # Set material properties - force all to white
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [1.0, 1.0, 1.0, 1.0])
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])  # No specular highlights
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)

                # Draw the face
                glBegin(GL_POLYGON)

                for i, pos_idx in enumerate(positions):
                    # OBJ indices start at 1, convert to 0-based
                    pos_idx -= 1

                    # Apply texture coordinate if available
                    if i < len(tex_coords) and tex_coords[i] > 0:
                        tex_idx = tex_coords[i] - 1
                        if tex_idx < len(scene.tex_coords):
                            u, v = scene.tex_coords[tex_idx]
                            glTexCoord2f(u, v)

                    # Apply normal if available
                    if i < len(normals) and normals[i] > 0:
                        norm_idx = normals[i] - 1
                        if norm_idx < len(scene.normals):
                            nx, ny, nz = scene.normals[norm_idx]
                            glNormal3f(nx, ny, nz)

                    # Apply vertex position
                    if pos_idx < len(scene.vertices):
                        x, y, z = scene.vertices[pos_idx]
                        glVertex3f(x, y, z)

                glEnd()

        glDisable(GL_TEXTURE_2D)

    def _draw_indexed_vertices(self, material):
        """Draw vertices using indexed drawing (VAOs) for performance"""
        # Assume material.vertices contains interleaved vertex data
        # and material.indices contains the index array

        # Bind the VAO for this material
        glBindVertexArray(material.vao_id)

        # Draw the indexed vertices
        glDrawElements(GL_TRIANGLES, len(material.indices), GL_UNSIGNED_INT, None)

        # Unbind the VAO
        glBindVertexArray(0)

    def _draw_primitive_cube(self, cube_id, size=44):
        """Fallback primitive cube rendering (original implementation)"""
        half = size / 2.0

        # Define vertices of the cube
        vertices = [
            [-half, -half, -half],  # 0
            [half, -half, -half],   # 1
            [half, half, -half],    # 2
            [-half, half, -half],   # 3
            [-half, -half, half],   # 4
            [half, -half, half],    # 5
            [half, half, half],     # 6
            [-half, half, half]     # 7
        ]

        # Draw cube edges
        glDisable(GL_TEXTURE_2D)
        glColor3f(*self.cube_color)
        glLineWidth(2)
        glBegin(GL_LINES)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

        # Define faces with vertex indices and texture coordinates
        faces = [
            (0, 1, 2, 3),  # Back (-Z)  - face_0 when looking at it
            (4, 5, 6, 7),  # Front (+Z) - face_2 (opposite of face_0)
            (0, 1, 5, 4),  # Bottom (-Y)
            (2, 3, 7, 6),  # Top (+Y)
            (0, 3, 7, 4),  # Left (-X)  - face_3
            (1, 2, 6, 5)   # Right (+X) - face_1
        ]

        # Texture coordinates for each face (standard square mapping)
        tex_coords = [
            (0, 0), (1, 0), (1, 1), (0, 1)  # Standard UV mapping
        ]

        # Draw faces with textures or colors
        # No transparency - use solid colors

        # Get textures for this cube if available
        cube_textures = self.textures.get(cube_id, {})

        for face_idx, face_verts in enumerate(faces):
            # Determine face type for texture lookup
            # This is simplified - you may need to map face_idx to actual detected face
            texture_id = None

            if face_idx == 2:  # Bottom face
                glDisable(GL_TEXTURE_2D)
                glColor3f(1.0, 1.0, 1.0)  # White for bottom
            elif face_idx == 3:  # Top face
                glDisable(GL_TEXTURE_2D)
                glColor3f(1.0, 1.0, 1.0)  # White for top
            else:
                # Wall faces - use white with full opacity
                glDisable(GL_TEXTURE_2D)
                glColor3f(1.0, 1.0, 1.0)  # White for walls

            glBegin(GL_QUADS)
            for i, vertex_idx in enumerate(face_verts):
                if texture_id:
                    glTexCoord2f(*tex_coords[i])
                glVertex3fv(vertices[vertex_idx])
            glEnd()

        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)

        # Draw coordinate axes at cube center
        axis_length = size * 0.8
        glLineWidth(2)

        # X axis (red)
        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)
        glEnd()

        # Y axis (green)
        glColor3f(0, 1, 0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)
        glEnd()

        # Z axis (blue)
        glColor3f(0, 0, 1)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)
        glEnd()

        glLineWidth(1)
        glPopMatrix()

    def _draw_camera(self):
        """Draw robot as a box of dimensions 56x70x90mm positioned on surface with camera at 38mm height"""
        glDisable(GL_LIGHTING)
        glPushMatrix()

        # Robot box dimensions: 56x70x90mm (width x height x depth) (COZMO-28)
        width = 56.0   # X dimension
        height = 70.0  # Y dimension
        depth = 90.0   # Z dimension

        # Position the robot box so camera is at origin and bottom is on surface
        # Camera is at (0,0,0) and 38mm above surface, so robot box needs to be offset
        # Robot front face should be at Z=0, bottom at Y=0, and camera at Y=38mm
        robot_x = 0.0
        robot_y = 0.0  # Robot bottom at Y=0 (on ground), camera at Y=38mm
        robot_z = -depth/2.0  # Position so front face is at Z=0

        glTranslatef(robot_x, robot_y, robot_z)

        # Half dimensions for box drawing
        half_width = width / 2.0
        half_height = height / 2.0
        half_depth = depth / 2.0

        # Define vertices of the box (centered at translation point)
        vertices = [
            [-half_width, 0, -half_depth],          # 0 - back bottom left (bottom at Y=0)
            [half_width, 0, -half_depth],           # 1 - back bottom right
            [half_width, height, -half_depth],      # 2 - back top right
            [-half_width, height, -half_depth],     # 3 - back top left
            [-half_width, 0, half_depth],           # 4 - front bottom left
            [half_width, 0, half_depth],            # 5 - front bottom right
            [half_width, height, half_depth],       # 6 - front top right
            [-half_width, height, half_depth]       # 7 - front top left
        ]

        # Robot color - solid gray
        glColor3f(0.5, 0.5, 0.5)  # Gray without transparency
        glLineWidth(3)

        # Draw box edges
        glBegin(GL_LINES)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

        # Draw filled faces for better visibility
        glColor3f(0.5, 0.5, 0.5)  # Same gray for fill
        glBegin(GL_QUADS)
        # Front face
        glVertex3fv(vertices[4])
        glVertex3fv(vertices[5])
        glVertex3fv(vertices[6])
        glVertex3fv(vertices[7])

        # Back face
        glVertex3fv(vertices[0])
        glVertex3fv(vertices[1])
        glVertex3fv(vertices[2])
        glVertex3fv(vertices[3])

        # Left face
        glVertex3fv(vertices[0])
        glVertex3fv(vertices[3])
        glVertex3fv(vertices[7])
        glVertex3fv(vertices[4])

        # Right face
        glVertex3fv(vertices[1])
        glVertex3fv(vertices[2])
        glVertex3fv(vertices[6])
        glVertex3fv(vertices[5])

        # Top face
        glVertex3fv(vertices[3])
        glVertex3fv(vertices[2])
        glVertex3fv(vertices[6])
        glVertex3fv(vertices[7])

        # Bottom face (on surface)
        glVertex3fv(vertices[0])
        glVertex3fv(vertices[1])
        glVertex3fv(vertices[5])
        glVertex3fv(vertices[4])
        glEnd()

        glLineWidth(1)
        glPopMatrix()
        glEnable(GL_LIGHTING)

    def handle_input(self):
        """Handle keyboard and mouse input for camera control"""
        keys = pygame.key.get_pressed()

        move_speed = 10
        rot_speed = 2

        # Camera movement
        if keys[K_w]:
            self.camera_pos[2] += move_speed
        if keys[K_s]:
            self.camera_pos[2] -= move_speed
        if keys[K_a]:
            self.camera_pos[0] += move_speed
        if keys[K_d]:
            self.camera_pos[0] -= move_speed
        if keys[K_q]:
            self.camera_pos[1] += move_speed
        if keys[K_e]:
            self.camera_pos[1] -= move_speed

        # Camera rotation
        if keys[K_UP]:
            self.camera_rot[0] += rot_speed
        if keys[K_DOWN]:
            self.camera_rot[0] -= rot_speed
        if keys[K_LEFT]:
            self.camera_rot[1] += rot_speed
        if keys[K_RIGHT]:
            self.camera_rot[1] -= rot_speed

    def render(self):
        """Render the 3D world"""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.handle_input()

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup camera
        glLoadIdentity()
        glTranslatef(*self.camera_pos)
        glRotatef(self.camera_rot[0], 1, 0, 0)
        glRotatef(self.camera_rot[1], 0, 1, 0)
        glRotatef(self.camera_rot[2], 0, 0, 1)

        # Draw world
        self._draw_grid()
        self._draw_axes()
        self._draw_camera()
        self._draw_floor()

        # Draw all cubes (create snapshot to avoid iteration errors)
        cubes_snapshot = dict(self.cubes)
        for cube_id, cube_data in cubes_snapshot.items():
            self._draw_cube(
                cube_id,
                cube_data['position'],
                cube_data['rotation'],
                cube_data['size']
            )


        pygame.display.flip()
        return True

    def close(self):
        """Clean up and close the viewer"""
        pygame.quit()

    def _draw_floor(self, size=2000):
        """Draw a solid floor plane at Y=0"""
        glDisable(GL_LIGHTING)
        glColor3f(0.8, 0.8, 0.8)  # Light gray floor
        glBegin(GL_QUADS)
        glVertex3f(-size, 0, -size)
        glVertex3f(size, 0, -size)
        glVertex3f(size, 0, size)
        glVertex3f(-size, 0, size)
        glEnd()
        glEnable(GL_LIGHTING)


if __name__ == "__main__":
    # Test the viewer
    viewer = World3DViewer()

    # Add a test cube
    position = [100, 0, 200]
    rotation = np.eye(3)
    viewer.update_cube("test_cube", position, rotation)

    clock = pygame.time.Clock()
    running = True

    print("3D Viewer Controls:")
    print("  W/S/A/D - Move camera")
    print("  Q/E - Move camera up/down")
    print("  Arrow keys - Rotate camera")
    print("  Close window to exit")

    while running:
        running = viewer.render()
        clock.tick(60)

    viewer.close()
