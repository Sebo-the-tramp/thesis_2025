import numpy as np
import json
import struct
import pygltflib
from pathlib import Path
import logging

def inspect_glb_structure(file_path):
    """
    Inspect GLB file structure similar to how Three.js would parse it
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Load the GLB file in binary mode
    with open(file_path, 'rb') as f:
        # Read the GLB header
        magic = f.read(4)
        if magic != b'glTF':
            print("Not a valid GLB file")
            return
        
        version = struct.unpack('<I', f.read(4))[0]
        length = struct.unpack('<I', f.read(4))[0]
        
        print(f"=== GLB Header ===")
        print(f"Version: {version}")
        print(f"Total length: {length} bytes")
        
        # Read JSON chunk
        chunk_length = struct.unpack('<I', f.read(4))[0]
        chunk_type = f.read(4)
        
        if chunk_type == b'JSON':
            json_data = json.loads(f.read(chunk_length).decode('utf-8'))
            print("\n=== JSON Data Structure ===")
            
            # Print mesh information
            if 'meshes' in json_data:
                print("\nMeshes:")
                for i, mesh in enumerate(json_data['meshes']):
                    print(f"\nMesh {i}:")
                    for j, primitive in enumerate(mesh['primitives']):
                        print(f"\nPrimitive {j}:")
                        print("Attributes:", primitive.get('attributes', {}))
                        
                        # Check material
                        if 'material' in primitive:
                            material_idx = primitive['material']
                            if 'materials' in json_data:
                                material = json_data['materials'][material_idx]
                                print("\nMaterial properties:")
                                print(json.dumps(material, indent=2))
            
            # Print accessor information
            if 'accessors' in json_data:
                print("\nAccessors:")
                for i, accessor in enumerate(json_data['accessors']):
                    print(f"\nAccessor {i}:")
                    print(f"Type: {accessor.get('type')}")
                    print(f"Component Type: {accessor.get('componentType')}")
                    print(f"Count: {accessor.get('count')}")
                    if 'bufferView' in accessor:
                        print(f"Buffer View: {accessor['bufferView']}")
    
    # Now let's try to load it with pygltflib for additional inspection
    gltf = pygltflib.GLTF2().load(file_path)
    
    print("\n=== Additional Buffer Information ===")
    for i, buffer_view in enumerate(gltf.bufferViews):
        print(f"\nBufferView {i}:")
        print(f"Buffer: {buffer_view.buffer}")
        print(f"ByteOffset: {buffer_view.byteOffset}")
        print(f"ByteLength: {buffer_view.byteLength}")
        print(f"Target: {buffer_view.target}")  # 34962 is ARRAY_BUFFER (vertices, colors), 34963 is ELEMENT_ARRAY_BUFFER (indices)
        
        # Try to identify the content type
        content_type = "Unknown"
        for accessor in gltf.accessors:
            if accessor.bufferView == i:
                if hasattr(accessor, 'type'):
                    content_type = accessor.type
                break
        print(f"Content Type: {content_type}")

def try_threejs_style_load(file_path):
    """
    Try to load the GLB file similar to how Three.js would handle it
    """
    gltf = pygltflib.GLTF2().load(file_path)
    
    print("\n=== Three.js Style Analysis ===")
    
    # Check for possible color attributes (Three.js naming conventions)
    color_attributes = [
        'COLOR_0',          # Standard color attribute
        'color',            # Alternative naming
        '_color',           # Three.js internal naming
        'vertexColors',     # Three.js material property
    ]
    
    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            print("\nChecking primitive attributes:")
            
            # Print all available attributes
            attrs = primitive.attributes.__dict__
            print("Available attributes:", [k for k in attrs.keys() if not k.startswith('_')])
            
            # Check for color information in attributes
            for color_attr in color_attributes:
                if hasattr(primitive.attributes, color_attr):
                    print(f"Found color attribute: {color_attr}")
                    
                    # Get the accessor for this color attribute
                    color_accessor = gltf.accessors[getattr(primitive.attributes, color_attr)]
                    print(f"Color format: {color_accessor.componentType}")
                    print(f"Color type: {color_accessor.type}")
                    print(f"Number of colors: {color_accessor.count}")
            
            # Check material properties
            if primitive.material is not None:
                material = gltf.materials[primitive.material]
                print("\nMaterial properties:")
                if hasattr(material, 'pbrMetallicRoughness'):
                    pbr = material.pbrMetallicRoughness
                    if hasattr(pbr, 'baseColorFactor'):
                        print(f"Base color factor: {pbr.baseColorFactor}")
                    if hasattr(pbr, 'baseColorTexture'):
                        print(f"Has base color texture: {pbr.baseColorTexture is not None}")

if __name__ == "__main__":
    file_path = "./_assets/00002-00013.glb"
    print("=== Inspecting GLB Structure ===")
    inspect_glb_structure(file_path)
    print("\n=== Trying Three.js Style Loading ===")
    try_threejs_style_load(file_path)