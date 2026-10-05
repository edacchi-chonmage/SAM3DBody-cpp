"""Import a person+board BVH into Blender, add a skateboard mesh on the 'board' bone, export FBX, re-import and check.

Usage: /Applications/Blender.app/Contents/MacOS/Blender -b --python blender_skate_fbx.py -- IN.bvh OUT.fbx
Also renders a Workbench still (frame 120) to <dir of OUT.fbx>/fbx_check.png.
"""
import sys
import math
import os

import bpy
import bmesh
from mathutils import Vector, Matrix

bvh, fbx = sys.argv[sys.argv.index("--") + 1:][:2]
png = os.path.join(os.path.dirname(os.path.abspath(fbx)), "fbx_check.png")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_anim.bvh(filepath=bvh, global_scale=0.01, frame_start=1, use_fps_scale=False,
                        update_scene_fps=False, update_scene_duration=True, rotate_mode='NATIVE')
sc = bpy.context.scene
print("IMPORT objects:", [(o.name, o.type) for o in bpy.data.objects])
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
print("armatures: 1 (one armature, two root bones)", [b.name for b in arm.data.bones if b.parent is None])
arms = [arm]
nf = int(max(a.animation_data.action.frame_range[1] for a in arms))
sc.render.fps, sc.render.fps_base = 30000, 1001
sc.frame_start, sc.frame_end = 1, nf

# ---- body meshes: one capsule per main bone, bone-parented (bone-parent space origin = bone tail)
BODY = {"hip": .06, "abdomen": .06, "chest": .06, "neck": .04, "neck1": .04, "head": None,
        "lCollar": .04, "rCollar": .04, "lShldr": .045, "rShldr": .045, "lForeArm": .04, "rForeArm": .04,
        "lHand": .035, "rHand": .035, "lButtock": .05, "rButtock": .05, "lThigh": .06, "rThigh": .06,
        "lShin": .045, "rShin": .045, "lFoot": .04, "rFoot": .04}
body_mat = bpy.data.materials.new("body"); body_mat.diffuse_color = (0.85, 0.55, 0.40, 1)
board_mat = bpy.data.materials.new("boardmat"); board_mat.diffuse_color = (0.15, 0.30, 0.80, 1)

def bone_mesh(bone, r):
    L = bone.length
    bm_ = bmesh.new()
    if r is None:   # head: 11 cm sphere at the bone middle
        bmesh.ops.create_uvsphere(bm_, u_segments=16, v_segments=10, radius=0.11,
                                  matrix=Matrix.Translation((0, L * 0.5, 0)))
    else:           # capsule along +Y (bone axis) from head (0) to tail (L)
        bmesh.ops.create_cone(bm_, cap_ends=True, segments=12, radius1=r, radius2=r, depth=L,
                              matrix=Matrix.Translation((0, L * 0.5, 0)) @ Matrix.Rotation(math.pi / 2, 4, 'X'))
        for y in (0, L):
            bmesh.ops.create_uvsphere(bm_, u_segments=12, v_segments=6, radius=r, matrix=Matrix.Translation((0, y, 0)))
    me = bpy.data.meshes.new("m_" + bone.name)
    bm_.to_mesh(me); bm_.free()
    me.materials.append(body_mat)
    o = bpy.data.objects.new("body_" + bone.name, me)
    bpy.context.scene.collection.objects.link(o)
    o.parent, o.parent_type, o.parent_bone = arm, 'BONE', bone.name
    o.matrix_parent_inverse = Matrix.Translation((0, -L, 0))
    return o

for bn, r in BODY.items():
    b = arm.data.bones[bn]
    if b.length > 1e-4:
        bone_mesh(b, r)

# skateboard mesh: deck top at board bone origin, wheels below
def box(name, size, loc):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object; o.name = name; o.scale = size; return o
parts = [box("deck", (0.80, 0.02, 0.21), (0, -0.01, 0))]
for sx in (-0.27, 0.27):
    for sz in (-0.08, 0.08):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.027, depth=0.03, location=(sx, -0.02 - 0.027, sz),
                                            rotation=(math.pi / 2, 0, 0))
        parts.append(bpy.context.object)
bpy.ops.object.select_all(action='DESELECT')
for o in parts: o.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
bm = bpy.context.object; bm.name = "skateboard"; bm.data.materials.append(board_mat)
bpy.ops.object.select_all(action='DESELECT')
bm.select_set(True); arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
arm.data.bones.active = arm.data.bones["board"]
bpy.ops.object.parent_set(type='BONE')
bpy.ops.object.mode_set(mode='OBJECT')
# Keep the mesh exactly on the board pose: set the parent inverse at frame 1 so that
# world = BVH board transform (row 0 of MOTION: last 6 values, Z Y X euler, cm -> m).
from mathutils import Matrix, Euler
v = [float(x) for x in open(bvh).read().split("MOTION")[1].strip().splitlines()[2].split()][-6:]
# the BVH importer maps BVH (x, y, z) to Blender world (x, -z, y)
C = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
want = C @ Matrix.Translation(Vector(v[:3]) * 0.01) @ Euler(
    (math.radians(v[5]), math.radians(v[4]), math.radians(v[3])), 'XYZ').to_matrix().to_4x4()   # BVH 'Zrot Yrot Xrot' = Rz@Ry@Rx = Blender order 'XYZ'
sc.frame_set(1)
pb = arm.pose.bones["board"]
parent_m = arm.matrix_world @ pb.matrix @ Matrix.Translation((0, pb.bone.length, 0))
bm.matrix_parent_inverse = parent_m.inverted() @ want
sc.frame_set(1)

def report(tag):
    for f in (1, 300):
        sc.frame_set(f)
        o = bpy.data.objects.get("skateboard")
        print(f"{tag} frame {f}: skateboard world loc", tuple(round(v, 3) for v in o.matrix_world.translation))

report("pre-export")
bpy.ops.export_scene.fbx(filepath=fbx, bake_anim=True, add_leaf_bones=False,
                         # default bake_anim_use_all_actions=True kept on purpose: with False the re-import range shifts +1 (2..480)
                         axis_forward='-Z', axis_up='Y', object_types={'ARMATURE', 'MESH'},
                         bake_anim_simplify_factor=0.0)

# ---- re-import into empty scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx, axis_forward='-Z', axis_up='Y')
sc = bpy.context.scene
print("REIMPORT objects:", [(o.name, o.type, o.parent.name if o.parent else None) for o in bpy.data.objects])
for o in bpy.data.objects:
    if o.animation_data and o.animation_data.action:
        print("action", o.name, tuple(o.animation_data.action.frame_range))
sc.frame_start, sc.frame_end = 1, nf
report("reimport")
rows = open(bvh).read().split("MOTION")[1].strip().splitlines()[2:]
for f in (1, 300):
    r = [float(x) for x in rows[f - 1].split()][-6:]   # BVH frame f-1 (0-based) = Blender frame f
    print(f"expected BVH row {f - 1} board loc (x, -z, y)*0.01:", (round(r[0] * .01, 3), round(-r[2] * .01, 3), round(r[1] * .01, 3)))
print("fbx size MB", os.path.getsize(fbx) / 1e6)

# ---- render check
sc.frame_set(120)
bm = bpy.data.objects["skateboard"]
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
hip = arm.matrix_world @ arm.pose.bones["hip"].head
centre = (bm.matrix_world.translation + hip) / 2 + Vector((0, 0, 0.4))
bpy.ops.object.camera_add(location=centre + Vector((0, -5.0, 0.6)))   # camera side = smaller depth (y); scene is Z-up
cam = bpy.context.object
cam.rotation_euler = (centre - cam.location).to_track_quat('-Z', 'Y').to_euler()
sc.camera = cam
sc.render.engine = 'BLENDER_WORKBENCH'
sc.display.shading.color_type = 'MATERIAL'
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.render.filepath = png
bpy.ops.render.render(write_still=True)
print("rendered", png, "board", tuple(bm.matrix_world.translation), "hip", tuple(hip))
