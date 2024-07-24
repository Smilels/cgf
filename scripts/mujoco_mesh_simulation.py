import argparse
import json
import os
import sys
import time
import warnings
from typing import Literal
from xml.etree import ElementTree
import mujoco
import mujoco.viewer

import numpy as np
import torch
import viser
from manopth.manolayer import ManoLayer
from manopth.demo import display_hand
from IPython import embed

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cgf.robotics import KinematicsLayer, rescale_urdf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
END_LINKS = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]  # end links for the allegro hand
MANO_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/mano_v1_2_models")
URDF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data/original_robot/allegro_hand_ros/allegro_hand_description/"
)
DENSE_GEO_PATH = os.path.join(os.path.dirname(__file__), "..", "data/geometry/allegro_hand/right_dense_geo.npz")

# Path to the xml file
XML_PATH = os.path.join(os.path.dirname(__file__), "..", "assets/mujoco/right_hand_root.xml")

class MuJoCoMeshWorld:
  # inline hard code class for efficiency
  def template_xml(meshn, camera_dir="camera"):
    ap = 3.14159265359 if camera_dir == 'camera' else 0 # otherwise OpenGL frame
    gravity = 9.8 if camera_dir == 'camera' else -9.8
    light_direction = '0 1.0 4' if camera_dir == 'camera' else '0 -1.0 -4'
    ligh_pos = '0 -1.0 -4' if camera_dir == 'camera' else '0 1.0 4'

    OBJ_ASSET = """<mesh file="obj_mesh.obj"/>\n"""
    OBJ_QUAT_BODY = """<geom mesh="obj_mesh" class="visual" condim="6"/>\n"""
    for i in range(meshn):
      OBJ_ASSET += f"""        <mesh file="obj_mesh{i}.obj"/>\n"""
      OBJ_QUAT_BODY += f"""          <geom mesh="obj_mesh{i}" class="collision" condim="6"/>\n"""
    xml_string = \
    f"""<mujoco model="init">
      <compiler autolimits="true" angle="radian"/>
      <option gravity="0 {gravity} 0"/>
      <default>
        <default class="visual">
          <geom group="2" type="mesh" contype="0" conaffinity="0"/>
        </default>
        <default class="collision">
          <geom group="3" type="mesh"/>
        </default>
      </default>

      <asset>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
          width="512" height="512"/>	
        <material name='MatGnd' reflectance='.1' texture="texplane" texrepeat="2 2" texuniform="true"/>  
        <material name="floor" reflectance=".1"/>

        <mesh file="mesh0.obj"/>
        <mesh file="mesh1.obj"/>
        <mesh file="mesh2.obj"/>
        <mesh file="mesh3.obj"/>
        <mesh file="mesh4.obj"/>
        <mesh file="mesh5.obj"/>
        <mesh file="mesh6.obj"/>
        <mesh file="mesh7.obj"/>
        <mesh file="mesh8.obj"/>
        <mesh file="mesh9.obj"/>
        <mesh file="mesh10.obj"/>
        <mesh file="mesh11.obj"/>
        <mesh file="mesh12.obj"/>
        <mesh file="mesh13.obj"/>
        <mesh file="mesh14.obj"/>
        <mesh file="mesh15.obj"/>

        {OBJ_ASSET}
      </asset>

      
      <worldbody>
        <camera name="camera1" pos="0 0 0" euler="{ap} 0 0"/>
        <light directional='false' diffuse='.8 .8 .8' specular='0.3 0.3 0.3' pos='{ligh_pos}' dir='{light_direction}'/>

        <body name="obj">
          <freejoint />
          {OBJ_QUAT_BODY}
        </body>

        <body name="hand">
            <geom name="mesh0" mesh="mesh0" type="mesh"/>
            <geom name="mesh1" mesh="mesh1" type="mesh"/>
            <geom name="mesh2" mesh="mesh2" type="mesh"/>
            <geom name="mesh3" mesh="mesh3" type="mesh"/>
            <geom name="mesh4" mesh="mesh4" type="mesh"/>
            <geom name="mesh5" mesh="mesh5" type="mesh"/>
            <geom name="mesh6" mesh="mesh6" type="mesh"/>
            <geom name="mesh7" mesh="mesh7" type="mesh"/>
            <geom name="mesh8" mesh="mesh8" type="mesh"/>
            <geom name="mesh9" mesh="mesh9" type="mesh"/>
            <geom name="mesh10" mesh="mesh10" type="mesh"/>
            <geom name="mesh11" mesh="mesh11" type="mesh"/>
            <geom name="mesh12" mesh="mesh12" type="mesh"/>
            <geom name="mesh13" mesh="mesh13" type="mesh"/>
            <geom name="mesh14" mesh="mesh14" type="mesh"/>
            <geom name="mesh15" mesh="mesh15" type="mesh"/>
        </body>

      </worldbody>

      <actuator>
        <adhesion name="hand_act" body="hand" ctrlrange="0 10" gain="100"/>
      </actuator>

      </mujoco>
    """
    return xml_string

  def getworld(obj_name, camera_dir):
    if camera_dir == 'camera':
      xml, obj_asset = MuJoCoMeshWorld.dexycb_world[MuJoCoMeshWorld.obj_index[obj_name]]
    return xml, deepcopy(obj_asset)

  FILE_ROOT = os.getcwd().replace('main', '') + 'MuJoCo_data'
  obj_names = sorted([p for p in os.listdir(f'{FILE_ROOT}/skeletons_CoACD') if '0' in p])
  obj_index = {}
  dexycb_world = []
  for i, n in enumerate(obj_names):
    obj_index[n] = i

  for obj in obj_names:
    obj_folder = f'{FILE_ROOT}/skeletons_CoACD/{obj}/'
    meshes = [p for p in os.listdir(obj_folder) if p.endswith('.obj')]
    obj_asset = {}
    for m in meshes:
      with open(f'{FILE_ROOT}/skeletons_CoACD/{obj}/{m}', 'rb') as f:
        obj_asset[m] = f.read() 

    dex_xml = template_xml(len(meshes) - 1, 'camera')
    dexycb_world.append((dex_xml, obj_asset))



def get_mapping(mano_layer):
    finger_index = torch.argmax(mano_layer.th_weights, dim=-1)
    verts_mapping = {}
    for idx, k in enumerate(finger_index):
        kp = k.item()
        if kp in verts_mapping:
            verts_mapping[kp].add(idx)
        else:
            verts_mapping[kp] = set([idx])

    faces = mano_layer.th_faces

    face_mapping = {}

    for f in faces:
        for k, v in verts_mapping.items():
            for v_idx in f:
                v_idx_p = v_idx.item()
                if v_idx_p in v:
                    face_mapping.setdefault(k, []).append(f)
                    break

    for k, v in face_mapping.items():
        for f in v:
            for v_idx in f:
                v_idxp = v_idx.item()
                verts_mapping[k].add(v_idxp)

    for k, v in verts_mapping.items():
        verts_mapping[k] = sorted(list(v))

    face_mapping_reshape = {}

    for k,v in face_mapping.items():
        for jdx, f in enumerate(v):
            t = torch.zeros(3).long()
            for i, vdx in enumerate(f):
                vdxp = vdx.item()
                t[i] = verts_mapping[k].index(vdxp)
            face_mapping_reshape.setdefault(k, []).append(t)

    for k, v in face_mapping_reshape.items():
        face_mapping_reshape[k] = torch.stack(v, dim=0)

    return verts_mapping, face_mapping_reshape


def main(data_root: str, mano_side: Literal["left", "right"] = "left", seq_id: str = None, aug_idx: int = 0, obj_name: str="chefcan"), :

    # human hand
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mano_layer = ManoLayer(flat_hand_mean=False, ncomps=45, side=mano_side, mano_root=MANO_DIR, use_pca=True)
        mano_layer = mano_layer.to(DEVICE)



    with mujoco.viewer.launch_passive(mj_model, model_data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            with open(os.path.join(args.data_root, "processed", "meta.json"), "r") as f:
                meta = json.load(f)

            pose_m_aug_npz_file = np.load(os.path.join(data_root, "processed", "pose_m_aug.npz"))
            seq_ids = [x[:-5] for x in pose_m_aug_npz_file.files if "mano" in x]
            seq_ids = [x for x in seq_ids if meta[x]["mano_sides"][0] == args.mano_side]
            sorted_seq_ids = sorted(seq_ids)

            seq_betas = np.array([meta[seq_id]["betas"] for seq_id in sorted_seq_ids])
            valid_frames_range = np.array([meta[seq_id]["valid_frames_range"] for seq_id in sorted_seq_ids])
            # add 1 to the end frame
            valid_frames_range[:, 1] += 1
            
            for seq_id in sorted_seq_ids[:11]:
            # if seq_id is None:
            #     seq_id = sorted_seq_ids[0]
                assert seq_id in sorted_seq_ids
                begin_f, end_f = valid_frames_range[sorted_seq_ids.index(seq_id)]


                # display mano hand
                mano_pose = pose_m_aug_npz_file[f"{seq_id}_mano"][:, begin_f:end_f]
                mano_pose = torch.from_numpy(mano_pose[aug_idx, :, 0]).clone().to(DEVICE, dtype=DTYPE)
                betas = (
                    torch.from_numpy(seq_betas[sorted_seq_ids.index(seq_id)])
                    .clone()
                    .expand(mano_pose.shape[0], -1)
                    .to(DEVICE, dtype=DTYPE)
                )
                mano_verts, mano_joint = mano_layer(mano_pose[..., :48], betas, mano_pose[..., 48:51])

                # object and hand load
                verts_mapping, face_mapping_reshape = get_mapping(mano_layer)

                xml, obj_asset = MuJoCoMeshWorld.getworld(obj_name, "camera") # get data
                hand_asset = {}
                ct = 0
                for k, v in verts_mapping.items():
                    v0 = mano_verts_abs[v, :]
                    f0 = face_mapping_reshape[k].int()
                    with tempfile.NamedTemporaryFile(suffix='.obj') as file_obj:
                            pytorch3d.io.save_obj(file_obj.name, verts=v0.clone().detach(), faces=f0.clone().detach())
                            hand_asset[f'mesh{ct}.obj'] = file_obj.read()
                    ct += 1

                ASSET = {}
                ASSET.update(obj_asset)
                ASSET.update(hand_asset)
                try:
                    mj_model = mujoco.MjModel.from_xml_string(xml, ASSET) # always create new model here
                except:
                    print("use convex instead")
                    ct = 0
                    hand_asset = {}
                    for k, v in verts_mapping.items():
                        v0 = mano_verts_abs[v, :].cpu().numpy()
                        convex_m = trimesh.convex.convex_hull(v0, qhull_options='QbB Pp Qt')
                        convex_v = torch.tensor(convex_m.vertices)
                        convex_f = torch.tensor(convex_m.faces).int()
                        with tempfile.NamedTemporaryFile(suffix='.obj') as file_obj:
                            pytorch3d.io.save_obj(file_obj.name, verts=convex_v.clone().detach(), faces=convex_f.clone().detach())
                            hand_asset[f'mesh{ct}.obj'] = file_obj.read()
                        ct += 1
                    
                    ASSET = {}
                    ASSET.update(obj_asset)
                    ASSET.update(hand_asset)
                    mj_model = mujoco.MjModel.from_xml_string(xml, ASSET) # always create new model here
                
                model_data = mujoco.MjData(mj_model)
                renderer = mujoco.Renderer(mj_model)

                data.qpos[:MODEL_NV - OBJ_ROT_DIM] = obj_trans
                data.qpos[MODEL_NV - OBJ_ROT_DIM:] = obj_rot

                # display robot hand in mujoco
                qpos = np.load(os.path.join(data_root, "processed", f"retargeting_{mano_side}", f"{aug_idx}.npz"))["qpos"][aug_idx]
                print("here")

                # tf3ds = kinematics_layer(torch.from_numpy(qpos).to(DEVICE, dtype=DTYPE))
                # print(tf3ds)

                model_data.ctrl[3:] = qpos

                mujoco.mj_step(mj_model, model_data)
                #mujoco.mj_forward(model, data) # no step
                renderer.update_scene(model_data)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                viewer.sync()
                time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--mano_side", type=str, choices=["left", "right"], default="right")
    args = parser.parse_args()
    main(args.data_root, args.mano_side, obj_name="chef_can")
