"""V8 stateless water approximation, mirrored in src/core/water.js.
SI units; native world Z up. One submerged-envelope center per physical body.
Directional projected areas use collider dimensions, never bounding-sphere radius.
Quadratic drag is passive; buoyancy is external work and excluded
from dragPower. No added mass, free surface history, shielding, lift or CFD.
"""
import numpy as np
WATER_VERSION='directional-submerged-body-v1'
POOL_DEPTH=5.
DENSITY=1000.
DRAG_COEFFICIENT=.8
ANGULAR_COEFFICIENT=1.
BUOYANCY_MASS_RATIO=1.015

def rotation(q):
 w,x,y,z=np.moveaxis(np.asarray(q),-1,0)
 return np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],axis=-1).reshape(*np.shape(w),3,3)

def shape_parameters(types,sizes):
 types=np.asarray(types);sizes=np.asarray(sizes);r=sizes.copy()
 r=np.where((types==2)[...,None],sizes[...,:1],r)
 r=np.where((types==3)[...,None],np.stack([sizes[...,0],sizes[...,0],sizes[...,0]+sizes[...,1]],axis=-1),r)
 area=np.stack([np.pi*r[...,1]*r[...,2],np.pi*r[...,0]*r[...,2],np.pi*r[...,0]*r[...,1]],axis=-1)
 area=np.where((types==6)[...,None],area*(4/np.pi),area)
 cap=np.stack([4*sizes[...,0]*sizes[...,1]+np.pi*sizes[...,0]**2]*2+[np.pi*sizes[...,0]**2],axis=-1)
 area=np.where((types==3)[...,None],cap,area)
 return r,area

def immersion(types,sizes,position,quaternion,water_z):
 R=rotation(quaternion);row=R[...,2,:];types=np.asarray(types);sizes=np.asarray(sizes)
 extent=np.sqrt(np.sum((row*sizes)**2,axis=-1))
 extent=np.where(types==2,sizes[...,0],extent)
 extent=np.where(types==3,sizes[...,0]+sizes[...,1]*np.abs(row[...,2]),extent)
 extent=np.where(types==6,np.sum(np.abs(row)*sizes,axis=-1),extent)
 fraction=np.clip((np.asarray(water_z)-position[...,2]+extent)/np.maximum(2*extent,1e-9),0,1)
 inside=(position[...,0]>=-.44)&(position[...,0]<=11.34)&(np.abs(position[...,1])<=3.56)
 return fraction*inside,extent,R

def body_water(types,sizes,position,quaternion,com,linear,angular,mass,water_z):
 """Vectorizable arrays (...,3/4). linear is velocity at COM, not geom origin.
 Returns force/torque at COM. A vertical envelope approximates wet volume and its
 centroid; it is deliberately not an exact polygon/ellipsoid slice integration.
 """
 fraction,extent,R=immersion(types,sizes,position,quaternion,water_z)
 r,area=shape_parameters(types,sizes)
 center=np.array(position,copy=True);center[...,2]-=extent*(1-fraction)
 offset=center-com;velocity=linear+np.cross(angular,offset)
 local_v=np.einsum('...ji,...j->...i',R,velocity);local_w=np.einsum('...ji,...j->...i',R,angular)
 local_f=-.5*DENSITY*DRAG_COEFFICIENT*area*fraction[...,None]*local_v*np.linalg.norm(local_v,axis=-1,keepdims=True)
 k=.5*DENSITY*ANGULAR_COEFFICIENT*r*(np.roll(r,1,axis=-1)**4+np.roll(r,2,axis=-1)**4)
 local_t=-k*fraction[...,None]*local_w*np.linalg.norm(local_w,axis=-1,keepdims=True)
 drag=np.einsum('...ij,...j->...i',R,local_f);angular_drag=np.einsum('...ij,...j->...i',R,local_t)
 force=drag.copy();force[...,2]+=9.81*np.asarray(mass)*BUOYANCY_MASS_RATIO*fraction
 torque=angular_drag+np.cross(offset,force)
 drag_power=np.sum(local_f*local_v+local_t*local_w,axis=-1)
 return dict(force=force,torque=torque,dragForce=drag,dragTorque=angular_drag+np.cross(offset,drag),dragPower=drag_power,fraction=fraction,extent=extent,area=area,center=center)

def apply_native(model,data,height):
 """Reference bridge; used by exact prototype/parity tests, no owning handles."""
 import mujoco
 ids=np.arange(1,16);bodies=model.geom_bodyid[ids]
 velocity=np.empty((15,6))
 for i,b in enumerate(bodies):mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,int(b),velocity[i],0)
 result=body_water(model.geom_type[ids],model.geom_size[ids],data.geom_xpos[ids],data.geom_xquat[ids] if hasattr(data,'geom_xquat') else data.sensordata[45:150].reshape(15,7)[:,3:],data.xipos[bodies],velocity[:,3:],velocity[:,:3],model.body_mass[bodies],-height)
 data.xfrc_applied[:]=0
 data.xfrc_applied[bodies,:3]=result['force'];data.xfrc_applied[bodies,3:]=result['torque']
 return result
