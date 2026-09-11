"""Independent V8 water invariants, browser arithmetic parity and MuJoCo bridge.

These tests use analytic states; no actor commands or training targets are made.
"""
import json, shutil, subprocess, sys, unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training'))
from water import body_water, immersion, shape_parameters, apply_native

def sample_cases(n=96):
 rng=np.random.default_rng(190929)
 q=rng.normal(size=(n,4));q/=np.linalg.norm(q,axis=-1)[:,None]
 pos=np.column_stack([rng.uniform(2,4,n),rng.uniform(-.5,.5,n),rng.uniform(-.5,.5,n)])
 return dict(types=np.resize([2,3,4,6],n),sizes=rng.uniform(.03,.24,(n,3)),position=pos,quaternion=q,com=pos+rng.uniform(-.1,.1,(n,3)),linear=rng.uniform(-18,18,(n,3)),angular=rng.uniform(-30,30,(n,3)),mass=rng.uniform(.5,26,n),water_z=np.zeros(n))

def one(**updates):
 data=dict(types=np.array(3),sizes=np.array([.06,.21,0]),position=np.array([3.,.2,-.1]),quaternion=np.array([1.,0,0,0]),com=np.array([3.03,.18,-.04]),linear=np.array([1.,-2,-8]),angular=np.array([3.,9,-2]),mass=np.array(3.5),water_z=np.array(0.))
 data.update({k:np.asarray(v) for k,v in updates.items()});return data

class WaterTests(unittest.TestCase):
 def test_passivity_measured_from_actual_force_and_com_moment(self):
  data=sample_cases(384);out=body_water(**data)
  power=np.sum(out['dragForce']*data['linear']+out['dragTorque']*data['angular'],axis=-1)
  np.testing.assert_allclose(power,out['dragPower'],rtol=2e-13,atol=1e-8)
  self.assertLessEqual(np.max(power),1e-9)
  for value in out.values():self.assertTrue(np.isfinite(value).all())
 def test_principal_geometry(self):
  r,h=.06,.21
  _,area=shape_parameters(np.array(3),np.array([r,h,0.]))
  np.testing.assert_allclose(area,[4*r*h+np.pi*r*r]*2+[np.pi*r*r],rtol=1e-14)
  q=np.array([[1.,0,0,0],[2**-.5,0,2**-.5,0]])
  _,extent,_=immersion(np.array([3,3]),np.array([[r,h,0.]]*2),np.array([[3.,0,0]]*2),q,0.)
  np.testing.assert_allclose(extent,[r+h,r],atol=1e-14)
 def test_dry_boundary_and_basin(self):
  for position in [[3.,0,.27],[-.45,0,-3],[11.35,0,-3],[3.,3.57,-3]]:
   out=body_water(**one(position=position));self.assertAlmostEqual(float(out['fraction']),0.)
   np.testing.assert_allclose(out['force'],0.,atol=1e-13);np.testing.assert_allclose(out['torque'],0.,atol=1e-13)
  for z,f,c in [[0,.5,-.135],[-.27,1,-.27]]:
   out=body_water(**one(position=[3.,0,z],com=[3.,0,z],linear=[0.,0,0],angular=[0.,0,0]))
   self.assertAlmostEqual(float(out['fraction']),f);self.assertAlmostEqual(out['center'][2],c)
 def test_com_reference_change_preserves_wrench_and_power(self):
  data=one();a=body_water(**data);shift=np.array([.11,-.07,.04])
  b=body_water(**{**data,'com':data['com']+shift,'linear':data['linear']+np.cross(data['angular'],shift)})
  np.testing.assert_allclose(a['force'],b['force'],rtol=1e-13)
  np.testing.assert_allclose(b['torque'],a['torque']-np.cross(shift,a['force']),rtol=1e-13)
  np.testing.assert_allclose(a['dragPower'],b['dragPower'],rtol=1e-13)
 def test_translation_with_water_plane(self):
  data=sample_cases();a=body_water(**data);shift=np.array([.2,-.1,4.1])
  b=body_water(**{**data,'position':data['position']+shift,'com':data['com']+shift,'water_z':data['water_z']+shift[2]})
  for key in ['force','torque','dragPower','fraction']:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-9)
 def test_sphere_and_capsule_axis_symmetry(self):
  q=[np.cos(np.pi/8),0,0,np.sin(np.pi/8)]
  for kind in [2,3]:
   data=one(types=kind,sizes=[.13,.2,0.],position=[3.,0,-2],com=[3.,0,-2],linear=[1.,0,2],angular=[2.,0,1])
   a=body_water(**data);b=body_water(**{**data,'quaternion':np.array(q)})
   for key in ['dragForce','dragTorque']:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
 def test_batch_scalar_and_browser_arithmetic_agree(self):
  data=sample_cases();expected=body_water(**data)
  rows=[]
  for i in range(len(data['types'])):
   row={k:v[i] for k,v in data.items()};scalar=body_water(**row)
   for key in expected:np.testing.assert_allclose(scalar[key],expected[key][i],rtol=1e-12,atol=1e-10)
   row={k:v.tolist() for k,v in row.items()};row['type']=row.pop('types');row['size']=row.pop('sizes');row['waterZ']=row.pop('water_z');rows.append(row)
  node=shutil.which('node')
  if node is None:self.skipTest('Node is required for browser arithmetic parity')
  source="import {bodyWater} from './src/core/water.js'; let text=''; for await (const s of process.stdin) text+=s; process.stdout.write(JSON.stringify(JSON.parse(text).map(bodyWater)));"
  result=subprocess.run([node,'--input-type=module','-e',source],input=json.dumps(rows),capture_output=True,text=True,cwd=ROOT,check=True)
  actual=json.loads(result.stdout)
  for key in expected:np.testing.assert_allclose([row[key] for row in actual],expected[key],rtol=2e-12,atol=1e-9,err_msg=key)
 def test_native_bridge_uses_geom_frames_and_com_velocity_not_body_origin_velocity(self):
  import mujoco
  m=mujoco.MjModel.from_xml_path(str(ROOT/'public/physics/diver.xml'));d=mujoco.MjData(m)
  d.qpos[1:4]=[3.,0,-8.]
  d.qpos[4:8]=np.array([.8,.2,.3,.4])/np.linalg.norm([.8,.2,.3,.4])
  d.qvel[:]=np.random.default_rng(5019).uniform(-2,2,m.nv);mujoco.mj_forward(m,d)
  ids=np.arange(1,16);bodies=m.geom_bodyid[ids];linear=[];angular=[];quats=[]
  for geom,body in zip(ids,bodies):
   jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jacBodyCom(m,d,jp,jr,int(body))
   linear.append(jp@d.qvel);angular.append(jr@d.qvel)
   q=np.zeros(4);mujoco.mju_mat2Quat(q,d.geom_xmat[geom]);quats.append(q)
  expected=body_water(m.geom_type[ids],m.geom_size[ids],d.geom_xpos[ids],np.array(quats),d.xipos[bodies],np.array(linear),np.array(angular),m.body_mass[bodies],-7.5)
  # The batched trainer avoids per-body native calls by using appended sensors.
  # This shortcut is valid only while each geom center is its body's true COM.
  np.testing.assert_array_equal(m.body_geomnum[bodies],np.ones(15,dtype=int))
  np.testing.assert_allclose(d.geom_xpos[ids],d.xipos[bodies],atol=1e-14)
  np.testing.assert_allclose(d.sensordata[270:360].reshape(15,6),np.c_[linear,angular],rtol=1e-13,atol=1e-13)
  actual=apply_native(m,d,7.5)
  for key in ['force','torque','dragPower']:np.testing.assert_allclose(actual[key],expected[key],rtol=1e-11,atol=1e-9,err_msg=key)
  np.testing.assert_allclose(d.xfrc_applied[bodies,:3],actual['force']);np.testing.assert_allclose(d.xfrc_applied[bodies,3:],actual['torque'])
  np.testing.assert_allclose(d.xfrc_applied[np.setdiff1d(np.arange(m.nbody),bodies)],0)

if __name__=='__main__':unittest.main()
