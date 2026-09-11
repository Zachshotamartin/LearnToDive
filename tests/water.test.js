import test from 'node:test';
import assert from 'node:assert/strict';
import {bodyWater, immersion, shapeParameters, rotation} from '../src/core/water.js';

// Independent mechanical invariants; these do not tune the force law to a dive.
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
const rot=(q,v)=>{const R=rotation(q);return [0,1,2].map(i=>dot(R.slice(3*i,3*i+3),v));};
const product=([w,x,y,z],[v,a,b,c])=>[w*v-x*a-y*b-z*c,w*a+x*v+y*c-z*b,w*b-x*c+y*v+z*a,w*c+x*b-y*a+z*v];
const near=(a,b,tol=2e-10)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
const vectorNear=(a,b)=>a.forEach((x,i)=>near(x,b[i]));
const base=(extra={})=>({type:3,size:[.06,.21,0],position:[3,.2,-.1],quaternion:[1,0,0,0],com:[3.03,.18,-.04],linear:[1,-2,-8],angular:[3,9,-2],mass:3.5,waterZ:0,...extra});
function randomCases(n=384){
 let seed=928721;
 const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
 return Array.from({length:n},(_,i)=>{
  const q=Array.from({length:4},()=>2*random()-1), norm=Math.hypot(...q);
  const position=[2+2*random(),random()-.5,random()-.5];
  return base({type:[2,3,4,6][i%4],size:[.03+.15*random(),.05+.2*random(),.03+.15*random()],position,quaternion:q.map(x=>x/norm),com:add(position,Array.from({length:3},()=>.2*(random()-.5))),linear:Array.from({length:3},()=>36*(random()-.5)),angular:Array.from({length:3},()=>60*(random()-.5)),mass:.5+25*random()});
 });
}

test('water drag is passive at the supplied COM, including its off-center torque',()=>{
 for(const input of randomCases()){
  const out=bodyWater(input), power=dot(out.dragForce,input.linear)+dot(out.dragTorque,input.angular);
  near(power,out.dragPower);
  assert.ok(power<=1e-9,`drag added ${power} W`);
  assert.ok([...out.force,...out.torque,out.fraction].every(Number.isFinite));
  assert.ok(out.fraction>=0&&out.fraction<=1);
 }
});

test('shape extents and principal projected areas use actual sphere, capsule, ellipsoid and box sizes',()=>{
 const r=.06,h=.21;
 vectorNear(shapeParameters(3,[r,h,0]).area,[4*r*h+Math.PI*r*r,4*r*h+Math.PI*r*r,Math.PI*r*r]);
 near(immersion(3,[r,h,0],[3,0,0],[1,0,0,0],0).extent,r+h);
 near(immersion(3,[r,h,0],[3,0,0],[Math.SQRT1_2,0,Math.SQRT1_2,0],0).extent,r);
 near(immersion(2,[r,0,0],[3,0,0],[.5,.5,.5,.5],0).extent,r);
 const sizes=[.13,.055,.04],q=[Math.cos(Math.PI/8),0,Math.sin(Math.PI/8),0];
 near(immersion(6,sizes,[3,0,0],q,0).extent,(sizes[0]+sizes[2])*Math.SQRT1_2);
 near(immersion(4,sizes,[3,0,0],q,0).extent,Math.hypot(sizes[0],sizes[2])*Math.SQRT1_2);
});

test('dry/outside bodies exert zero force; wet-envelope fraction and center meet their declared boundaries',()=>{
 for(const position of [[3,0,.5],[-.45,0,-3],[11.35,0,-3],[3,3.57,-3]]){
  const out=bodyWater(base({position}));near(out.fraction,0);vectorNear(out.force,[0,0,0]);vectorNear(out.torque,[0,0,0]);
 }
 for(const [z,f,center] of [[.27,0,0],[0,.5,-.135],[-.27,1,-.27]]){
  const out=bodyWater(base({position:[3,0,z],com:[3,0,z],linear:[0,0,0],angular:[0,0,0]}));
  near(out.fraction,f);near(out.center[2],center);near(out.force[2],9.81*3.5*1.015*f);
 }
});

test('principal axial flow uses the small capsule end area; submerged spin cannot propel a body',()=>{
 const input=base({position:[3,0,-2],com:[3,0,-2],angular:[0,0,0]});
 const axial=bodyWater({...input,linear:[0,0,2]}),side=bodyWater({...input,linear:[2,0,0]});
 near(axial.dragForce[2],-.5*1000*.8*Math.PI*.06**2*4);
 near(side.dragForce[0],-.5*1000*.8*(4*.06*.21+Math.PI*.06**2)*4);
 const spin=bodyWater({...input,linear:[0,0,0],angular:[4,-3,9]});
 vectorNear(spin.dragForce,[0,0,0]);assert.ok(dot(spin.dragTorque,[4,-3,9])<0);
});

test('a changed COM reference preserves the physical force and transforms its moment correctly',()=>{
 const input=base(),shift=[.11,-.07,.04],a=bodyWater(input);
 const b=bodyWater({...input,com:add(input.com,shift),linear:add(input.linear,cross(input.angular,shift))});
 vectorNear(a.force,b.force);vectorNear(a.dragForce,b.dragForce);
 vectorNear(b.torque,sub(a.torque,cross(shift,a.force)));
 vectorNear(b.dragTorque,sub(a.dragTorque,cross(shift,a.dragForce)));
 near(a.dragPower,b.dragPower);
});

test('world translation and yaw preserve partial-immersion forces and torques',()=>{
 for(const input of randomCases(48)){
  const a=bodyWater(input),shift=[.2,-.1,4.1];
  const b=bodyWater({...input,position:add(input.position,shift),com:add(input.com,shift),waterZ:input.waterZ+shift[2]});
  vectorNear(a.force,b.force);vectorNear(a.torque,b.torque);
  const yaw=[Math.cos(.31),0,0,Math.sin(.31)],pivot=[3,0,0];
  const point=p=>add(pivot,rot(yaw,sub(p,pivot)));
  const c=bodyWater({...input,position:point(input.position),com:point(input.com),quaternion:product(yaw,input.quaternion),linear:rot(yaw,input.linear),angular:rot(yaw,input.angular)});
  vectorNear(c.force,rot(yaw,a.force));vectorNear(c.torque,rot(yaw,a.torque));near(c.fraction,a.fraction);
 }
});

test('arbitrary rigid rotation preserves fully submerged drag (gravity is excluded)',()=>{
 const input=base({position:[3,0,-3],com:[3,0,-3],quaternion:[.5,.5,.5,.5]}),q=[Math.cos(.47),Math.sin(.47),0,0],a=bodyWater(input);
 const b=bodyWater({...input,quaternion:product(q,input.quaternion),linear:rot(q,input.linear),angular:rot(q,input.angular)});
 vectorNear(b.dragForce,rot(q,a.dragForce));vectorNear(b.dragTorque,rot(q,a.dragTorque));near(a.dragPower,b.dragPower);
});

test('sphere orientation and capsule axial roll cannot change a physically identical shape',()=>{
 const q=[Math.cos(Math.PI/8),0,0,Math.sin(Math.PI/8)];
 for(const type of [2,3]){
  const input=base({type,size:[.13,.2,0],position:[3,0,-2],com:[3,0,-2],linear:[1,0,2],angular:[2,0,1]}),a=bodyWater(input),b=bodyWater({...input,quaternion:q});
  vectorNear(a.dragForce,b.dragForce);vectorNear(a.dragTorque,b.dragTorque);
 }
});
