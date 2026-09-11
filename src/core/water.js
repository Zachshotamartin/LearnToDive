/** Stateless V8 water law; exact arithmetic counterpart of training/water.py.
 * Native coordinates: X forward, Y lateral, Z up. Quaternion order wxyz.
 * Single-body wet-envelope approximation, not fluid simulation/CFD. */
export const POOL_DEPTH = 5;
export const WATER_VERSION = 'directional-submerged-body-v1';
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
export function rotation([w,x,y,z]) { return [1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]; }
const world=(R,v)=>[0,1,2].map(i=>dot(R.slice(i*3,i*3+3),v));
const local=(R,v)=>[0,1,2].map(i=>R[i]*v[0]+R[3+i]*v[1]+R[6+i]*v[2]);
export function shapeParameters(type,size) {
 const r=type===2?[size[0],size[0],size[0]]:type===3?[size[0],size[0],size[0]+size[1]]:[...size];
 const area=[0,1,2].map(i=>(type===6?4:Math.PI)*r[(i+1)%3]*r[(i+2)%3]);
 if(type===3){area[0]=area[1]=4*size[0]*size[1]+Math.PI*size[0]**2;area[2]=Math.PI*size[0]**2;}
 return {r,area};
}
export function immersion(type,size,position,quaternion,waterZ) {
 const R=rotation(quaternion),row=R.slice(6,9);
 const extent=type===2?size[0]:type===3?size[0]+size[1]*Math.abs(row[2]):type===6?dot(row.map(Math.abs),size):Math.hypot(...row.map((v,i)=>v*size[i]));
 const inside=position[0]>=-.44&&position[0]<=11.34&&Math.abs(position[1])<=3.56;
 const fraction=inside?Math.max(0,Math.min(1,(waterZ-position[2]+extent)/Math.max(2*extent,1e-9))):0;
 return {fraction,extent,R};
}
export function bodyWater({type,size,position,quaternion,com,linear,angular,mass,waterZ}) {
 const {fraction,extent,R}=immersion(type,size,position,quaternion,waterZ),{r,area}=shapeParameters(type,size);
 const center=[...position];center[2]-=extent*(1-fraction);
 const offset=center.map((v,i)=>v-com[i]),spin=cross(angular,offset),velocity=linear.map((v,i)=>v+spin[i]);
 const v=local(R,velocity),w=local(R,angular);
 const f=v.map((x,i)=>-.5*1000*.8*area[i]*fraction*x*Math.hypot(...v));
 const t=w.map((x,i)=>-.5*1000*r[i]*(r[(i+1)%3]**4+r[(i+2)%3]**4)*fraction*x*Math.hypot(...w));
 const dragForce=world(R,f),angularDrag=world(R,t),force=[...dragForce];force[2]+=9.81*mass*1.015*fraction;
 const moment=cross(offset,force),dragMoment=cross(offset,dragForce);
 return {force,torque:angularDrag.map((v,i)=>v+moment[i]),dragForce,dragTorque:angularDrag.map((v,i)=>v+dragMoment[i]),dragPower:dot(f,v)+dot(t,w),fraction,extent,area,center};
}
