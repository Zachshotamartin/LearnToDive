"""NumPy batch equivalent of src/core/physics.js; no rendered or analytic policy trajectories."""
import numpy as np
DT=.02
G=9.81
def norm(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
def rotate(q,v):
    uv=np.cross(q[:,:3],v);return v+2*(q[:,3,None]*uv+np.cross(q[:,:3],uv))
def mul(a,b):
    return np.concatenate((a[:,3,None]*b[:,:3]+b[:,3,None]*a[:,:3]+np.cross(a[:,:3],b[:,:3]),(a[:,3]*b[:,3]-np.sum(a[:,:3]*b[:,:3],axis=1))[:,None]),axis=1)
def time_of_flight(p):return (p[:,1]+np.sqrt(p[:,1]**2+2*G*p[:,0]))/G
def features(p):
    v=(1/time_of_flight(p)-.65)*3
    return np.stack((np.ones(len(p)),v,v*v,p[:,3]/.09,p[:,1]-2.5),axis=1)
def decode(raw):return np.stack((np.clip(45+raw[:,0]*18,10,95),np.clip(1.2+raw[:,1]*.9,0,4),np.clip(.5+raw[:,2]*.22,0,.95),np.clip(.58+raw[:,3]*.12,.18,.86)),axis=1)
def actions(weights,p):return decode(features(p)@weights.T)
def baseline(p):return np.stack((12*(np.pi-p[:,3])/time_of_flight(p),np.zeros(len(p)),np.zeros(len(p)),np.full(len(p),.2)),axis=1)
def simulate(p,a,trace=False):
    n=len(p);q=np.zeros((n,4));q[:,2]=np.sin(p[:,3]/2);q[:,3]=np.cos(p[:,3]/2)
    L=rotate(q,np.stack((np.zeros(n),a[:,1],a[:,0]),axis=1));Lhat=norm(L)
    t=np.zeros(n);tuck=np.zeros(n);flips=np.zeros(n);twists=np.zeros(n);phase=np.zeros(n);done=np.zeros(n,dtype=bool);x=np.zeros(n);z=np.zeros(n);y=p[:,0]+.9;vy=p[:,1].copy();vz=np.zeros(n);T=time_of_flight(p)
    history=[]
    for _ in range(176):
        active=~done
        if not np.any(active):break
        tuck+=np.clip(np.where(t/T<a[:,3],a[:,2],0)-tuck,-DT*3.5,DT*3.5)*active
        I=np.stack((12-8.8*tuck,.9+.8*tuck,12-8.8*tuck),axis=1)
        conj=q*np.array([-1,-1,-1,1]);localL=rotate(conj,L)
        precession=np.linalg.norm(L,axis=1)*DT/I[:,0];spin=(1/I[:,1]-1/I[:,0])*localL[:,1]*DT
        world=np.concatenate((Lhat*np.sin(precession[:,None]/2),np.cos(precession[:,None]/2)),axis=1)
        body=np.stack((np.zeros(n),np.sin(spin/2),np.zeros(n),np.cos(spin/2)),axis=1)
        previous=rotate(q,np.tile([0,1,0],(n,1)));newq=norm(mul(mul(world,q),body));q=np.where(active[:,None],newq,q)
        axis=rotate(q,np.tile([0,1,0],(n,1)))
        pa=norm(previous-np.sum(previous*Lhat,axis=1)[:,None]*Lhat);pb=norm(axis-np.sum(axis*Lhat,axis=1)[:,None]*Lhat)
        flips+=np.arctan2(np.sum(Lhat*np.cross(pa,pb),axis=1),np.sum(pa*pb,axis=1))/(2*np.pi)*active
        right=norm(np.cross(axis,Lhat));forward=np.cross(right,axis);bodyright=rotate(q,np.tile([1,0,0],(n,1)))
        newphase=np.arctan2(np.sum(bodyright*forward,axis=1),np.sum(bodyright*right,axis=1))
        twists+=np.arctan2(np.sin(newphase-phase),np.cos(newphase-phase))/(2*np.pi)*active;phase=np.where(active,newphase,phase)
        t+=DT*active;x=1.6*t;z=.5*p[:,2]*t*t;y=p[:,0]+.9+p[:,1]*t-.5*G*t*t;vy=p[:,1]-G*t;vz=p[:,2]*t
        extent=(.9-.42*tuck)*np.abs(axis[:,1])+(.13+.13*tuck)*np.sqrt(np.maximum(0,1-axis[:,1]**2))
        done|=(y<=extent)|(t>=3.5)
        if trace:history.append({'time':float(t[0]),'q':q[0].tolist(),'L':L[0].tolist(),'tuck':float(tuck[0]),'x':float(x[0]),'y':float(y[0]),'z':float(z[0]),'flips':float(flips[0]),'twists':float(twists[0])})
    angle=np.arccos(np.clip(-axis[:,1],-1,1))*180/np.pi;extension=1-tuck
    valid=(y<=extent)&done&(x>.3)&(x<8)&(np.abs(z)<2.8)&(angle<=30)&(extension>=.7)
    alignment=np.maximum(0,np.cos(angle*np.pi/180))**2;horizontal=np.sqrt(1.6**2+vz**2)
    area=.12+.88*np.sin(angle*np.pi/180)**2+.45*tuck;splash=np.clip(area*(.45+.035*np.abs(vy))+.035*horizontal**2,0,1)
    difficulty=np.clip(np.maximum(0,np.abs(flips)-.5)/2.5+np.abs(twists)/2.5,0,1)
    score=valid*(20*difficulty+20*extension+45*alignment+15*(1-splash))
    reward=valid*3+score/35+.25*np.cos(angle*np.pi/180)+.1*extension-.1*splash
    return {'score':score,'valid':valid,'angle':angle,'flips':np.abs(flips),'twists':np.abs(twists),'splash':splash,'reward':reward,'q':q,'L':L,'time':t,'tuck':tuck,'trace':history}
def contexts(rng,n):return np.stack((rng.uniform(2,12,n),rng.uniform(1.5,3.5,n),rng.uniform(-.6,.6,n),rng.uniform(-.09,.09,n)),axis=1)
