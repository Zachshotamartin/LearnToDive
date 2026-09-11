import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
import {Matrix4,Vector3,Quaternion} from 'three';
import {createStickFigure,NEUTRAL_POSE} from '@zachshotamartin/stick-figure';
test('physical head sphere matches the actual shared rendered head radius and retains mass',()=>{
 const figure=createStickFigure();figure.apply(NEUTRAL_POSE);
 try{
  const joints=figure.root.getObjectByName('RoundEndpoints'),matrix=new Matrix4(),position=new Vector3(),q=new Quaternion(),scale=new Vector3();let radius=null;
  for(let i=0;i<joints.count;i++){joints.getMatrixAt(i,matrix);matrix.decompose(position,q,scale);if(position.distanceTo(new Vector3(...NEUTRAL_POSE.head))<1e-6){assert.ok(Math.abs(scale.x-scale.y)<1e-8&&Math.abs(scale.x-scale.z)<1e-8);radius=scale.x;}}
  assert.notEqual(radius,null);
  const xml=fs.readFileSync(new URL('../public/physics/diver.xml',import.meta.url),'utf8'),head=xml.match(/<body name="head"[\s\S]*?<geom[^>]*>/)[0];
  const physical=Number(head.match(/size="([^"]+)"/)[1]);assert.ok(Math.abs(physical-radius)<1e-7,`physical${physical} / rendered${radius}`);assert.match(head,/mass="5"/);
 }finally{figure.dispose();}
});
