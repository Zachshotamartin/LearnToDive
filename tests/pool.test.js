import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Box3, Scene } from 'three';
import { createPool } from '../src/visuals/pool.js';
import { POOL_DEPTH } from '../src/core/water.js';

test('visible pool floor agrees with the physical basin depth and camera bounds', () => {
  const xml = readFileSync(new URL('../public/physics/diver.xml', import.meta.url), 'utf8');
  const floor = xml.match(/<geom name="pool_floor"[^>]+>/)?.[0];
  assert.ok(floor, 'The physical basin must have an actual floor');
  const position = floor.match(/pos="([^"]+)"/)[1].split(' ').map(Number);
  const size = floor.match(/size="([^"]+)"/)[1].split(' ').map(Number);
  const physicalSurface = position[2] + size[2];
  const scene = new Scene(), pool = createPool(scene);
  try {
    const visible = pool.root.getObjectByName('pool-floor');
    assert.ok(visible);
    const bounds = new Box3().setFromObject(visible);
    assert.ok(Math.abs(bounds.max.y - physicalSurface) < 1e-6);
    assert.ok(Math.abs(bounds.max.y + POOL_DEPTH) < 1e-6);
    assert.ok(pool.bounds.min.y <= bounds.min.y + 1e-6);
    assert.ok(pool.bounds.containsBox(bounds));
  } finally {
    pool.dispose();
  }
  assert.equal(scene.children.length, 0);
});

test('all four physical basin walls reach from the five-meter floor to water level',()=>{
 const xml=readFileSync(new URL('../public/physics/diver.xml',import.meta.url),'utf8');
 for(const name of ['pool_left','pool_right','pool_near','pool_far']){
  const g=xml.match(new RegExp(`<geom name="${name}"[^>]+>`))[0];
  const p=g.match(/pos="([^"]+)"/)[1].split(' ').map(Number),size=g.match(/size="([^"]+)"/)[1].split(' ').map(Number);
  assert.equal(p[2]+size[2],0,name+' must reach water');assert.equal(p[2]-size[2],-POOL_DEPTH,name+' must reach floor');
 }
});
