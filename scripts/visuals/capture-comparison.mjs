/** A neutral geometry comparison, explicitly not a learned dive or game rollout. */
import {chromium} from '@playwright/test';
import {spawn} from 'node:child_process';
import {mkdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const output=process.argv[2]||path.join(repo,'output/visual-audit');await mkdir(output,{recursive:true});
const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','--host','127.0.0.1','--port','5197'],{cwd:repo,stdio:'ignore'});await new Promise(r=>setTimeout(r,900));
const browser=await chromium.launch({channel:'chromium'});
try{
 const page=await browser.newPage({viewport:{width:1200,height:900},deviceScaleFactor:1.5});
 await page.route('**/figure-comparison',r=>r.fulfill({contentType:'text/html',body:'<style>body{margin:0;background:#142020}canvas{display:block}</style>'}));
 await page.goto('http://127.0.0.1:5197/figure-comparison');

  // An optional live portfolio URL supplies its actual computed CSS background.
  // Capture the canvas element after browser compositing, without raster edits.
  let captureBackground = '#142020';
  if (process.env.CAPTURE_PAGE_URL) {
    const reference = await browser.newPage();
    await reference.goto(process.env.CAPTURE_PAGE_URL);
    captureBackground = await reference.evaluate(() => getComputedStyle(document.body).background);
    await reference.close();
  }
  for(const [i,match] of [...captureBackground.matchAll(/url\(["']?([^"')]+)["']?\)/g)].entries()) {
    const url = match[1], response = await fetch(url);
    if(!response.ok)throw Error('Background texture fetch failed: '+response.status);
    const body=Buffer.from(await response.arrayBuffer());
    const localURL=new URL('/capture-background-'+i,new URL(page.url()).origin).href;
    await page.route(localURL,r=>r.fulfill({contentType:response.headers.get('content-type')||'image/webp',body}));
    captureBackground=captureBackground.replace(url,localURL);
  }
  await page.evaluate(async background => {
    document.body.style.background = background;
    const urls = [...background.matchAll(/url\(["']?([^"')]+)["']?\)/g)].map(m => m[1]);
    await Promise.all(urls.map(url => new Promise(resolve => {
      const image = new Image(); image.onload = image.onerror = resolve; image.src = url;
    })));
  }, captureBackground);
 await page.evaluate(async()=>{
  const THREE=await import('/node_modules/three/build/three.module.js');
  const {createDiver,NEUTRAL_POSE}=await import('/src/visuals/diver.js');
  const figures=[createDiver(),createDiver()];await Promise.all(figures.map(f=>f.ready));
  figures[1].root.traverse(m=>{if(m.material)m.material.color.set('#d9514c');});
  const pose=NEUTRAL_POSE;
  const scene=new THREE.Scene(),renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});renderer.setSize(1200,900);renderer.setPixelRatio(1.5);renderer.setClearColor(0,0);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;document.body.append(renderer.domElement);
  scene.add(new THREE.HemisphereLight('#edf7ff','#637574',2));const sun=new THREE.DirectionalLight('#fff2d9',3);sun.position.set(3,5,4);scene.add(sun);
  figures.forEach((f,i)=>{f.apply(pose);f.root.position.x=i? .58:-.58;scene.add(f.root);});
  const camera=new THREE.OrthographicCamera(-1.45,1.45,1.0875,-1.0875,.01,50);camera.position.set(0,1.1,5);camera.lookAt(0,.88,0);renderer.render(scene,camera);
  window.cleanup = () => {figures.forEach(f=>f.dispose());renderer.dispose();};
 });
 await page.locator('canvas').screenshot({path:path.join(output,'shared-stick-figure-comparison.png')});await page.evaluate(()=>cleanup());console.log(JSON.stringify({capture:'shared-stick-figure-comparison.png',captureBackground}));
}finally{await browser.close();server.kill();}
