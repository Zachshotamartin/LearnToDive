import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {chromium} from '@playwright/test';
const port=process.env.DIVE_TEST_PORT||'5197',url=`http://127.0.0.1:${port}`;
let browser;const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','--host','127.0.0.1','--port',port,'--strictPort'],{stdio:'ignore'});
try{
 for(let i=0;i<100;i++){if(await fetch(url).then(r=>r.ok).catch(()=>false))break;await new Promise(r=>setTimeout(r,100));}
 browser=await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(url);await page.locator('[name="autoplay"]').uncheck();
 const actions=[];
 for(const policy of ['pretrained','untrained']){
  await page.locator('[name="policy"]').selectOption(policy);
  await page.locator('.learn-to-dive[data-state="ready"]').waitFor({timeout:45000});
  const pending=page.waitForEvent('download');await page.locator('[data-action="export"]').click();
  const data=JSON.parse(await fs.readFile(await(await pending).path(),'utf8'));
  assert.equal(data.policy,policy);assert.equal(data.physics.contract.observationSize,76);assert.equal(data.physics.contract.actionSize,9);
  assert.equal(data.physics.contract.controlSemantics,'joint-servo-full-entry-v8');
  assert.equal(data.physics.contract.judgeVersion,'full-submersion-physical-entry-v8');
  assert(data.actions.length>10);assert(data.actions.every(row=>row.targets.every(Number.isFinite)));
  assert(Number.isFinite(data.result.execution));actions.push(data.actions);
 }
 assert.notDeepEqual(actions[0],actions[1]);assert.deepEqual(errors,[]);
 console.log('Browser: current trained and initial policies run and export finite joint actions under the v8 runtime contract. No page errors; no model-quality claim.');
}finally{await browser?.close();server.kill();}
