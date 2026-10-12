// Browser behavior evidence over the actual generated 3D fields.
const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
(async () => {
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH,
    headless: true, args: ['--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1440,height:1100}});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8098/viewer/');
  await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('loaded'));
  const ready = await page.locator('#frame').textContent();
  await page.locator('#timeline').fill('19');
  const scrubbed = await page.locator('#frame').textContent();
  if(!scrubbed.includes('Frame 20 / 120'))throw new Error('timeline did not select frame');
  await page.getByRole('button',{name:'Play simulation',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#timeline').value!=='19');
  await page.getByRole('button',{name:'Pause simulation',exact:true}).click();
  const played = await page.locator('#frame').textContent();
  const canvas=page.locator('#reference');
  const before=await canvas.evaluate(c=>c.toDataURL());
  const box=await canvas.boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);
  await page.mouse.down();await page.mouse.move(box.x+box.width/2+90,box.y+box.height/2+35);await page.mouse.up();
  const after=await canvas.evaluate(c=>c.toDataURL());
  if(before===after)throw new Error('orbit did not change rendered 3D view');
  await page.locator('#timeline').fill('119');
  await page.screenshot({path:'runs/trajectory-v2/viewer-preview.png',fullPage:true});
  const desktopOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  await page.setViewportSize({width:390,height:844});
  const mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  if(errors.length||desktopOverflow||mobileOverflow)throw new Error(JSON.stringify({errors,desktopOverflow,mobileOverflow}));
  fs.writeFileSync('runs/trajectory-v2/viewer-check.json',JSON.stringify({ready,scrubbed,played,
    orbit_changed:true,desktopOverflow,mobileOverflow,errors},null,2)+'\n');
  console.log('3D viewer: load, scrub, playback, orbit, desktop/mobile layout passed');
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
