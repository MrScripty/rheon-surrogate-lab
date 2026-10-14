// Browser evidence over the actual CPU recipe's newly retained replay.
const fs = require('fs');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
(async () => {
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium',
    headless: true, args: ['--no-sandbox']});
  const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(process.env.VIEWER_URL || 'http://127.0.0.1:8098/viewer/?run=pressure-parity&compute=cpu');
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('loaded'));
  const label = await page.locator('#modelLabel').textContent();
  if (!label.includes('2.8.0+cpu') || !label.includes('CPU inference')) throw new Error('Actual runtime label differs');
  await page.locator('#timeline').fill('19');
  const scrubbed = await page.locator('#frame').textContent();
  if (!scrubbed.includes('Frame 20 / 120')) throw new Error('Timeline did not scrub');
  await page.getByRole('button', {name:'Play simulation', exact:true}).click();
  await page.waitForFunction(() => document.querySelector('#timeline').value !== '19');
  await page.getByRole('button', {name:'Pause simulation', exact:true}).click();
  const canvas = page.locator('#reference');
  const before = await canvas.evaluate(c => c.toDataURL());
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 80, box.y + box.height / 2 + 30);
  await page.mouse.up();
  if (before === await canvas.evaluate(c => c.toDataURL())) throw new Error('Orbit did not change the field view');
  const desktopOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  await page.setViewportSize({width:390, height:844});
  const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  if (errors.length || desktopOverflow || mobileOverflow) throw new Error(JSON.stringify({errors, desktopOverflow, mobileOverflow}));
  const result = {status:'passed', url:page.url(), label, scrubbed, playback:true,
    orbit:true, desktopOverflow, mobileOverflow, errors,
    scope:'Existing lab CPU viewer; no managed Tuldok Run or GPU backend claim'};
  fs.writeFileSync(process.env.VIEWER_RECEIPT || 'runs/tuldok-viewer-check.json', JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(error => {console.error(error); process.exit(1);});
